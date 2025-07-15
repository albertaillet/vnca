from jax import numpy as np
from jax import lax, vmap, checkpoint
from jax.random import split, normal, bernoulli, randint
from jax.nn import elu, sigmoid

from einops import rearrange, repeat

from equinox import Module
from equinox.nn import Sequential, Conv2d, ConvTranspose2d, Linear, Lambda

from functools import partial

# typing
from jax import Array
from typing import Optional, Sequence, Tuple, Any
    

def sample_gaussian(mu: Array, logvar: Array, shape: Sequence[int], *, key: Array) -> Array:
    std: Array = np.exp(0.5 * logvar)
    # use the reparameterization trick
    return mu + std * normal(key=key, shape=shape)


def sample_bernoulli(logits: Array, shape: Sequence[int], *, key: Array) -> Array:
    p = sigmoid(logits)
    return bernoulli(key=key, p=p, shape=shape)


def crop(x: Array, shape: Tuple[int, int, int]) -> Array:
    '''Crop an image to a given size.'''
    c, h, w = shape
    ch, cw = x.shape[-2:]
    hh, ww = (h - ch) // 2, (w - cw) // 2
    return x[:c, hh : h - hh, ww : w - ww]


def pad(x: Array, p: int, c: float) -> Array:
    '''Pad an image of shape (c, h, w) with contant c.'''
    return np.pad(x, ((0, 0), (p, p), (p, p)), mode='constant', constant_values=c)


def flatten(x: Array) -> Array:
    return rearrange(x, 'c h w -> (c h w)')


def damage(x: Array, *, key: Array) -> Array:
    '''Set the cell states of a H//2 x W//2 square to zero.'''
    l, h, w = x.shape
    h_half, w_half = h // 2, w // 2
    hmask, wmask = randint(
        key=key,
        shape=(2,),
        minval=np.zeros(2, dtype=np.int32),
        maxval=np.array([h_half, w_half], dtype=np.int32),
    )
    update = np.zeros((l, h_half, w_half), dtype=np.float32)
    return lax.dynamic_update_slice(x, update, (0, hmask, wmask))


def double(x: Array) -> Array:
    return repeat(x, 'c h w -> c (h 2) (w 2)')


Flatten: Lambda = Lambda(flatten)

Sigmoid: Lambda = Lambda(sigmoid)

Double: Lambda = Lambda(double)

Elu: Lambda = Lambda(elu)


class Encoder(Sequential):
    def __init__(self, latent_size: int, *, key: Array):
        keys = split(key, 6)
        super().__init__(
            [
                Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), key=keys[0]),
                Elu,
                Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[1]),
                Elu,
                Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[2]),
                Elu,
                Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[3]),
                Elu,
                Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[4]),
                Elu,
                Flatten,
                Linear(in_features=2048, out_features=2 * latent_size, key=keys[5]),
                Lambda(partial(rearrange, pattern='(p l) -> p l', p=2, l=latent_size)),
            ]
        )


class LinearDecoder(Linear):
    def __init__(self, latent_size: int, *, key: Array):
        super().__init__(in_features=latent_size, out_features=2048, key=key)


class ConvolutionalDecoder(Sequential):
    def __init__(self, *, key: Array):
        keys = split(key, 5)
        super().__init__(
            [
                ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[0]),
                Elu,
                ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[1]),
                Elu,
                ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[2]),
                Elu,
                ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[3]),
                Elu,
                ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), key=keys[4]),
            ]
        )


class BaselineDecoder(Sequential):
    def __init__(self, latent_size: int, *, key: Array):
        key1, key2 = split(key, 2)
        super().__init__(
            [
                LinearDecoder(latent_size, key=key1),
                Lambda(partial(rearrange, pattern='(c h w) -> c h w', h=2, w=2, c=512)),  # reshape from 2048 to 512x2x2
                ConvolutionalDecoder(key=key2),
                Lambda(partial(pad, p=2, c=float('-inf'))),  # pad from 28x28 to 32x32
            ]
        )


class Residual(Sequential):
    def __init__(self, latent_size: int, *, key: Array) -> None:
        key1, key2 = split(key, 2)
        super().__init__(
            [
                Conv2d(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=key1),
                Elu,
                Conv2d(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=key2),
            ]
        )

    def __call__(self, x: Array, *, key: Optional[Array] = None) -> Array:
        return x + super().__call__(x, key=key)


class Conv2dZeroInit(Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = np.zeros_like(self.weight)
        self.bias = np.zeros_like(self.bias)


class NCAStep(Sequential):
    def __init__(self, latent_size: int, *, key: Array) -> None:
        keys = split(key, 6)
        super().__init__(
            [
                Conv2d(latent_size, latent_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), key=keys[0]),
                Residual(latent_size=latent_size, key=keys[1]),
                Residual(latent_size=latent_size, key=keys[2]),
                Residual(latent_size=latent_size, key=keys[3]),
                Residual(latent_size=latent_size, key=keys[4]),
                Conv2dZeroInit(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=keys[5]),
            ]
        )


class NCAStepSimple(Sequential):
    def __init__(self, latent_size: int, *, key: Array) -> None:
        key1, key2 = split(key)
        super().__init__(
            [
                Conv2d(latent_size, latent_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), key=key1),
                Elu,
                Conv2dZeroInit(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=key2),
            ]
        )


class AutoEncoder(Module):
    latent_size: int

    def __call__(self, x: Array, *, key: Array, M: int = 1) -> Tuple[Array, Array, Array, Array]:
        # get parameters for the latent distribution
        mean, logvar = self.encoder(x)

        # sample from the latent distribution M times
        z = sample_gaussian(mean, logvar, (M, *mean.shape), key=key)

        # vmap over the M samples and reconstruct the M images
        x_hat = vmap(self.decoder)(z)

        # vmap over the M samples and crop the images to the original size
        x_hat = vmap(partial(crop, shape=x.shape))(x_hat)

        return x_hat, z, mean, logvar

    def encoder(self, x: Array) -> Array:
        raise NotImplementedError

    def decoder(self, z: Array) -> Array:
        raise NotImplementedError

    def center(self) -> Array:
        z_center = np.zeros((self.latent_size))
        return self.decoder(z_center)

    def sample(self, *, key: Array) -> Array:
        mean = np.zeros(self.latent_size)
        logvar = np.zeros(self.latent_size)
        z = sample_gaussian(mean, logvar, shape=(self.latent_size,), key=key)
        return self.decoder(z)


class BaselineVAE(AutoEncoder):
    encoder: Encoder
    decoder: Sequential
    latent_size: int

    def __init__(self, latent_size: int = 256, *, key: Array) -> None:
        key1, key2 = split(key, 2)
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size=latent_size, key=key1)
        self.decoder = BaselineDecoder(latent_size=latent_size, key=key2)


class DoublingVNCA(AutoEncoder):
    encoder: Encoder
    step: NCAStep
    double: Lambda
    latent_size: int
    K: int
    N_nca_steps: int

    def __init__(self, latent_size: int = 256, K: int = 5, N_nca_steps: int = 8, *, key: Array) -> None:
        key1, key2 = split(key)
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size=latent_size, key=key1)
        self.step = NCAStep(latent_size=latent_size, key=key2)
        self.double = Double
        self.K = K
        self.N_nca_steps = N_nca_steps

    def decoder(self, z: Array) -> Array:
        # Add height and width dimensions
        z = rearrange(z, 'c -> c 1 1')

        # Apply the Doubling and NCA steps
        for _ in range(self.K):
            z = self.double(z)
            for _ in range(self.N_nca_steps):
                z = z + self.step(z)
        return z

    def growth_stages(self, n_channels: int = 1, *, key: Array) -> Array:
        mean = np.zeros(self.latent_size)
        logvar = np.zeros(self.latent_size)
        z = sample_gaussian(mean, logvar, (self.latent_size,), key=key)

        # Add height and width dimensions
        z = rearrange(z, 'c -> c 1 1')

        def process(z: Array) -> Array:
            '''Process a latent sample by taking the image channels, applying sigmoid and padding.'''
            logits = z[:n_channels]
            probs = sigmoid(logits)
            pad_size = ((2 ** (self.K)) - probs.shape[1]) // 2
            return pad(probs, p=pad_size, c=0.0)

        # Decode the latent sample and save the processed image channels
        stages_probs = []
        for _ in range(self.K):
            z = self.double(z)
            stages_probs.append(process(z))
            for _ in range(self.N_nca_steps):
                z = z + self.step(z)
                stages_probs.append(process(z))

        return np.array(stages_probs)


class NonDoublingVNCA(AutoEncoder):
    encoder: Encoder
    step: NCAStepSimple
    latent_size: int
    N_nca_steps: int
    N_nca_steps_min: int
    N_nca_steps_max: int

    def __init__(self, latent_size: int = 256, N_nca_steps: int = 36, N_nca_steps_min: int = 32, N_nca_steps_max: int = 64, *, key: Array) -> None:
        key1, key2 = split(key)
        self.encoder = Encoder(latent_size=latent_size, key=key1)
        self.step = NCAStepSimple(latent_size=latent_size, key=key2)
        self.latent_size = latent_size
        self.N_nca_steps = N_nca_steps
        self.N_nca_steps_min = N_nca_steps_min
        self.N_nca_steps_max = N_nca_steps_max

    def decoder(self, z: Array) -> Array:
        # repeat the latent sample over the image dimensions
        z = repeat(z, 'c -> c h w', h=32, w=32)

        # decode the latent sample by applying the NCA steps
        return self.decode_grid(z, T=self.N_nca_steps)

    def decode_grid_random(self, z: Array, *, key: Array) -> Array:
        T = randint(key, shape=(1,), minval=self.N_nca_steps_min, maxval=self.N_nca_steps_max)
        return self.decode_grid(z, T=T)

    def decode_grid(self, z: Array, T: Array) -> Array:
        # Apply the NCA steps
        true_fun = lambda z: z + self.step(z)
        false_fun = lambda z: z

        @checkpoint
        def scan_fn(z: Array, t: int) -> Tuple[Array, Any]:
            z = lax.cond(t, true_fun, false_fun, z)
            return z, None

        z, _ = lax.scan(scan_fn, z, (np.arange(self.N_nca_steps_max) < T))

        return z

    def nca_stages(self, n_channels: int = 1, T: int = 36, damage_idx: set = set(), *, key: Array) -> Array:
        mean = np.zeros(self.latent_size)
        logvar = np.zeros(self.latent_size)
        z = sample_gaussian(mean, logvar, (self.latent_size,), key=key)
        z = repeat(z, 'c -> c h w', h=32, w=32)

        # Decode the latent sample and save the processed image channels
        stages_probs = []
        for i in range(T):
            z = z + self.step(z)
            if i in damage_idx:
                key, damage_key = split(key)
                z = damage(z, key=damage_key)
            stages_probs.append(sigmoid(z[:n_channels]))

        return np.array(stages_probs)
