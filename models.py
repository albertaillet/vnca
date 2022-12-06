from jax import vmap
import jax.numpy as np
from jax.random import split, normal, bernoulli
from jax.nn import elu, sigmoid

from einops import rearrange, repeat

from equinox import Module
from equinox.nn import Sequential, Conv2d, ConvTranspose2d, Linear, Lambda

from functools import partial

# typing
from jax import Array
from typing import Optional, Sequence, Tuple
from jax.random import PRNGKeyArray


def sample_gaussian(mu: Array, logvar: Array, shape: Sequence[int], *, key: PRNGKeyArray) -> Array:
    std: Array = np.exp(0.5 * logvar)
    # use the reparameterization trick
    return mu + std * normal(key=key, shape=shape)


def sample_bernoulli(logits: Array, shape: Sequence[int], *, key: PRNGKeyArray) -> Array:
    p = sigmoid(logits)
    return bernoulli(key=key, p=p, shape=shape)


def crop(x: Array, shape: Tuple[int, int, int]) -> Array:
    '''Crop an image to a given size.'''
    c, h, w = shape
    ch, cw = x.shape[-2:]
    hh, ww = (h - ch) // 2, (w - cw) // 2
    return x[:c, hh : h - hh, ww : w - ww]


def pad(x: Array, p: int) -> Array:
    '''Pad an image of shape (c, h, w) with zeros.'''
    return np.pad(x, ((0, 0), (p, p), (p, p)), mode='constant', constant_values=0)


def flatten(x: Array) -> Array:
    return rearrange(x, 'c h w -> (c h w)')


Flatten: Lambda = Lambda(flatten)


def double(x: Array) -> Array:
    return repeat(x, 'c h w -> c (h 2) (w 2)')


Double: Lambda = Lambda(double)


Elu: Lambda = Lambda(elu)


class Encoder(Sequential):
    def __init__(self, latent_size: int, *, key: PRNGKeyArray):
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
                Lambda(partial(rearrange, pattern='(p l) -> p l', l=latent_size, p=2)),
            ]
        )


class LinearDecoder(Linear):
    def __init__(self, latent_size: int, *, key: PRNGKeyArray):
        super().__init__(in_features=latent_size, out_features=2048, key=key)


class ConvolutionalDecoder(Sequential):
    def __init__(self, *, key: PRNGKeyArray):
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
    def __init__(self, latent_size: int, *, key: PRNGKeyArray):
        key1, key2 = split(key, 2)
        super().__init__(
            [
                LinearDecoder(latent_size, key=key1),
                Lambda(partial(rearrange, pattern='(c h w) -> c h w', h=2, w=2, c=512)),  # reshape from 2048 to 512x2x2
                ConvolutionalDecoder(key=key2),
                Lambda(partial(pad, p=2)),  # pad from 28x28 to 32x32
            ]
        )


class Residual(Sequential):
    def __init__(self, latent_size: int, *, key: PRNGKeyArray) -> None:
        key1, key2 = split(key, 2)
        super().__init__(
            [
                Conv2d(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=key1),
                Elu,
                Conv2d(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=key2),
            ]
        )

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        return x + super().__call__(x, key=key)


class Conv2dZeroInit(Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = np.zeros_like(self.weight)
        self.bias = np.zeros_like(self.bias)


class NCAStep(Sequential):
    def __init__(self, latent_size: int, *, key: PRNGKeyArray) -> None:
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


class AutoEncoder(Module):
    def __call__(self, x: Array, *, key: PRNGKeyArray, M: int = 1) -> Tuple[Array, Array, Array]:
        # get parameters for the latent distribution
        mean, logvar = self.encoder(x)

        # sample from the latent distribution M times
        z = sample_gaussian(mean, logvar, (M, *mean.shape), key=key)

        # vmap over the M samples and reconstruct the M images
        x_hat = vmap(self.decoder)(z)

        # vmap over the M samples and crop the images to the original size
        x_hat = vmap(partial(crop, shape=x.shape))(x_hat)

        return x_hat, mean, logvar

    def encoder(self, x: Array) -> Array:
        raise NotImplementedError

    def decoder(self, z: Array) -> Array:
        raise NotImplementedError

    def center(self) -> Array:
        z_center = np.zeros((self.latent_size))
        return self.decoder(z_center)

    def sample(self, *, key: PRNGKeyArray) -> Array:
        mean = np.zeros(self.latent_size)
        logvar = np.zeros(self.latent_size)
        z = sample_gaussian(mean=mean, logvar=logvar, shape=(self.latent_size,), key=key)
        return self.decoder(z)


class BaselineVAE(AutoEncoder):
    encoder: Encoder
    decoder: Sequential
    latent_size: int

    def __init__(self, latent_size: int = 256, *, key: PRNGKeyArray) -> None:
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

    def __init__(self, latent_size: int = 256, K: int = 5, N_nca_steps: int = 8, *, key: PRNGKeyArray) -> None:
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

    def growth_stages(self, n_channels: int = 1, *, key: PRNGKeyArray) -> Array:
        z = sample_gaussian(np.zeros(self.latent_size), np.zeros(self.latent_size), (self.latent_size,), key=key)
        
        # Add height and width dimensions
        z = rearrange(z, 'c -> c 1 1')

        def process(z: Array) -> Array:
            '''Process a latent sample by taking the image channels, applying sigmoid and padding.'''
            logits = z[:n_channels]
            probs = sigmoid(logits)
            pad_size = ((2 ** (self.K)) - probs.shape[1]) // 2
            return pad(probs, p=pad_size)

        # Decode the latent sample and save the processed image channels
        stages_probs = []
        for _ in range(self.K):
            z = self.double(z)
            stages_probs.append(process(z))
            for i in range(self.N_nca_steps):
                z = z + self.step(z)
                stages_probs.append(process(z))

        return np.array(stages_probs)


class NonDoublingVNCA(AutoEncoder):
    encoder: Encoder
    step: NCAStep
    latent_size: int
    N_nca_steps: int

    def __init__(self, latent_size: int = 256, N_nca_steps: int = 36, *, key: PRNGKeyArray) -> None:
        key1, key2 = split(key)
        self.encoder = Encoder(latent_size=latent_size, key=key1)
        self.step = NCAStep(latent_size=latent_size, key=key2)
        self.latent_size = latent_size
        self.N_nca_steps = N_nca_steps

    def decoder(self, z: Array) -> Array:
        # repeat the latent sample over the image dimensions
        z = repeat(z, 'c -> c h w', h=32, w=32)

        # Apply the NCA steps
        for _ in range(self.N_nca_steps):
            z = z + self.step(z)
        return z

    def nca_stages(self, n_channels: int = 1, *, key: PRNGKeyArray) -> Array:
        z = sample_gaussian(np.zeros(self.latent_size), np.zeros(self.latent_size), (self.latent_size,), key=key)
        z = repeat(z, 'c -> c h w', h=32, w=32)
 
        def process(z: Array) -> Array:
            '''Process a latent sample by taking the image channels, applying sigmoid.'''
            return sigmoid(z[:n_channels])
 
        # Decode the latent sample and save the processed image channels
        stages_probs = []
        for _ in range(self.N_nca_steps):
            z = z + self.step(z)
            stages_probs.append(process(z))
 
        return np.array(stages_probs)
