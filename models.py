from jax._src.api import vmap
import jax.numpy as np
from jax.random import split, normal
from jax.nn import elu

from einops import rearrange, repeat

from equinox import Module
from equinox.nn import Sequential, Conv2d, ConvTranspose2d, Linear, Lambda

from functools import partial

# typing
from jax import Array
from typing import Optional, Sequence, Tuple
from jax.random import PRNGKeyArray


def sample(mu: Array, logvar: Array, shape: Sequence[int], *, key: PRNGKeyArray) -> Array:
    std: Array = np.exp(0.5 * logvar)
    # use the reparameterization trick
    return mu + std * normal(key, shape)


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
                Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), key=keys[0]),
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


class LinearDecoder(Sequential):
    def __init__(self, latent_size: int, *, key: PRNGKeyArray):
        super().__init__(
            [
                Linear(in_features=latent_size, out_features=2048, key=key),
                Lambda(partial(rearrange, pattern='(c h w) -> c h w', h=2, w=2, c=512)),
            ]
        )


class BaselineDecoder(Sequential):
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


class Residual(Module):
    conv1: Conv2d
    conv2: Conv2d

    def __init__(self, latent_size: int, *, key: PRNGKeyArray) -> None:
        key1, key2 = split(key, 2)
        self.conv1 = Conv2d(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=key1)
        self.conv2 = Conv2d(latent_size, latent_size, kernel_size=(1, 1), stride=(1, 1), key=key2)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        res = x
        x = self.conv1(x)
        x = elu(x)
        x = self.conv2(x)
        return x + res


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


class BaselineVAE(Module):
    encoder: Encoder
    linear_decoder: LinearDecoder
    decoder: BaselineDecoder
    latent_size: int

    def __init__(self, latent_size: int = 256, *, key: PRNGKeyArray) -> None:
        key1, key2, key3 = split(key, 3)
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size=latent_size, key=key1)
        self.linear_decoder = LinearDecoder(latent_size=latent_size, key=key2)
        self.decoder = BaselineDecoder(key=key3)

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> Tuple[Array, Array, Array]:
        # get parameters for the latent distribution
        mean, logvar = self.encoder(x)

        # sample from the latent distribution
        z = sample(mean, logvar, mean.shape, key=key)

        # decode the latent sample
        z = self.linear_decoder(z)

        # reconstruct the image
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def center(self) -> Array:
        c = self.linear_decoder(np.zeros(self.latent_size))
        return self.decoder(c)


def crop(x: Array, size: Tuple[int, ...]) -> Array:
    """Crop an image to a given size."""
    c, h, w = size
    ch, cw = x.shape[-2:]
    hh, ww = (h - ch) // 2, (w - cw) // 2
    return x[:c, hh:-hh, ww:-ww]


class DoublingVNCA(Module):
    encoder: Encoder
    step: NCAStep
    double: Lambda
    latent_size: int
    K: int
    N_nca_steps: int

    def __init__(self, latent_size: int = 256, K: int = 5, N_nca_steps: int = 8, *, key: PRNGKeyArray) -> None:
        key1, key2 = split(key)
        self.encoder = Encoder(latent_size=latent_size, key=key1)
        self.step = NCAStep(latent_size=latent_size, key=key2)
        self.double = Double
        self.latent_size = latent_size
        self.K = K
        self.N_nca_steps = N_nca_steps

    def __call__(self, x: Array, *, key: PRNGKeyArray, M: int = 1) -> Tuple[Array, Array, Array]:
        # get parameters for the latent distribution
        mean, logvar = self.encoder(x)

        # sample from the latent distribution M times
        z = sample(mean, logvar, (M, *mean.shape), key=key)

        # Add height and width dimensions
        z = rearrange(z, 'M c -> M c 1 1')
        print(z.shape)
        # vmap over the M samples
        z = vmap(self.decoder)(z)

        # resize the image to the original size
        # z = vmap(crop, in_axes=(0, None))(z, x.shape)

        return z, mean, logvar

    def decoder(self, z: Array) -> Array:
        for _ in range(self.K):
            z = self.double(z)
            for _ in range(self.N_nca_steps):
                z = z + self.step(z)
        return z


class NonDoublingVNCA(Module):
    encoder: Encoder
    step: NCAStep
    latent_size: int
    N_nca_steps: int

    def __init__(self, latent_size: int = 256, N_nca_steps: int = 8, *, key: PRNGKeyArray) -> None:
        key1, key2 = split(key)
        self.encoder = Encoder(latent_size=latent_size, key=key1)
        self.step = NCAStep(latent_size=latent_size, key=key2)
        self.latent_size = latent_size
        self.N_nca_steps = N_nca_steps

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> Tuple[Array, Array, Array]:
        # get parameters for the latent distribution
        mean, logvar = self.encoder(x)

        # sample from the latent distribution
        z_0 = sample(mean, logvar, mean.shape, key=key)
        z = repeat(z_0, 'c -> c h w', h=28, w=28)

        # run the NCA steps
        for _ in range(self.N_nca_steps):
            z = z + self.step(z)

        return z, mean, logvar
