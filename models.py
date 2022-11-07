import jax.numpy as np
from jax.random import split, normal
from jax.nn import elu
from einops import rearrange, repeat

from equinox import Module
from equinox.nn import Sequential, Conv2d, ConvTranspose2d, Linear, Lambda


# typing
from jax import Array
from typing import Optional, Tuple
from jax.random import PRNGKeyArray


def flatten(x: Array) -> Array:
    return rearrange(x, 'c h w -> (c h w)')


Flatten: Module = Lambda(flatten)


def double(x: Array) -> Array:
    return repeat(x, 'c h w -> c (h 2) (w 2)')


Double: Module = Lambda(double)


Elu: Module = Lambda(elu)


class Encoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
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
                Linear(in_features=2048, out_features=256, key=keys[5]),
            ]
        )


class LinearDecoder(Linear):
    def __init__(self, key: PRNGKeyArray):
        super().__init__(in_features=128, out_features=2048, key=key)


class BaselineDecoder(Sequential):
    def __init__(self, key: PRNGKeyArray):
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

    def __init__(self, key: PRNGKeyArray) -> None:
        key1, key2 = split(key, 2)
        self.conv1 = Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), key=key1)
        self.conv2 = Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), key=key2)

    def __call__(self, x: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        res = x
        x = self.conv1(x)
        x = elu(x)
        x = self.conv2(x)
        return x + res


class NCAStep(Sequential):
    def __init__(self, key: PRNGKeyArray) -> None:
        keys = split(key, 6)
        super().__init__(
            [
                Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), key=keys[0]),
                Residual(key=keys[1]),
                Residual(key=keys[2]),
                Residual(key=keys[3]),
                Residual(key=keys[4]),
                Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), key=keys[5]),
            ]
        )


class BaselineVAE(Module):
    encoder: Encoder
    linear_decoder: LinearDecoder
    decoder: BaselineDecoder

    def __init__(self, key: PRNGKeyArray) -> None:
        key1, key2, key3 = split(key, 3)
        self.encoder = Encoder(key=key1)
        self.linear_decoder = LinearDecoder(key=key2)
        self.decoder = BaselineDecoder(key=key3)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Tuple[Array, Array, Array]:
        # get parameters for the latent distribution
        z_params = self.encoder(x)

        # sample from the latent distribution
        mean, logvar = rearrange(z_params, '(c p) -> p c', c=128, p=2)
        z = sample(key, mean, logvar)

        # decode the latent sample
        z = self.linear_decoder(z)
        z = rearrange(z, '(c h w) -> c h w', h=2, w=2, c=512)

        # reconstruct the image
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def center(self) -> Array:
        c = self.linear_decoder(np.zeros(128))
        c = rearrange(c, '(c h w) -> c h w', h=2, w=2, c=512)
        return self.decoder(c)


class DoublingVNCA(Module):
    encoder: Encoder
    step: NCAStep
    double: Double

    def __init__(self, key: PRNGKeyArray, K: int = 4, N_nca_steps: int = 9) -> None:
        key1, key2 = split(key)
        self.encoder = Encoder(key=key1)
        self.step = NCAStep(key=key2)
        self.double = Double
        self.K = K
        self.N_nca_steps = N_nca_steps

    def __call__(self, x: Array, key: PRNGKeyArray) -> Tuple[Array, Array, Array]:
        # get parameters for the latent distribution
        z_params = self.encoder(x)

        # sample from the latent distribution
        mean, logvar = rearrange(z_params, '(c p) -> p c', c=128, p=2)
        z = sample(key, mean, logvar)
        z = rearrange(z, 'c -> c 1 1')

        # run the doubling and NCA steps
        z = self.double(z)
        for _ in range(self.K):
            for _ in range(self.N_nca_steps):
                z = z + self.step(z)
            z = self.double(z)

        return z, mean, logvar


def sample(key: PRNGKeyArray, mu: Array, logvar: Array) -> Array:
    std: Array = np.exp(0.5 * logvar)
    # use the reparameterization trick
    return mu + std * normal(key, mu.shape)
