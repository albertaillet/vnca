# %% Imports

import jax.numpy as np
from jax.random import PRNGKey, split, normal
from jax.nn import elu
from einops import rearrange, reduce
from optax import adam, exponential_decay

import equinox as eqx
from equinox import Module
from equinox.nn import Sequential, Conv2d, ConvTranspose2d, Linear, Lambda

import io
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# typing
from jax import Array
from typing import Optional, Any
from jax.random import PRNGKeyArray
from optax import GradientTransformation

TARGET_SIZE = 28
LATENT_SIZE = 16

# %% Define the neural nets


def flatten(x: Array) -> Array:
    return rearrange(x, 'c h w -> (c h w)')


Flatten: Module = Lambda(flatten)


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
                #Elu,
                ConvTranspose2d(256, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[1]),
                #Elu,
                ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[2]),
                #Elu,
                ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[3]),
                #Elu,
                ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), key=keys[4]),
            ]
        )


class Residual(Module):
    conv1: Conv2d
    conv2: Conv2d

    def __init__(self, key: PRNGKeyArray) -> None:
        key1, key2 = split(key, 2)
        self.conv1 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), key=key1)
        self.conv2 = Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), key=key2)

    def __call__(self, x: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        res = x
        x = self.conv1(x)
        x = elu(x)
        x = self.conv2(x)
        return x + res


class VNCADecoder(Sequential):
    def __init__(self, key: PRNGKeyArray) -> None:
        keys = split(key, 6)
        super().__init__(
            [
                Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), key=keys[0]),
                Residual(key=keys[1]),
                Residual(key=keys[2]),
                Residual(key=keys[3]),
                Residual(key=keys[4]),
                Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), key=keys[5]),
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

    def __call__(self, x: Array, key: PRNGKeyArray) -> tuple[Array, Array, Array]:
        # get paramets for the latent distribution
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
        c = self.linear_decoder(np.zeros((128)))

        c = rearrange(c, '(c h w) -> c h w', h=2, w=2, c=512)
        return self.decoder(c)


def sample(key: PRNGKeyArray, mu: Array, logvar: Array) -> Array:
    std: Array = np.exp(0.5 * logvar)
    # use the reparameterization trick
    return mu + std * normal(key, mu.shape)


@eqx.filter_value_and_grad
def loss_fn(model: Module, x: Array, key: PRNGKeyArray) -> float:
    recon_x, mean, logvar = model(x, key)
    recon_loss = np.mean(np.square(recon_x - x))
    kl_loss = -0.5 * np.mean(1 + logvar - np.square(mean) - np.exp(logvar))
    return recon_loss + kl_loss


@eqx.filter_jit
def make_step(model: Module, x: Array, key: PRNGKeyArray, opt_state: tuple, optim: GradientTransformation) -> tuple[float, Module, Any]:
    loss, grads = loss_fn(model, x, key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def make_seed(size: int, latent: int = LATENT_SIZE):
    x = np.zeros([size, size, latent], np.float32)
    x = x.at[size // 2, size // 2, 3:].set(1.0)
    return x


def load_image(url: str, target_size: int = TARGET_SIZE) -> Array:
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    img = img.resize((target_size, target_size))
    img = np.float32(img) / 255.0
    img = img.at[..., :3].set(img[..., :3] * img[..., 3:])
    img = reduce(img, 'h w c -> h w', 'mean')
    return img


def load_emoji(emoji: str) -> Array:
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
    return load_image(url)


# %% Load the image
img = load_emoji('🌗')
x = rearrange(img, 'h w -> 1 h w')
# plt.imshow(img, cmap='gray')
# plt.show()

# %% Create the model
model_key = PRNGKey(0)
vae = BaselineVAE(key=model_key)

# plt.imshow(vae.center()[0], cmap='gray')
# plt.show()

# %% Call the model
# recon_x, mean, logvar = vae(x, PRNGKey(0))
# plt.imshow(recon_x[0], cmap='gray')

# %% Train the model
# lr = 3e-5
lr = exponential_decay(3e-5, 60, 0.1, staircase=True)
opt = adam(lr)
opt_state = opt.init(eqx.filter(vae, eqx.is_array))

n_gradient_steps = 100
for key in split(model_key, n_gradient_steps):
    loss, vae, opt_state = make_step(vae, x, key, opt_state, opt)
    print(loss)
# %%

# It has trained but not very good

plt.imshow(vae(x, PRNGKey(0))[0][0], cmap='gray')
plt.show()
plt.imshow(vae.center()[0], cmap='gray')
plt.show()
