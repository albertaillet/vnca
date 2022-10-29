# %% Create Neural Cellular Automata using JAX

import jax.numpy as np
from jax import jit, vmap, value_and_grad
from jax.random import PRNGKey, split, normal
from jax.nn import elu
from einops import rearrange, reduce, repeat
from optax import adam, apply_updates

from equinox import Module
from equinox.nn import Sequential, Conv2d, ConvTranspose2d, Linear, Lambda

import io
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# typing
from jax.numpy import ndarray
from typing import Optional

TARGET_SIZE = 28
LATENT_SIZE = 16

# %% Define the neural nets

Flatten = Lambda(lambda x: rearrange(x, 'c h w -> (c h w)'))

def debug(x: ndarray) -> ndarray:
  print(x.shape)
  return x

Debug = Lambda(debug)

Elu = Lambda(elu)


class Encoder(Sequential):
  def __init__(self, key):
    keys = split(key, 6)
    super().__init__([
      Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), key=keys[0]),
      Elu,
      Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[1]),
      Elu,
      Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[2]),
      Elu,
      Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[3]),
      Elu,
      Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), key=keys[4]),
      Elu,
      Flatten,
      Linear(in_features=2048, out_features=256, key=keys[5]),
    ])


class LinearDecoder(Linear):
  def __init__(self, key):
    super().__init__(in_features=128, out_features=2048, key=key)


class BaselineDecoder(Sequential):
  def __init__(self, key):
    keys = split(key, 5)
    super().__init__([
      ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[0]),
      Elu,
      ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[1]),
      Elu,
      ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[2]),
      Elu,
      ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1), key=keys[3]),
      Elu,
      ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), key=keys[4]),
    ])


class Residual(Module):
  conv1: Conv2d
  conv2: Conv2d

  def __init__(self, key: PRNGKey) -> None:
    key1, key2 = split(key, 2)
    self.conv1 = Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), key=key1)
    self.conv2 = Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), key=key2)

  def __call__(self, x: ndarray, key: Optional[PRNGKey]=None) -> ndarray:
    res = x
    x = self.conv1(x)
    x = elu(x)
    x = self.conv2(x)
    return x + res


class VNCADecoder(Sequential):
  def __init__(self, key: PRNGKey) -> None:
    keys = split(key, 6)
    super().__init__([
      Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), key=keys[0]),
      Residual(key=keys[1]),
      Residual(key=keys[2]),
      Residual(key=keys[3]),
      Residual(key=keys[4]),
      Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), key=keys[5]),
    ])


class BaselineVAE(Module):
  encoder: Encoder
  linear_decoder: LinearDecoder
  decoder: BaselineDecoder

  def __init__(self, key: PRNGKey) -> None:
    key1, key2, key3 = split(key, 3)
    self.encoder = Encoder(key=key1)
    self.linear_decoder = LinearDecoder(key=key2)
    self.decoder = BaselineDecoder(key=key3)

  def __call__(self, x: ndarray, key: PRNGKey) -> tuple[ndarray]:
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

def sample(key: PRNGKey, mu: ndarray, logvar: ndarray) -> ndarray:
  std = np.exp(0.5 * logvar)
  # use the reparameterization trick
  return mu + std * normal(key, mu.shape)

def loss_fn(model: Module, x: ndarray, key: PRNGKey) -> ndarray:
  recon_x, mean, logvar = model(x, key)
  recon_loss = np.mean(np.square(recon_x - x))
  kl_loss = -0.5 * np.mean(1 + logvar - np.square(mean) - np.exp(logvar))
  return recon_loss + kl_loss

def make_seed(size: int, latent: int=LATENT_SIZE):
  x = np.zeros([size, size, latent], np.float32)
  x = x.at[size//2, size//2, 3:].set(1.0)
  return x

def load_image(url: str, target_size: int=TARGET_SIZE) -> ndarray:
  r = requests.get(url)
  img = Image.open(io.BytesIO(r.content))
  img = img.resize((target_size, target_size))
  img = np.float32(img)/255.0
  img = img.at[..., :3].set(img[..., :3] * img[..., 3:])
  img = reduce(img, 'h w c -> h w 1', 'mean')
  return img

def load_emoji(emoji: str) -> str:
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url)

# %% Load the image
img = load_emoji('🌗')
plt.imshow(img, cmap='gray')
plt.show()

# %% Train the model
vae = BaselineVAE(PRNGKey(0))

# %%
loss_fn(vae, rearrange(img, 'h w c -> c h w'), PRNGKey(0))
# %% Get a table of the model
#loss_fn.en(vae, img, PRNGKey(0))
