# %% Create Neural Cellular Automata using JAX

import jax.numpy as np
from jax import jit, vmap, value_and_grad
from jax import lax, random
from jax.nn import elu
from einops import rearrange, reduce, repeat
from optax import adam, apply_updates

import haiku as hk
from haiku import Module, Sequential, Conv2D, Conv2DTranspose, Linear, transform

import io
import optax
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dataclasses import dataclass

# typing
from jax.numpy import ndarray

TARGET_SIZE = 32
LATENT_SIZE = 16

# %% Define the neural cellular automata

class Flatten(Module):
  def __call__(self, x: ndarray) -> ndarray:
    return rearrange(x, 'b h w c -> b (h w c)')
  

class Elu(Module):
  def __call__(self, x: ndarray) -> ndarray:
    return elu(x, alpha=1.0)


class Encoder(Module):
  def __call__(self, x: ndarray) -> ndarray:
    return Sequential(
      [
        Conv2D(output_channels=32, kernel_shape=(5, 5), stride=(1, 1), padding=(2, 2)),
        Elu(),
        Conv2D(output_channels=64, kernel_shape=(5, 5), stride=(2, 2), padding=(2, 2)),
        Elu(),
        Conv2D(output_channels=128, kernel_shape=(5, 5), stride=(2, 2), padding=(2, 2)),
        Elu(),
        Conv2D(output_channels=256, kernel_shape=(5, 5), stride=(2, 2), padding=(2, 2)),
        Elu(),
        Conv2D(output_channels=512, kernel_shape=(5, 5), stride=(2, 2), padding=(2, 2)),
        Elu(),
        Flatten(),
        Linear(output_size=256),
      ]
    )(x)


class LinearDecoder(Module):
  def __call__(self, z: ndarray) -> ndarray:
    return Linear(2048)(z)


class BaselineDecoder(Module):
  def __call__(self, z: ndarray) -> ndarray:
    return Sequential(
      [
        Conv2DTranspose(output_channels=256, kernel_shape=(5, 5), stride=(2, 2), padding=((2, 2), (2, 2))),
        Elu(),
        Conv2DTranspose(output_channels=128, kernel_shape=(5, 5), stride=(2, 2), padding=((2, 2), (2, 2))),
        Elu(),
        Conv2DTranspose(output_channels=64, kernel_shape=(5, 5), stride=(2, 2), padding=((2, 2), (2, 2))),
        Elu(),
        Conv2DTranspose(output_channels=32, kernel_shape=(5, 5), stride=(2, 2), padding=((2, 2), (2, 2))),
        Elu(),
        Conv2DTranspose(output_channels=1, kernel_shape=(5, 5), stride=(1, 1), padding=((4, 4), (4, 4))),
      ]
    )(z)


class Residual(Module):
  def __call__(self, x: ndarray) -> ndarray:
    residual = x
    x = Conv2D(256, (1, 1), stride=(1, 1))(x)
    x = Elu()(x)
    x = Conv2D(256, (1, 1), stride=(1, 1))(x)
    return x + residual


class VNCADecoder(Module):
  def __call__(self, z: ndarray) -> ndarray:
    return Sequential(
      [
        Conv2D(256, (3, 3), stride=(1, 1)),
        Residual(),
        Residual(),
        Residual(),
        Residual(),
        Conv2D(256, (1, 1), stride=(1, 1)),
      ]
    )(z)


@dataclass
class BaselineVAE(Module):
  encoder: Encoder
  linear_decoder: LinearDecoder
  decoder: BaselineDecoder

  def __call__(self, x, z_rng):
    z_params = self.encoder(x)
    mean, logvar = rearrange(z_params, 'b (c p) -> p b c', c=128, p=2)
    z = sample(z_rng, mean, logvar)
    z = self.linear_decoder(z)
    z = rearrange(z, 'b (h w c) -> b h w c', h=2, w=2, c=512)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar


def sample(rng, mu, logvar):
  std = np.exp(0.5 * logvar)
  # use the reparameterization trick
  return mu + std * random.normal(rng, mu.shape)

def make_seed(size, latent=LATENT_SIZE):
  x = np.zeros([size, size, latent], np.float32)
  x = x.at[size//2, size//2, 3:].set(1.0)
  return x

def load_image(url: str, target_size:int =TARGET_SIZE) -> ndarray:
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
img = load_emoji('🔔')
plt.imshow(img, cmap='gray')
plt.show()

# %% Initialize the model
@transform
def model(x, rng):
  vae = BaselineVAE(
      encoder=Encoder(),
      linear_decoder=LinearDecoder(),
      decoder=BaselineDecoder(),
  )
  return vae(x, rng)

ex_input = rearrange(img, 'h w c -> 1 h w c')
rng = random.PRNGKey(0)
params = model.init(rng, ex_input, rng)

# %% Show model table
print(hk.experimental.tabulate(model)(ex_input, rng))

# %%
