# %% Create Neural Cellular Automata using JAX

import jax.numpy as np
from jax import jit, vmap, value_and_grad
from jax import lax, random
from einops import rearrange, reduce, repeat
from optax import adam, apply_updates

from flax import linen as nn
from flax.training import train_state

import io
import optax
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# typing
from jax.numpy import ndarray

TARGET_SIZE = 32
LATENT_SIZE = 16

# %% Define the neural cellular automata

class Flatten(nn.Module):
  @nn.compact
  def __call__(self, x: ndarray) -> ndarray:
    return rearrange(x, 'b h w c -> b (h w c)')


class Elu(nn.Module):
  alpha: float

  @nn.compact
  def __call__(self, x: ndarray) -> ndarray:
    return nn.elu(x, alpha=1.0)


class Encoder(nn.Module):
  @nn.compact
  def __call__(self, x: ndarray) -> ndarray:
    return nn.Sequential(
      [
        nn.Conv(features=32, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.Conv(features=256, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.Conv(features=512, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        Flatten(),
        nn.Dense(256),
      ]
    )(x)


class LinearDecoder(nn.Module):
  @nn.compact
  def __call__(self, z: ndarray) -> ndarray:
    return nn.Dense(2048)(z)


class BaselineDecoder(nn.Module):
  @nn.compact
  def __call__(self, z: ndarray) -> ndarray:
    return nn.Sequential(
      [
        nn.ConvTranspose(features=256, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.ConvTranspose(features=128, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.ConvTranspose(features=64, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.ConvTranspose(features=32, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)),
        Elu(alpha=1.0),
        nn.ConvTranspose(features=1, kernel_size=(5, 5), strides=(1, 1), padding=(4, 4)),
      ]
    )(z)


class Residual(nn.Module):
  @nn.compact
  def __call__(self, x: ndarray) -> ndarray:
    residual = x
    x = nn.Conv(256, (1, 1), strides=(1, 1))(x)
    x = Elu(alpha=1.0)(x)
    x = nn.Conv(256, (1, 1), strides=(1, 1))(x)
    return x + residual


class VNCADecoder(nn.Module):
  @nn.compact
  def __call__(self, z: ndarray) -> ndarray:
    return nn.Sequential(
      [
        nn.Conv(256, (3, 3), strides=(1, 1)),
        Residual(),
        Residual(),
        Residual(),
        Residual(),
        nn.Conv(256, (1, 1), stride=(1, 1)),
      ]
    )(z)


class BaselineVAE(nn.Module):
  def setup(self):
    self.encoder = Encoder()
    self.linear_decoder = LinearDecoder()
    self.decoder = BaselineDecoder()

  @nn.compact
  def __call__(self, x, z_rng):
    z_params = self.encoder(x)
    mean, logvar = rearrange(z_params, 'b (c p) -> p b c', c=128, p=2)
    z = sample(z_rng, mean, logvar)
    z = self.linear_decoder(z)
    z = rearrange(z, 'b (h w c) -> b h w c', h=2, w=2, c=512)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))


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
img = load_emoji('🤡')
plt.imshow(img, cmap='gray')
plt.show()

# %% Train the model
vae = BaselineVAE()
lr = 1e-2
rng = random.PRNGKey(0)
params = vae.init(
  random.PRNGKey(0), 
  rearrange(img, 'h w c -> 1 h w c'),
  random.PRNGKey(0)
)['params']
state = train_state.TrainState.create(apply_fn=vae.apply, params=params, tx=adam(lr))

# %% Get a table of the model
vae.tabulate(
  random.PRNGKey(0), 
  rearrange(img, 'h w c -> 1 h w c'),
  random.PRNGKey(0),
  console_kwargs = {'force_terminal': False, 'force_jupyter': True}
)

# %%
def loss_fn(params, x, rng):
  recon_x, mean, logvar = vae.apply({'params': params}, x, rng)
  print(recon_x.shape, x.shape, mean.shape, logvar.shape)
  recon_loss = np.mean(np.square(recon_x - x))
  kl_loss = -0.5 * np.mean(1 + logvar - np.square(mean) - np.exp(logvar))
  return recon_loss + kl_loss

def train_step(state, x, rng):
  loss, grads = value_and_grad(loss_fn)(state.params, x, rng)
  state = state.apply_gradients(grads=grads)
  return state, loss

for epoch in range(100):
  rng, rng_step = random.split(rng)
  state, loss = train_step(state, rearrange(img, 'h w c -> 1 h w c'), rng_step)
  print('Epoch %d, loss %.4f'%(epoch, loss))


# %% Train NCA
n_total_steps = 75
n_growing_steps = 64
iterations = 2000
theta = 0.0
lr = 2e-3

def loss(params, img, theta):
  losses = []
  output = make_seed(TARGET_SIZE)
  for i in range(n_total_steps):
    output = nca(params, output, theta)
    if i >= n_growing_steps:
      loss = np.mean((output[..., :4] - img)**2)
      losses.append(loss)

  return np.mean(np.array(losses)), output[..., :4].clip(0, 1)

opt = adam(lr)
opt_state = opt.init(params)
for iteration in range(iterations):

  (l, out), grads = value_and_grad(loss, has_aux=True)(params, img, theta)

  def norm(x):
    return x / (np.linalg.norm(x) + 1e-8)
  
  grads = [(norm(dw), norm(db)) for (dw, db) in grads]
  updates, opt_state = opt.update(grads, opt_state)
  params = apply_updates(params, updates)

  print(f'Iteration {iteration} Loss {l:.4f}')

# %%
n = 8
_, im = loss(params, img, 0)
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.show()

# %%
plt.imshow(img)
plt.axis('off')
# %%
