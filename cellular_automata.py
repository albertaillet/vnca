# %% Create Neural Cellular Automata using JAX

import jax.numpy as np
from jax import jit, vmap
from jax import lax, nn, random
from einops import rearrange, reduce, repeat

import io
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# typing
from jax.numpy import ndarray

TARGET_SIZE = 40

# %% Define the neural cellular automata
@jit
def convolve(image: ndarray, kernel: ndarray) -> ndarray:
  lhs = rearrange(image, 'h w -> 1 1 h w')
  rhs = rearrange(kernel, 'h w -> 1 1 h w')
  conv = lax.conv(lhs, rhs, (1, 1), 'SAME')
  return rearrange(conv, '1 1 h w -> h w')

@jit
def percieve(image: ndarray, theta: float) -> ndarray:
  '''Perceive the image using a convolutional kernel'''

  # Sobel operators 
  d = np.array([-1, 0, 1])
  v = np.array([1, 2, 1])
  dx = np.outer(v, d)
  dy = np.outer(d, v)
  sobel = np.array([dx,dy])
  
  # Rotate sobel operators
  rotation = np.array(
    [
      [np.cos(theta), -np.sin(theta)],
      [np.sin(theta), np.cos(theta)]
    ]
  )
  kernels = np.einsum('ij, jlm -> ilm', rotation, sobel)
  
  f = vmap(convolve, (2, None), -1)
  conv = vmap(f, (None, 0), -1)(image, kernels)

  image = rearrange(image, 'h w c -> h w c 1')
  return np.concatenate([image, conv], axis=-1)

@jit
def nca(params: list[tuple[ndarray]], image: ndarray, theta: float) -> ndarray:
  '''Neural Cellular Automata'''
  perception = percieve(image, theta)
  perception = rearrange(perception, 'h w c k -> (h w) (c k)')
  
  def _predict(params: list, input: ndarray) -> ndarray:
    (w1, b1), (w2, b2) = params
    h1 = nn.relu(np.dot(w1, input) + b1)
    return np.dot(w2, h1) + b2
  
  change = vmap(_predict, (None, 0), 0)(params, perception)
  h, w, *_ = image.shape
  change = rearrange(change, '(h w) c -> h w c', h=h, w=w)

  alive_mask = get_alive_mask(image)

  return image + change * alive_mask

def get_alive_mask(image: ndarray) -> ndarray:
  '''Get alive mask using maxpool in 3x3 window.'''
  mask = (image[:,:,3] > 0.1)
  pool = lax.reduce_window(
    operand = mask, 
    init_value = False, 
    computation = lambda x, y: x | y,
    window_dimensions=(3, 3), 
    window_strides=(1, 1), 
    padding='SAME'
  )
  return rearrange(pool, 'h w -> h w 1')

def random_layer_params(m, n, key, scale):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key, scale=1e-2):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def load_image(url: str) -> ndarray:
  r = requests.get(url)
  img = Image.open(io.BytesIO(r.content))
  img = np.float32(img)/255.0
  # premultiply RGB by Alpha
  return img

def load_emoji(emoji: str) -> str:
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url)

# %% Load the image
img = load_emoji('🥰')
plt.imshow(img)

# %% Test NCA
LATENT_SIZE = 4
params = init_network_params([LATENT_SIZE*3, 32, LATENT_SIZE], random.PRNGKey(0), scale=10)
output = nca(params, img, theta=0.0)
output.shape


# %%
plt.imshow(output, cmap='gray')
plt.colorbar()
# %%
