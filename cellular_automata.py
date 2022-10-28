# %% Create Neural Cellular Automata using JAX

import jax.numpy as np
from jax import jit, vmap, value_and_grad
from jax import lax, random
from jax.nn import relu
from einops import rearrange, reduce, repeat
from optax import adam, apply_updates

import io
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# typing
from jax.numpy import ndarray

TARGET_SIZE = 40
LATENT_SIZE = 16

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
  kernels = np.einsum('ij, jlm -> ilm', rotation, sobel, optimize=True)
  
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
    h1 = relu(np.dot(w1, input) + b1)
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

def make_seed(size, latent=LATENT_SIZE):
  x = np.zeros([size, size, latent], np.float32)
  x = x.at[size//2, size//2, 3:].set(1.0)
  return x

def load_image(url: str, target_size:int =TARGET_SIZE) -> ndarray:
  r = requests.get(url)
  img = Image.open(io.BytesIO(r.content))
  img = img.resize((target_size, target_size))
  img = np.float32(img)/255.0
  # premultiply RGB by Alpha
  img = img.at[..., :3].set(img[..., :3] * img[..., 3:])
  return img

def load_emoji(emoji: str) -> str:
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url)

# %% Load the image
img = load_emoji('🥰')
plt.imshow(img)

# %% Test NCA
params = init_network_params([LATENT_SIZE*3, 128, LATENT_SIZE], random.PRNGKey(0), scale=10)
seed = make_seed(TARGET_SIZE)
output = nca(params, seed, theta=0.0)
plt.imshow(output[..., :3], cmap='gray')
plt.colorbar()

# %% Train NCA
n_total_steps = 75
n_growing_steps = 64
iterations = 2000
theta = 0.0
lr = 2e-3
params = init_network_params([LATENT_SIZE*3, 32, LATENT_SIZE], random.PRNGKey(0))

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
