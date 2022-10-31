# %% Imports

import equinox as eqx
import jax.numpy as np
from jax.random import PRNGKey, split
from einops import rearrange, reduce
from optax import adam, exponential_decay
import matplotlib.pyplot as plt 

from models import BaselineVAE
from data_loading import load_emoji

# typing
from jax import Array
from equinox import Module
from typing import Optional, Any
from jax.random import PRNGKeyArray
from optax import GradientTransformation

TARGET_SIZE = 28
LATENT_SIZE = 16

# %% Define the neural nets
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


# %% Load the image
img = load_emoji('🌗', TARGET_SIZE)
x = rearrange(img, 'h w -> 1 h w')
plt.imshow(img, cmap='gray')
plt.show()

# %% Create the model
model_key = PRNGKey(0)
vae = BaselineVAE(key=model_key)

# %% Train the model
lr = exponential_decay(3e-5, 60, 0.1, staircase=True)
opt = adam(lr)
opt_state = opt.init(eqx.filter(vae, eqx.is_array))

n_gradient_steps = 100
for key in split(model_key, n_gradient_steps):
    loss, vae, opt_state = make_step(vae, x, key, opt_state, opt)
    print(loss)

# %%
plt.imshow(vae(x, PRNGKey(0))[0][0], cmap='gray')
plt.show()
plt.imshow(vae.center()[0], cmap='gray')
plt.show()

# %%
