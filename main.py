# Imports
import equinox as eqx
import jax.numpy as np
from jax.random import PRNGKey, split
from einops import rearrange, repeat
from optax import adam, exponential_decay
import matplotlib.pyplot as plt

from models import BaselineVAE
from datasets.mnist import load_mnist

# typing
from jax import Array, vmap
from equinox import Module
from typing import Optional, Any
from jax.random import PRNGKeyArray
from optax import GradientTransformation

TARGET_SIZE = 28
MODEL_KEY = PRNGKey(0)
DATA_KEY = PRNGKey(1)

# %%
@eqx.filter_value_and_grad
def loss_fn(model: Module, x: Array, key: PRNGKeyArray) -> float:
    keys = split(key, len(x))
    recon_x, mean, logvar = vmap(model)(x, keys)
    recon_loss = np.mean(np.square(recon_x - x))
    kl_loss = -0.5 * np.mean(1 + logvar - np.square(mean) - np.exp(logvar))
    return recon_loss + kl_loss


@eqx.filter_jit
def make_step(model: Module, x: Array, key: PRNGKeyArray, opt_state: tuple, optim: GradientTransformation) -> tuple[float, Module, Any]:
    loss, grads = loss_fn(model, x, key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

# %% Create the model
vae = BaselineVAE(key=MODEL_KEY)

# %% Train the model
batch_size = 32
lr = exponential_decay(3e-5, 60, 0.1, staircase=True)
opt = adam(lr)
opt_state = opt.init(eqx.filter(vae, eqx.is_array))

train_data, test_data = load_mnist(batch_size=batch_size, key=DATA_KEY)

n_gradient_steps = 100
steps = range(n_gradient_steps)
train_keys = split(DATA_KEY, n_gradient_steps)

for step, batch, key in zip(steps, train_data, train_keys):
    loss, vae, opt_state = make_step(vae, batch, key, opt_state, opt)
    print(step, loss)

# %%
plt.imshow(vae.center()[0], cmap='gray')
plt.show()