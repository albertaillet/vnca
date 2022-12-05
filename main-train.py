# %% Clone the repo
!git clone https://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@github.com/albertaillet/vnca.git -b tpu-iwelbo  # type: ignore
%cd /kaggle/working/vnca  # type: ignore

# %% Install correct packages
%%capture  # type: ignore
%pip install --upgrade jax tensorflow_probability tensorflow jaxlib numpy equinox einops optax distrax wandb  # type: ignore

# %%
import os
if 'TPU_NAME' in os.environ:
    import requests

    if 'TPU_DRIVER_MODE' not in globals():
        url = 'http:' + os.environ['TPU_NAME'].split(':')[1] + ':8475/requestversion/tpu_driver_nightly'
        resp = requests.post(url)
        TPU_DRIVER_MODE = 1

    from jax.config import config

    config.FLAGS.jax_xla_backend = 'tpu_driver'
    config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
    print('Registered TPU:', config.FLAGS.jax_backend_target)
else:
    print('No TPU detected. Can be changed under Runtime/Change runtime type.')

# %% Imports
import time
import wandb
import numpy
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

import equinox as eqx
import jax.numpy as np
from jax.random import PRNGKey, split
from jax import lax, nn
from jax import pmap, local_device_count, local_devices, device_put_replicated, tree_map
from einops import rearrange
from optax import adam, clip_by_global_norm, chain

from data import mnist
from loss import iwelbo_loss
from models import BaselineVAE, DoublingVNCA

# typing
from jax import Array
from equinox import Module
from typing import Any
from jax.random import PRNGKeyArray
from optax import GradientTransformation
from typing import Tuple

MODEL_KEY = PRNGKey(0)
DATA_KEY = PRNGKey(1)

# %% Step function and utils
@partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(3,6), out_axes=(None, 0, 0))
def make_step(data: Array, index: Array, params, static, key: PRNGKeyArray, opt_state: tuple, optim: GradientTransformation) -> Tuple[float, Module, Any]:
    
    def step(carry, index):
        params, opt_state, key = carry
        x = data[index]
        key, subkey = split(key)

        model = eqx.combine(params, static)
        loss, grads = iwelbo_loss(model, x, subkey)
        loss = lax.pmean(loss, axis_name='num_devices')
        grads = lax.pmean(grads, axis_name='num_devices')

        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state, key), loss

    (params, opt_state, key), loss = lax.scan(step, (params, opt_state, key), index)
    return loss, params, opt_state

def save_model(model, step):
    model_file_name = f'{model.__class__.__name__}_gstep{step}.eqx'
    eqx.tree_serialise_leaves(model_file_name, model)
    wandb.save(model_file_name)
    
def restore_model(model_like, file_name, run_path=None):
    wandb.restore(file_name, run_path=run_path)
    model = eqx.tree_deserialise_leaves(file_name, model_like)
    return model

# %% Create model and define parameters
model = DoublingVNCA(key=MODEL_KEY)

n_tpus = local_device_count()
devices = local_devices()
data = mnist.load_mnist_train_on_tpu(devices=local_devices())

n_tpus, devices

# %% Initialise wandb and log hyperparameters
wandb.init(project='vnca', entity='albertaillet')

wandb.config.model_type = model.__class__.__name__
wandb.config.batch_size = 32
wandb.config.batch_size_per_tpu = wandb.config.batch_size // n_tpus
wandb.config.n_gradient_steps = 600_000
wandb.config.l = 250
wandb.config.n_tpu_steps = wandb.config.n_gradient_steps // wandb.config.l

wandb.config.n_tpus = n_tpus
wandb.config.lr = 1e-4
# wandb.config.lr_init_value = 3e-4 # when using exponential_decay
# wandb.config.lr_transition_steps = 100_000
# wandb.config.lr_decay_rate = 0.3
# wandb.config.lr_staircase = True
wandb.config.grad_norm_clip = 10.0

wandb.config.model_key = MODEL_KEY
wandb.config.data_key = DATA_KEY
wandb.config.log_every = 5_000

# %% Training loop
train_keys = split(DATA_KEY, wandb.config.n_tpu_steps * n_tpus)

train_keys = rearrange(train_keys, '(n t) k -> n t k', t=n_tpus, n=wandb.config.n_tpu_steps)

params, static = eqx.partition(model, eqx.is_array)

opt = chain(adam(wandb.config.lr), clip_by_global_norm(wandb.config.grad_norm_clip))
opt_state = opt.init(params)

params = device_put_replicated(params, devices)
opt_state = device_put_replicated(opt_state, devices)

pbar = tqdm(
    zip(
        range(wandb.config.n_tpu_steps),
        mnist.indicies_tpu_iterator(n_tpus, wandb.config.batch_size_per_tpu, data.shape[1], wandb.config.n_tpu_steps, DATA_KEY, wandb.config.l), 
        train_keys
    ),
    total=wandb.config.n_tpu_steps,
)

for i, idx, key in pbar:
    step_time = time.time()
    loss, params, opt_state = make_step(data, idx, params, static, key, opt_state, opt)
    step_time = time.time() - step_time
    
    n_gradient_steps = i * wandb.config.l
    pbar.set_postfix({'loss': f'{np.mean(loss):.3}'})
    
    wandb.log(
        {
            'loss': float(np.mean(loss)),
            'avg_step_time': (pbar.last_print_t - pbar.start_t) / i  if i > 0 else None,
            'step_time': step_time,
        },
        step = n_gradient_steps
    )  
    
    if n_gradient_steps % wandb.config.log_every == 0:
        model = eqx.combine(tree_map(partial(np.mean, axis=0), params), static)
        save_model(model, n_gradient_steps)
        if isinstance(model, DoublingVNCA):
            stages = model.growth_stages(key=DATA_KEY)
            growth_plot = numpy.array(255*stages, dtype=numpy.uint8)[0]
            wandb.log(
                {'growth_plot': wandb.Image(growth_plot)},
                step = n_gradient_steps
            )

model = eqx.combine(tree_map(partial(np.mean, axis=0), params), static)

# %%
wandb.run.finish()

# %%
local_train_data, local_test_data = mnist.get_mnist()
fig = local_test_data[201]
plt.imshow(nn.sigmoid(model(fig, key=DATA_KEY)[0][0]), cmap='gray')
plt.show()
plt.imshow(np.pad(fig[0], ((2, 2), (2, 2))), cmap='gray')
plt.show()
