# %%
from IPython import get_ipython

get_ipython().system('git clone https://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@github.com/albertaillet/vnca.git -b log-outputs')
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


# %%
get_ipython().run_cell_magic(
    'capture', '', '%pip install --upgrade jax tensorflow_probability tensorflow jaxlib numpy equinox einops optax distrax wandb'
)


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


# %%
import time
import wandb
import numpy
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

import equinox as eqx
import jax.numpy as np
from jax.random import PRNGKey, split, permutation
from jax import lax, nn
from jax import pmap, local_device_count, local_devices, device_put_replicated, tree_map, vmap
from einops import rearrange
from optax import adam, clip_by_global_norm, chain

from data import mnist
from loss import iwelbo_loss
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA

# typing
from jax import Array
from equinox import Module
from typing import Any
from jax.random import PRNGKeyArray
from optax import GradientTransformation
from typing import Tuple

MODEL_KEY = PRNGKey(0)
DATA_KEY = PRNGKey(1)
TEST_KEY = PRNGKey(2)
LOGGING_KEY = PRNGKey(3)


# %%
@partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(3, 6), out_axes=(None, 0, 0))
def make_step(data: Array, index: Array, params, static, key: PRNGKeyArray, opt_state: tuple, optim: GradientTransformation) -> Tuple[float, Module, Any]:
    def step(carry, index):
        params, opt_state, key = carry
        x = data[index]
        key, subkey = split(key)

        model = eqx.combine(params, static)
        loss, grads = eqx.filter_value_and_grad(iwelbo_loss)(model, x, subkey)
        loss = lax.pmean(loss, axis_name='num_devices')
        grads = lax.pmean(grads, axis_name='num_devices')

        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state, key), loss

    (params, opt_state, key), loss = lax.scan(step, (params, opt_state, key), index)
    return loss, params, opt_state


@partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(2,))
def test_iwelbo(x: Array, params, static, key: PRNGKeyArray):
    model = eqx.combine(params, static)
    key, subkey = split(key)
    indices = permutation(key, np.arange(len(x)))[:8]  # Very inefficent
    loss = iwelbo_loss(model, x[indices], subkey, M=128)
    return lax.pmean(loss, axis_name='num_devices')


def save_model(model, step):
    model_file_name = f'{model.__class__.__name__}_gstep{step}.eqx'
    eqx.tree_serialise_leaves(model_file_name, model)
    wandb.save(model_file_name)


def restore_model(model_like, file_name, run_path=None):
    wandb.restore(file_name, run_path=run_path)
    model = eqx.tree_deserialise_leaves(file_name, model_like)
    return model


def to_img(x: Array) -> wandb.Image:
    '''Converts an array of shape (c, h, w) to a wandb Image'''
    return wandb.Image(numpy.array(255 * x, dtype=numpy.uint8)[0])


def to_grid(x: Array, ih: int, iw: int) -> Array:
    '''Rearranges a array of images with shape (n, c, h, w) to a grid of shape (c, ih*h, iw*w)'''
    return rearrange(x, '(ih iw) c h w -> c (ih h) (iw w)', ih=ih, iw=iw)


def log_center(model: AutoEncoder) -> wandb.Image:
    center = model.center()
    return to_img(center)


def log_samples(model: AutoEncoder, ih: int = 4, iw: int = 8) -> wandb.Image:
    keys = split(LOGGING_KEY, ih * iw)
    samples = vmap(model.sample)(key=keys)
    samples = to_grid(samples, ih=ih, iw=iw)
    return to_img(samples)


def log_reconstructions(model: AutoEncoder, ih: int = 4, iw: int = 8) -> wandb.Image:
    x = test_data[: ih * iw]
    reconstructions = vmap(model)(x, key=LOGGING_KEY)
    reconstructions = to_grid(reconstructions, ih=ih, iw=iw)
    return to_img(reconstructions)


def log_growth_stages(model: DoublingVNCA) -> wandb.Image:
    stages = model.growth_stages(key=LOGGING_KEY)
    stages = to_grid(stages, ih=model.K, iw=model.N_nca_steps + 1)
    return to_img(stages)


def log_nca_stages(model: NonDoublingVNCA, ih: int = 4) -> wandb.Image:
    stages = model.nca_stages(key=LOGGING_KEY)
    assert model.N_nca_steps % ih == 0, 'N_nca_steps must be divisible by ih'
    iw = model.N_nca_steps // ih 
    stages = rearrange(stages, '(ih iw) c h w -> c (ih h) (iw w)', ih=ih, iw=iw)
    return to_img(stages)


# %%
model = DoublingVNCA(key=MODEL_KEY)

n_tpus = local_device_count()
devices = local_devices()
data, test_data = mnist.load_mnist_on_tpu(devices=local_devices())
n_tpus, devices


# %%
wandb.init(project='vnca', entity='dladv-vnca')

wandb.config.model_type = model.__class__.__name__
wandb.config.latent_size = model.latent_size
wandb.config.batch_size = 128
wandb.config.batch_size_per_tpu = wandb.config.batch_size // n_tpus
wandb.config.n_gradient_steps = 30_000
wandb.config.l = 250
wandb.config.n_tpu_steps = wandb.config.n_gradient_steps // wandb.config.l

wandb.config.n_tpus = n_tpus
wandb.config.lr = 0.00004  # 1e-4
# wandb.config.lr_init_value = 3e-4 # when using exponential_decay
# wandb.config.lr_transition_steps = 100_000
# wandb.config.lr_decay_rate = 0.3
# wandb.config.lr_staircase = True
wandb.config.grad_norm_clip = 10.0

wandb.config.model_key = MODEL_KEY
wandb.config.data_key = DATA_KEY
wandb.config.test_key = TEST_KEY
wandb.config.logging_key = LOGGING_KEY
wandb.config.log_every = 5_000


# %%
train_keys = split(DATA_KEY, wandb.config.n_tpu_steps * n_tpus)
train_keys = rearrange(train_keys, '(n t) k -> n t k', t=n_tpus, n=wandb.config.n_tpu_steps)

test_keys = split(TEST_KEY, wandb.config.n_tpu_steps * n_tpus)
test_keys = rearrange(test_keys, '(n t) k -> n t k', t=n_tpus, n=wandb.config.n_tpu_steps)

params, static = eqx.partition(model, eqx.is_array)

opt = chain(adam(wandb.config.lr), clip_by_global_norm(wandb.config.grad_norm_clip))
opt_state = opt.init(params)

params = device_put_replicated(params, devices)
opt_state = device_put_replicated(opt_state, devices)

pbar = tqdm(
    zip(
        range(wandb.config.n_tpu_steps),
        mnist.indicies_tpu_iterator(n_tpus, wandb.config.batch_size_per_tpu, data.shape[1], wandb.config.n_tpu_steps, DATA_KEY, wandb.config.l),
        train_keys,
        test_keys,
    ),
    total=wandb.config.n_tpu_steps,
)

for i, idx, train_key, test_key in pbar:
    step_time = time.time()
    loss, params, opt_state = make_step(data, idx, params, static, train_key, opt_state, opt)
    step_time = time.time() - step_time

    n_gradient_steps = i * wandb.config.l
    pbar.set_postfix({'loss': f'{np.mean(loss):.3}'})

    wandb.log(
        {
            'loss': float(np.mean(loss)),
            'avg_step_time': (pbar.last_print_t - pbar.start_t) / i if i > 0 else None,
            'step_time': step_time,
            'test_loss': float(test_iwelbo(test_data, params, static, test_key)[0]),
        },
        step=n_gradient_steps,
    )

    if n_gradient_steps % wandb.config.log_every == 0:
        model = eqx.combine(tree_map(partial(np.mean, axis=0), params), static)
        save_model(model, n_gradient_steps)
        wandb.log(
            {
                'center': log_center(model),
                'reconstructions': log_reconstructions(model, test_data, test_key),
                'samples': log_samples(model),
                'growth_plot': log_growth_stages(model) if isinstance(model, DoublingVNCA) else None,
                'nca_stages': log_nca_stages(model) if isinstance(model, NonDoublingVNCA) else None,
            },
            step=n_gradient_steps,
        )

model = eqx.combine(tree_map(partial(np.mean, axis=0), params), static)


# %%
wandb.run.finish()


# %%
local_train_data, local_test_data = mnist.get_mnist()
fig = local_test_data[29]
plt.imshow(nn.sigmoid(model(fig, key=DATA_KEY)[0][0][0]), cmap='gray')
plt.show()
plt.imshow(np.pad(fig[0], ((2, 2), (2, 2))), cmap='gray')
plt.show()


# %%
