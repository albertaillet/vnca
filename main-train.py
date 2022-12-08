# %%
from IPython import get_ipython

get_ipython().system('git clone https://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@github.com/albertaillet/vnca.git')
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


# %%
get_ipython().run_cell_magic('capture', '', '%pip install --upgrade jax tensorflow_probability tensorflow jaxlib numpy equinox einops optax distrax wandb')


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
from tqdm import tqdm
from functools import partial

import equinox as eqx
import jax.numpy as np
from jax.random import PRNGKey, split, randint, permutation
from jax import lax
from jax.nn import sigmoid
from jax import pmap, local_device_count, local_devices, device_put_replicated, tree_map, vmap
from einops import rearrange
from optax import adam, clip_by_global_norm, chain

from data import binarized_mnist
from loss import forward, iwelbo_loss
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA, sample_gaussian, crop
from log_utils import save_model, restore_model, to_img, log_center, log_samples, log_reconstructions, log_growth_stages, log_nca_stages

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
        loss, grads = eqx.filter_value_and_grad(forward)(model, x, subkey)
        loss = lax.pmean(loss, axis_name='num_devices')
        grads = lax.pmean(grads, axis_name='num_devices')

        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state, key), loss

    (params, opt_state, key), loss = lax.scan(step, (params, opt_state, key), index)
    return loss, params, opt_state


def damage_half(x: Array, *, key: PRNGKeyArray) -> Array:
    '''Set half of the image to 0.0'''
    _, h, w = x.shape
    h_half, w_half = h // 2, w // 2
    hmask, wmask = randint(
        key = key, 
        shape = (2,), 
        minval = np.zeros(2, dtype=np.int32), 
        maxval = np.array([h_half, w_half], dtype=np.int32)
    )
    return x.at[:, hmask:hmask + h_half, wmask:wmask + h_half].set(0.0)


@partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(3, 6), out_axes=(None, 0, 0))
def make_pool_step(data: Array, index: Array, params, static, key: PRNGKeyArray, opt_state: tuple, optim: GradientTransformation, pool: Tuple[Array, Array]) -> Tuple[float, Module, Any]:
    batch_size = len(index)
    n_pool_samples = batch_size // 2

    def step(carry, index):
        params, opt_state, key = carry
        model: NonDoublingVNCA = eqx.combine(params, static)

        x = data[index]  # (batch_size, c, h, w)
        key, subkey, *damage_keys = split(key, batch_size + 2)

        if np.all(pool == 0.0):  # first step
            x_pool, z_pool = pool  # (pool_size, c, h, w), (pool_size, z_dim, h, w)
            x_pool_samples = x_pool[:n_pool_samples]
            z_pool_samples = z_pool[:n_pool_samples]
            
            x = x.at[n_pool_samples:].set(x_pool_samples)  # (batch_size, c, h, w)
            z_pool_samples = vmap(damage_half)(z_pool_samples, key=damage_keys)  # (batch_size, z_dim, h, w)
        
        @partial(eqx.filter_value_and_grad, has_aux=True)
        def forward(model: NonDoublingVNCA, x: Array, z_pool_samples: Array, *, key: PRNGKeyArray) -> Tuple[Array, Array]:
            mean, logvar = model.encoder(x)
            z_0 = sample_gaussian(mean, logvar, subkey)  # (batch_size, z_dim)
            z_0 = rearrange(z_0, 'b c -> b c h w', h=32, w=32)

            if np.all(pool == 0.0):  # first step
                z_0 = z_0.at[n_pool_samples:].set(z_pool_samples)  # (pool_size, z_dim, h, w)

            z_T = model.decode_grid(z_0)

            b, c, h, w = x.shape
            recon_x = vmap(partial(crop, shape=(c, h, w)))(z_T)
            
            # add M diminersion to x and recon_x
            x_M = rearrange(x, 'b c h w -> b m c h w', m=1)
            recon_x_M = rearrange(recon_x, 'b c h w -> b m c h w', m=1)

            loss = iwelbo_loss(recon_x_M, x_M, z_T, mean, logvar, key, M=1)

            return loss, (x, z_T)

        loss, grads, (x, z_T) = forward(model, x, z_pool_samples, key=subkey)
        
        # permuation of non-zero parts of pool
        zero_pool = (z_pool == 0.0).all(axis=(1, 2, 3))
        if np.all(zero_pool == False):
            i = np.argmax(zero_pool)
            # permute the already filled pool
            z_pool = z_pool.at[:i].set(permutation(pool[:i], key=key, axis=0))
            x_pool = x_pool.at[:i].set(permutation(pool[:i], key=key, axis=0))
            
            # fill the pool with the new samples
            z_pool = z_pool.at[i:batch_size].set(z_T)
            x_pool = x_pool.at[i:batch_size].set(x)
        else:
            z_pool = z_pool.at[:batch_size].set(z_T)
            x_pool = x_pool.at[:batch_size].set(x)
            z_pool = permutation(z_pool, key=key, axis=0)
            x_pool = permutation(x_pool, key=key, axis=0)
        
        loss = lax.pmean(loss, axis_name='num_devices')
        grads = lax.pmean(grads, axis_name='num_devices')

        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state, key), loss

    (params, opt_state, key), loss = lax.scan(step, (params, opt_state, key), index)
    return loss, params, opt_state


@partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(2, 4, 5))
def test_iwelbo(x: Array, params, static, key: PRNGKeyArray, M: int, batch_size: int):
    model = eqx.combine(params, static)
    key, subkey = split(key)
    indices = randint(key, (batch_size,), 0, len(x))
    loss = forward(model, x[indices], subkey, M=M)
    return lax.pmean(loss, axis_name='num_devices')


# %%
model = DoublingVNCA(key=MODEL_KEY, latent_size=2)

n_tpus = local_device_count()
devices = local_devices()
data, test_data = binarized_mnist.load_data_on_tpu(devices=local_devices(), key=TEST_KEY)
n_tpus, devices


# %%
wandb.init(project='vnca', entity='dladv-vnca', mode='offline')

wandb.config.model_type = model.__class__.__name__
wandb.config.latent_size = model.latent_size
wandb.config.batch_size = 32
wandb.config.batch_size_per_tpu = wandb.config.batch_size // n_tpus
wandb.config.n_gradient_steps = 100_000
wandb.config.l = 250
wandb.config.n_tpu_steps = wandb.config.n_gradient_steps // wandb.config.l

wandb.config.test_loss_batch_size = 8
wandb.config.test_loss_latent_samples = 128
wandb.config.beta = 1

wandb.config.n_tpus = n_tpus
wandb.config.lr = 1e-4
wandb.config.grad_norm_clip = 10.0

wandb.config.model_key = MODEL_KEY
wandb.config.data_key = DATA_KEY
wandb.config.test_key = TEST_KEY
wandb.config.logging_key = LOGGING_KEY
wandb.config.log_every = 10_000

wandb.config.pool_size = 1024 if isinstance(model, DoublingVNCA) else None


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

if wandb.config.pool_size is not None:
    x_pool = np.empty((wandb.config.pool_size, 1, 32, 32), dtype=np.float32)
    z_pool = np.empty((wandb.config.pool_size, model.latent_size, 32, 32), dtype=np.float32)
    pool = (x_pool, z_pool)

    pool = device_put_replicated(pool, devices)

# %%
pbar = tqdm(
    zip(
        range(1, wandb.config.n_tpu_steps + 1),
        binarized_mnist.indicies_tpu_iterator(n_tpus, wandb.config.batch_size_per_tpu, data.shape[1], wandb.config.n_tpu_steps, DATA_KEY, wandb.config.l),
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
            'test_loss': float(
                test_iwelbo(
                    test_data,
                    params,
                    static,
                    test_key,
                    wandb.config.test_loss_batch_size,
                    wandb.config.test_loss_latent_samples,
                )[0]
            ),
        },
        step=n_gradient_steps,
    )

    if n_gradient_steps % wandb.config.log_every == 0:
        model = eqx.combine(tree_map(partial(np.mean, axis=0), params), static)
        save_model(model, n_gradient_steps)
        wandb.log(
            {
                'center': to_img(log_center(model)),
                'reconstructions': to_img(log_reconstructions(model, test_data[0], key=LOGGING_KEY)),
                'samples': to_img(log_samples(model, key=LOGGING_KEY)),
                'growth_plot': to_img(log_growth_stages(model, key=LOGGING_KEY)) if isinstance(model, DoublingVNCA) else None,
                'nca_stages': to_img(log_nca_stages(model, key=LOGGING_KEY)) if isinstance(model, NonDoublingVNCA) else None,
            },
            step=n_gradient_steps,
            commit=True,
        )


# %%
wandb.finish()
