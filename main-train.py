# %%
from IPython import get_ipython

get_ipython().system('git clone https://github.com/albertaillet/vnca.git')


# %%
get_ipython().run_cell_magic('capture', '', '%pip install --upgrade jax tensorflow_probability tensorflow jaxlib numpy equinox einops optax distrax wandb datasets')


# %%
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


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
from jax import lax, jit
from jax import pmap, local_device_count, local_devices, device_put_replicated, device_put_sharded, tree_map, vmap
from einops import rearrange, repeat
from optax import adam, clip_by_global_norm, chain

from data import load_data_on_tpu, indicies_tpu_iterator
from loss import forward, vae_loss, iwae_loss
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA, sample_gaussian, crop, damage
from log_utils import save_model, restore_model, to_wandb_img, log_center, log_samples, log_reconstructions, log_growth_stages, log_nca_stages

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

        
@partial(pmap, donate_argnums=(1, 2, 4, 5, 7, 8), axis_name='num_devices', static_broadcasted_argnums=(3, 6), out_axes=(None, 0, 0, 0))
def make_pool_step(
    data: Array, index: Array, params, static, key: PRNGKeyArray, opt_state: tuple, optim: GradientTransformation, t_key: PRNGKeyArray, pool: Tuple[Array, Array]
) -> Tuple[float, Module, Any]:
    batch_size = index.shape[1]
    n_pool_samples = batch_size // 2
    n_half_pool_samples = n_pool_samples // 2
    axis_index = lax.axis_index('num_devices')

    @partial(jit, donate_argnums=(0, 1))
    def step(carry: Tuple, index: Array) -> Tuple:
        params, opt_state, key, t_key, pool = carry
        model: NonDoublingVNCA = eqx.combine(params, static)

        x = data[index]  # (batch_size, c, h, w)
        next_key, fwd_key, subkey = split(key, 3)
        t_key, next_t_key = split(t_key, 2)

        x_pool, z_pool = pool  # (pool_size, c, h, w), (pool_size, z_dim, h, w)
        x_pool_samples = x_pool[:n_pool_samples]
        z_pool_samples = z_pool[:n_pool_samples]

        damage_keys = split(subkey, n_half_pool_samples)
        x = x.at[n_pool_samples:].set(x_pool_samples)  # (batch_size, c, h, w)
        damaged_half = vmap(damage)(z_pool_samples[:n_half_pool_samples], key=damage_keys)  # (batch_size, z_dim, h, w)
        z_pool_samples = z_pool_samples.at[:n_half_pool_samples].set(damaged_half)

        @partial(eqx.filter_value_and_grad, has_aux=True)
        def forward(model: NonDoublingVNCA, x: Array, z_pool_samples: Array, *, key: PRNGKeyArray, t_key: PRNGKeyArray) -> Tuple[float, Tuple[Array, Array]]:

            # encode x and get parameters of latent distribution, sample z_0 and repeat to (pool_size, z_dim, h, w)
            mean, logvar = vmap(model.encoder, out_axes=1)(x)
            z_0 = sample_gaussian(mean, logvar, mean.shape, key=key)  # (batch_size, z_dim)
            z_0 = repeat(z_0, 'b c -> b c h w', h=32, w=32)

            # set second half of z_0 to z_pool_samples
            z_0 = z_0.at[n_pool_samples:].set(z_pool_samples)  # (pool_size, z_dim, h, w)

            # decode z_0 to z_T using a random number of steps sampled bewtween N_nca_steps_min and N_nca_steps_max using t_key
            z_T = vmap(partial(model.decode_grid_random, key=t_key))(z_0)

            # crop z_T to x.shape to get reconstructed x
            b, c, h, w = x.shape
            recon_x = vmap(partial(crop, shape=(c, h, w)))(z_T)

            # add M diminersion to recon_x
            recon_x_M = repeat(recon_x, 'b c h w -> b m c h w', m=1)

            # get loss
            loss = vae_loss(recon_x_M, x, mean, logvar, M=1)

            return loss, (x, z_T)

        (loss, (x, z_T)), grads = forward(model, x, z_pool_samples, key=fwd_key, t_key=t_key)

        # update pool
        z_pool = z_pool.at[:batch_size].set(z_T)
        x_pool = x_pool.at[:batch_size].set(x)

        # Permute whole pool
        p, c, h, w = z_pool.shape
        z_pool = permutation(next_t_key, lax.all_gather(z_pool, 'num_devices')).reshape(n_tpus, p, c, h, w)[axis_index]
        x_pool = permutation(next_t_key, lax.all_gather(x_pool, 'num_devices')).reshape(n_tpus, p, 1, h, w)[axis_index]

        pool = (x_pool, z_pool)

        loss = lax.pmean(loss, axis_name='num_devices')
        grads = lax.pmean(grads, axis_name='num_devices')

        updates, opt_state = optim.update(grads, opt_state)
        params = eqx.apply_updates(params, updates)

        return (params, opt_state, next_key, next_t_key, pool), loss

    (params, opt_state, key, t_key, pool), loss = lax.scan(step, (params, opt_state, key, t_key, pool), index)
    return loss, params, opt_state, pool


@partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(2, 4, 5))
def test_iwelbo(x: Array, params, static, key: PRNGKeyArray, K: int, batch_size: int):
    model = eqx.combine(params, static)
    key, subkey = split(key)
    indices = randint(key, (batch_size,), 0, len(x))
    loss = iwae_loss(model, x[indices], key=subkey, K=K)
    return lax.pmean(loss, axis_name='num_devices')


# %%
model = NonDoublingVNCA(key=MODEL_KEY, latent_size=128)

n_tpus = local_device_count()
devices = local_devices()
data, test_data = load_data_on_tpu(devices=local_devices(), dataset='binarized_mnist', key=TEST_KEY)
n_tpus, devices


# %%
wandb.init(project='vnca', entity='dladv-vnca')

wandb.config.model_type = model.__class__.__name__
wandb.config.latent_size = model.latent_size
wandb.config.batch_size = 128
wandb.config.n_gradient_steps = 100_000
wandb.config.pool_size = 1024 if isinstance(model, NonDoublingVNCA) else None


wandb.config.batch_size_per_tpu = wandb.config.batch_size // n_tpus
wandb.config.l = 125
wandb.config.n_tpu_steps = wandb.config.n_gradient_steps // wandb.config.l
wandb.config.pool_size_per_tpu = (wandb.config.pool_size // n_tpus) if isinstance(model, NonDoublingVNCA) else None

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
wandb.config.log_every = 2_000


# %%
train_keys = split(DATA_KEY, wandb.config.n_tpu_steps * n_tpus)
train_keys = rearrange(train_keys, '(n t) k -> n t k', t=n_tpus, n=wandb.config.n_tpu_steps)

t_keys = split(DATA_KEY, wandb.config.n_tpu_steps)
t_keys = repeat(t_keys, 'n k -> n t k', t=n_tpus, n=wandb.config.n_tpu_steps)

test_keys = split(TEST_KEY, wandb.config.n_tpu_steps * n_tpus)
test_keys = rearrange(test_keys, '(n t) k -> n t k', t=n_tpus, n=wandb.config.n_tpu_steps)

params, static = eqx.partition(model, eqx.is_array)

opt = chain(adam(wandb.config.lr), clip_by_global_norm(wandb.config.grad_norm_clip))
opt_state = opt.init(params)

params = device_put_replicated(params, devices)
opt_state = device_put_replicated(opt_state, devices)


# %%
if wandb.config.pool_size is not None:
    x_pool = data[0, : wandb.config.pool_size_per_tpu * n_tpus].copy()
    mean, logvar = vmap(model.encoder, out_axes=1)(x_pool)
    z_pool = sample_gaussian(mean, logvar, mean.shape, key=DATA_KEY)
    z_pool = repeat(z_pool, 'n c -> n c h w', h=32, w=32, n=wandb.config.pool_size_per_tpu * n_tpus, c=model.latent_size)

    # reshape to tpus
    x_pool = rearrange(x_pool, '(t n) c h w -> t n c h w', n=wandb.config.pool_size_per_tpu, t=n_tpus, c=1, h=32, w=32)
    z_pool = rearrange(z_pool, '(t n) c h w -> t n c h w', n=wandb.config.pool_size_per_tpu, t=n_tpus, c=wandb.config.latent_size, h=32, w=32)

    x_pool = device_put_sharded([*x_pool], devices)
    z_pool = device_put_sharded([*z_pool], devices)
    pool = (x_pool, z_pool)


# %%
pbar = tqdm(
    zip(
        range(1, wandb.config.n_tpu_steps + 1),
        indicies_tpu_iterator(n_tpus, wandb.config.batch_size_per_tpu, data.shape[1], wandb.config.n_tpu_steps, DATA_KEY, wandb.config.l),
        train_keys,
        test_keys,
        t_keys,
    ),
    total=wandb.config.n_tpu_steps,
)

for i, idx, train_key, test_key, t_key in pbar:
    step_time = time.time()
    if wandb.config.pool_size is None:
        loss, params, opt_state = make_step(data, idx, params, static, train_key, opt_state, opt)
    elif wandb.config.pool_size is not None:
        loss, params, opt_state, pool = make_pool_step(data, idx, params, static, train_key, opt_state, opt, t_key, pool)
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

    if n_gradient_steps % wandb.config.log_every == 0 or n_gradient_steps == wandb.config.l:
        model = eqx.combine(tree_map(partial(np.mean, axis=0), params), static)
        save_model(model, n_gradient_steps)
        wandb.log(
            {
                'center': to_wandb_img(log_center(model)),
                'reconstructions': to_wandb_img(log_reconstructions(model, test_data[0], key=LOGGING_KEY)),
                'samples': to_wandb_img(log_samples(model, key=LOGGING_KEY)),
                'growth_plot': to_wandb_img(log_growth_stages(model, key=LOGGING_KEY)) if isinstance(model, DoublingVNCA) else None,
                'nca_stages': to_wandb_img(log_nca_stages(model, key=LOGGING_KEY, ih=9, iw=8)) if isinstance(model, NonDoublingVNCA) else None,
            },
            step=n_gradient_steps,
            commit=True,
        )


# %%
wandb.finish()
