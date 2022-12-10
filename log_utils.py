import wandb
import numpy as np
import equinox as eqx
from jax.nn import sigmoid
from einops import rearrange
from jax.random import split, randint
from jax import vmap
from PIL import Image

from models import AutoEncoder, DoublingVNCA, NonDoublingVNCA

# typing
from jax import Array
from jax.random import PRNGKeyArray


def save_model(model, step):
    model_file_name = f'{model.__class__.__name__}_gstep{step}.eqx'
    eqx.tree_serialise_leaves(model_file_name, model)
    wandb.save(model_file_name)


def restore_model(model_like, file_name, run_path=None):
    wandb.restore(file_name, run_path=run_path)
    model = eqx.tree_deserialise_leaves(file_name, model_like)
    return model


def to_wandb_img(x: Array) -> wandb.Image:
    '''Converts an array of shape (c, h, w) to a wandb Image'''
    return wandb.Image(to_PIL_img(x))


def to_PIL_img(x: Array) -> Image:
    '''Converts an array of shape (c, h, w) to a PIL Image'''
    x = np.clip(x, 0, 1)
    return Image.fromarray(np.array(255 * x, dtype=np.uint8)[0])


@eqx.filter_jit
def to_grid(x: Array, ih: int, iw: int) -> Array:
    '''Rearranges a array of images with shape (n, c, h, w) to a grid of shape (c, ih*h, iw*w)'''
    return rearrange(x, '(ih iw) c h w -> c (ih h) (iw w)', ih=ih, iw=iw)


@eqx.filter_jit
def log_center(model: AutoEncoder) -> Array:
    return sigmoid(model.center())


@eqx.filter_jit
def log_samples(model: AutoEncoder, ih: int = 4, iw: int = 8, *, key: PRNGKeyArray) -> Array:
    keys = split(key, ih * iw)
    samples = vmap(model.sample)(key=keys)
    samples = sigmoid(samples)
    return to_grid(samples, ih=ih, iw=iw)


@eqx.filter_jit
def log_reconstructions(model: AutoEncoder, data: Array, ih: int = 4, iw: int = 8, *, key: PRNGKeyArray) -> Array:
    idx = randint(key, (ih * iw,), 0, len(data))
    keys = split(key, ih * iw)
    x = data[idx]
    reconstructions, _, _ = vmap(model)(x, key=keys)
    reconstructions = rearrange(reconstructions, 'n m c h w -> (n m) c h w')
    reconstructions = sigmoid(reconstructions)
    return to_grid(reconstructions, ih=ih, iw=iw)


@eqx.filter_jit
def log_growth_stages(model: DoublingVNCA, *, key: PRNGKeyArray) -> Array:
    stages = model.growth_stages(key=key)
    return to_grid(stages, ih=model.K, iw=model.N_nca_steps + 1)


@eqx.filter_jit
def log_nca_stages(model: NonDoublingVNCA, ih: int = 4, iw: int = 9, *, key: PRNGKeyArray) -> Array:
    stages = model.nca_stages(T=ih * iw, key=key)
    return to_grid(stages, ih=ih, iw=iw)
