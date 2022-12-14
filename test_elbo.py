# %%
# This script computes the IWELBO loss on the test set of the Binarized MNIST dataset, run on the kaggle TPUs
from IPython import get_ipython

get_ipython().system('git clone https://github.com/albertaillet/vnca.git')


# %%
get_ipython().run_cell_magic('capture', '', '%pip install --upgrade jax tensorflow_probability tensorflow jaxlib numpy equinox einops optax distrax wandb datasets')


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
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


# %%
from functools import partial

import jax.numpy as np
from jax.random import PRNGKey, split
from jax import lax
from jax import pmap, local_devices
from einops import rearrange

from data import load_data_on_tpu
from loss import forward
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA
from log_utils import restore_model

# typing
from jax import Array
from jax.random import PRNGKeyArray

MODEL_KEY = PRNGKey(0)
TEST_KEY = PRNGKey(2)


# %%
# Load and restore model
vnca_model = BaselineVAE(key=MODEL_KEY, latent_size=256)
vnca_model = restore_model(vnca_model, 'BaselineVAE_gstep100000.eqx', run_path='dladv-vnca/vnca/runs/h8xyupys')


# %%
# function to compute the IWELBO loss on the test set
def test_iwelbo(key: PRNGKeyArray, x: Array, model: AutoEncoder, M: int):
    key, subkey = split(key)
    loss = forward(model, x, subkey, M=M)
    return key, np.mean(loss)


# %%
# Load the test set on the TPU
_, test_data = load_data_on_tpu(devices=local_devices(), dataset='binarized_mnist', key=TEST_KEY)
test_data_tpu = rearrange(test_data, 't (b n) c h w -> n t b c h w', b=10)
test_data_tpu.shape


# %%
# Get the IWELBO loss on the test set
_, losses = lax.scan(
    pmap(partial(test_iwelbo, model=vnca_model, M=128), axis_name='num_devices'), 
    split(TEST_KEY, 8),
    test_data_tpu
)


# %%
# Print the IWELBO loss
np.mean(losses)

