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

import equinox as eqx
from jax.random import PRNGKey
from jax import lax
from einops import rearrange

from data import get_data
from loss import iwae_loss
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA
from log_utils import restore_model

# typing
from jax import Array

MODEL_KEY = PRNGKey(0)
TEST_KEY = PRNGKey(2)
K = 128
BATCH_SIZE = 8


# %%
# function to compute the IWELBO loss on the test set
@partial(eqx.filter_pmap, axis_name='nd', args=(0, None, None, None))
def test_iwelbo(batched_data: Array, model: AutoEncoder, key: Array, K: int):
    return lax.pmean(lax.map(partial(iwae_loss, model, key=TEST_KEY, K=K), batched_data).mean(), 'nd')


# %%
_, test_data = get_data('binarized_mnist')
batched_data = rearrange(test_data, "(p k b) c h w -> p k b c h w", p=8, b=5)
batched_data.shape


# %%
model = BaselineVAE(key=MODEL_KEY, latent_size=256)
model = restore_model(model, 'BaselineVAE_gstep100000.eqx', run_path='dladv-vnca/vnca/runs/h8xyupys')
params, static = eqx.partition(model, eqx.is_array)


# %%
test_iwelbo(batched_data, model=model, key=TEST_KEY, K=128)[0]
get_ipython().run_line_magic('time', 'test_iwelbo(batched_data,model=model,key=TEST_KEY,K=128)[0]')


# %%
_ = eqx.filter_jit(model)(test_data[0], key=TEST_KEY)
get_ipython().run_line_magic('time', '_ = eqx.filter_jit(model)(test_data[0],key=TEST_KEY)')


# %%
model = DoublingVNCA(key=MODEL_KEY, latent_size=256)
model = restore_model(model, 'DoublingVNCA_gstep100000.eqx', run_path='dladv-vnca/vnca/runs/14c2aulc')


# %%
test_iwelbo(batched_data, model=model, key=TEST_KEY, K=128)[0]
get_ipython().run_line_magic('time', 'test_iwelbo(batched_data,model=model,key=TEST_KEY,K=128)[0]')


# %%
_ = eqx.filter_jit(model)(test_data[0], key=TEST_KEY)
get_ipython().run_line_magic('time', '_ = eqx.filter_jit(model)(test_data[0],key=TEST_KEY)')


# %%
model = NonDoublingVNCA(key=MODEL_KEY, latent_size=128)
model = restore_model(model, "NonDoublingVNCA_gstep100000.eqx", run_path="dladv-vnca/vnca/runs/1mmyyzbu")


# %%
test_iwelbo(batched_data, model=model, key=TEST_KEY, K=128)[0]
get_ipython().run_line_magic('time', 'test_iwelbo(batched_data,model=model,key=TEST_KEY,K=128)[0]')


# %%
_ = eqx.filter_jit(model)(test_data[0], key=TEST_KEY)
get_ipython().run_line_magic('time', '_ = eqx.filter_jit(model)(test_data[0],key=TEST_KEY)')
