# %%
from IPython import get_ipython

# get_ipython().system('git clone https://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@github.com/albertaillet/vnca.git -b log-outputs')
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


# %%
get_ipython().run_cell_magic('capture', '', '%pip install --upgrade tensorflow_probability tensorflow jax jaxlib numpy equinox einops optax distrax scikit-learn keras')


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
from jax import numpy as np
from jax import random, lax, vmap
from sklearn.manifold import TSNE
from functools import partial

import numpy as onp
import matplotlib.pyplot as plt

from data.mnist import get_mnist
from keras.datasets.mnist import load_data

import equinox as eqx
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA, sample_gaussian

# typing
from jax import Array
from jax.random import PRNGKeyArray
from typing import Tuple, Any

DATA_KEY = random.PRNGKey(0)


# %%
# Load the data
_, test_data = get_mnist()
(_, (_, test_labels)) = load_data()
binarized_test_indices = np.load('../../input/tsne-20221207-ver2/binarized_test_labels.npy')
test_labels = test_labels[binarized_test_indices]


# %%
# Get a subset of the data to run t-SNE on
n = 5000
indices = np.arange(len(test_data))
indices = random.permutation(DATA_KEY, indices)[:n]
test_data_subset = test_data[indices]
test_labels_subset = test_labels[indices]
keys = random.split(DATA_KEY, n)


# %%
# Load the model
vnca_model = DoublingVNCA(latent_size=256, key=random.PRNGKey(0))
vnca_model = eqx.tree_deserialise_leaves('../../input/tsne-20221207/DoublingVNCA_gstep60000.eqx', vnca_model)


# %%
def get_latent_sample(_, t: Tuple[Array, PRNGKeyArray], *, model: AutoEncoder) -> Tuple[Any, Array]:
    x, key = t
    mean, logvar = model.encoder(x)
    z = sample_gaussian(mean, logvar, shape=mean.shape, key=key)
    return _, z


# %%
# Get the latent samples
_, z = lax.scan(partial(get_latent_sample, model=vnca_model), 0, (test_data_subset, keys))


# %%
# Run t-SNE
tsne = TSNE(n_components=2)
z_tsne = tsne.fit_transform(z)


# %%
# Plot the t-SNE reduced latent space with corresponding label color
def show_latent_space(z: Array, labels: list):
    plt.figure(figsize=(10, 10))
    plt.scatter(
        z[:, 0],
        z[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.69,
        vmin=np.min(labels) - 0.5,
        vmax=np.max(labels) + 0.5,
    )
    plt.colorbar(
        ticks=np.arange(np.min(labels), np.max(labels) + 1),
    )
    plt.axis('off')
    plt.show()


show_latent_space(z_tsne, test_labels)


# %%
