# %%
from IPython import get_ipython

get_ipython().system('git clone https://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@github.com/albertaillet/vnca.git -b log-outputs')


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
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


# %%
from jax import numpy as np
from jax import random, lax, vmap
from sklearn.manifold import TSNE
from functools import partial

import numpy as onp
import matplotlib.pyplot as plt

from data import binarized_mnist

import equinox as eqx
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA, sample_gaussian

# typing
from jax import Array
from jax.random import PRNGKeyArray
from typing import Tuple, Any

DATA_KEY = random.PRNGKey(0)


# %%
# Load the data
_, test_data = binarized_mnist.get_data()
test_labels = np.load('../../input/tsne-20221207-ver2/binarized_test_labels.npy')


# %%
# Load the model
vnca_model = DoublingVNCA(latent_size=256, key=random.PRNGKey(0))
vnca_model = eqx.tree_deserialise_leaves('../../input/tsne-20221207/DoublingVNCA_gstep60000.eqx', vnca_model)


# %%
def get_latent_sample(c, x: Array, *, model: AutoEncoder) -> Tuple[Any, Array]:
    mean, _ = model.encoder(x)
    return c, mean


# %%
# Get the latent samples
_, z_means = lax.scan(partial(get_latent_sample, model=vnca_model), 0, test_data)


# %%
# Run t-SNE
tsne = TSNE(n_components=2)
z_means_tsne = tsne.fit_transform(z_means)


# %%
# Plot the t-SNE reduced latent space with corresponding label color
def show_latent_space(data: Array, labels: Array, n: int = 5_000):
    if n < len(labels):
        indices = np.arange(len(labels))
        indices = random.permutation(DATA_KEY, indices)
        data = data[indices]
        labels = labels[indices]

    plt.figure(figsize=(10, 10))
    plt.scatter(
        data[:n, 0],
        data[:n, 1],
        c=labels[:n],
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

show_latent_space(z_means_tsne, test_labels, n=5_000)


# %%


