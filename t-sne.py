# %%
from IPython import get_ipython

get_ipython().system('git clone https://github.com/albertaillet/vnca.git')


# %%
get_ipython().system('pip install -r /kaggle/working/vnca/requirements.txt')


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
from jax import random, lax
from sklearn.manifold import TSNE
from functools import partial
import matplotlib.pyplot as plt

from data import get_data
from log_utils import restore_model
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA

# typing
from jax import Array
from typing import Tuple, Any

DATA_KEY = random.PRNGKey(0)


# %%
# Load the binarized data and labels
_, test_data = get_data('binarized_mnist')
test_labels = np.load('/kaggle/input/labels/binarized_test_labels.npy')


# %%
# Load and restore model
model = BaselineVAE(key=random.PRNGKey(0), latent_size=256)
model = restore_model(model, 'BaselineVAE_gstep10000.eqx', run_path='dladv-vnca/vnca/runs/h8xyupys')


# %%
# Get the latent samples
def get_latent_sample(carry: Any, x: Array, *, model: AutoEncoder) -> Tuple[Any, Array]:
    mean, _ = model.encoder(x)
    return carry, mean

_, z_means = lax.scan(partial(get_latent_sample, model=model), 0, test_data)


# %%
# Take a random subset of the latent samples
n = 5_000
indices = np.arange(len(z_means))
indices = random.permutation(DATA_KEY, indices)
z_means = z_means[indices][:n]
test_labels = test_labels[indices][:n]


# %%
# Run t-SNE to reduce the dimensionality of the latent space
tsne = TSNE(n_components=2)
z_means_tsne = tsne.fit_transform(z_means)


# %%
# Plot the t-SNE reduced latent space with corresponding label color
def show_latent_space(data: Array, labels: Array, n: int = 5_000):
    if n < len(labels):
        indices = np.arange(len(labels))
        indices = random.permutation(DATA_KEY, indices)
        data = data[indices][:n]
        labels = labels[indices][:n]

    plt.figure(figsize=(10, 10))
    plt.scatter(
        data[:, 0],
        data[:, 1],
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

show_latent_space(z_means_tsne, test_labels, n=5_000)


# %%


