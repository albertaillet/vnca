# %%
from IPython import get_ipython

get_ipython().system('git clone https://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@github.com/albertaillet/vnca.git -b log-outputs')
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


# %%
get_ipython().run_cell_magic('capture', '', '%pip install --upgrade jax jaxlib numpy equinox einops optax distrax scikit-learn')

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
binarized_test_indices = np.load('./data/raw/binarized_mnist/labels/test_label_indices.npy')
test_labels = test_labels[binarized_test_indices]


# %%
# Get a subset of the data to run t-SNE on
n = 250
indices = np.arange(len(test_data))
indices = random.permutation(DATA_KEY, indices)[:n]
test_data = test_data[indices]
test_labels = test_labels[indices]
keys = random.split(DATA_KEY, n)


# %%
# Load the model
vnca_model = DoublingVNCA(latent_size=256, key=random.PRNGKey(0))
vnca_model = eqx.tree_deserialise_leaves('models/DoublingVNCA_gstep60000.eqx', vnca_model)


# %%
def get_latent_sample(_, t: Tuple[Array, PRNGKeyArray], *, model: AutoEncoder) -> Tuple[Any, Array]:
    x, key = t
    mean, logvar = model.encoder(x)
    z = sample_gaussian(mean, logvar, shape=mean.shape, key=key)
    return _, z


# %%
# Get the latent samples
_, z = lax.scan(partial(get_latent_sample, model=vnca_model), 0, (test_data, keys))


# %%
z = onp.array(z)
onp.save('./z.npy', z)

# %%
z = onp.load('./z.npy')

# %%
# Run t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250, random_state=0, learning_rate=100, init='pca')
z_tsne = tsne.fit_transform(z)


# %%
# Plot the t-SNE
def show_latent_space(z: Array):
    plt.figure(figsize=(10, 10))
    plt.scatter(z[:, 0], z[:, 1], c=test_labels, cmap='tab10')
    cbar = plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # colorbar ticks in the middle of the colorbar
    cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.show()


show_latent_space()
