# %%
# Imports
from jax import vmap
from jax.nn import sigmoid
from jax import numpy as np
from jax.random import PRNGKey, normal
from models import BaselineVAE, DoublingVNCA, NonDoublingVNCA
from log_utils import restore_model, to_grid
import matplotlib.pyplot as plt

SAMPLE_KEY = PRNGKey(0)


# %%
# Load and restore model
vnca_model = DoublingVNCA(key=SAMPLE_KEY, latent_size=256)
vnca_model = restore_model(vnca_model, 'DoublingVNCA_gstep100000.eqx', run_path='dladv-vnca/vnca/runs/14c2aulc')


# %%
# Interpolate between two random latent vectors and plot
n = 8
fig, axs = plt.subplots(4, 1, figsize=(20, 10))
for i in range(4):
    z_1, z_2 = normal(PRNGKey(i), (2, 256))  # Sample two latent vectors
    d = (z_2 - z_1) / n
    interpolate = lambda i: z_1 + i * d

    z = vmap(interpolate)(np.arange(n))  # Interpolate the latent vectors
    out = vmap(vnca_model.decoder)(z)  # Decode
    out = out[:, :1, :, :]  # Remove channel dimension
    out = sigmoid(out)  # Sigmoid to get pixel values
    img = to_grid(out, ih=1, iw=n)  # Convert to grid
    axs[i].imshow(img[0], cmap='gray')
    axs[i].axis('off')

plt.tight_layout()
plt.show()
