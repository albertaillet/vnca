# %%
# Imports
from jax.nn import sigmoid
from jax import numpy as np
from jax.random import PRNGKey
from models import BaselineVAE, DoublingVNCA, NonDoublingVNCA
from einops import rearrange
from log_utils import restore_model

from matplotlib import cm
from matplotlib import pyplot as plt


SAMPLE_KEY = PRNGKey(16)


# %%
# Load and restore model
vnca_model = DoublingVNCA(key=SAMPLE_KEY, latent_size=256)
vnca_model = restore_model(vnca_model, 'DoublingVNCA_gstep100000.eqx', run_path='dladv-vnca/vnca/runs/14c2aulc')

# %%
# Get one sample
x = sigmoid(vnca_model.sample(key=SAMPLE_KEY))

# %%
# Plot
z, h, w = x.shape
H = np.arange(h)
W = np.arange(w)
H, W = np.meshgrid(H, W)
Z = np.zeros_like(H)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(7):
    im = rearrange(x[i], 'h w -> w h')
    cmap = cm.gray if i == 0 else cm.bone
    ax.plot_surface(H, W, Z - i * 0.2, rstride=1, cstride=1, facecolors=cmap(im))
    ax.axis('off')
ax.view_init(azim=25)
plt.show()

# %%
