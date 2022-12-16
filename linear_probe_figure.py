# %%
# Imports
from jax import vmap
from jax import numpy as np
from jax.random import PRNGKey
from jax.nn import sigmoid, softmax
import equinox as eqx
from einops import rearrange
import matplotlib.pyplot as plt

from log_utils import restore_model
from models import NonDoublingVNCA


MODEL_KEY = PRNGKey(0)


# %%
# Load and restore model
model = NonDoublingVNCA(key=MODEL_KEY, latent_size=128)
model = restore_model(model, "NonDoublingVNCA_gstep100000.eqx", run_path="dladv-vnca/vnca/runs/3k9mouaj")


# %%
probe = eqx.nn.Linear(128, 10, key=MODEL_KEY)
probe = eqx.tree_deserialise_leaves('128-probe.eqx', probe)


# %%
n = 9
keys = vmap(PRNGKey)(np.arange(n))
x_out = vmap(model.sample)(key=keys)

# %%
x = rearrange(x_out, "n c h w -> (n h w) c", h=32, w=32, c=128, n=n)
probed = vmap(probe)(x)
imgs = rearrange(x, "(n h w) c -> c n h w", h=32, w=32, c=128, n=n)[0]
probed = rearrange(probed, "(n h w) p -> n p h w", h=32, w=32, p=10, n=n)

# %%
n_imgs = 7
fig, axs = plt.subplots(3, n_imgs, figsize=(3 * n_imgs, 9))
for i, img, pred in zip(range(n_imgs), imgs[1:], probed[1:]):
    img = sigmoid(img)
    pred = softmax(pred, axis=0)
    pred_class = np.argmax(pred, axis=0)
    pred_prob = np.max(pred, axis=0)

    axs[0, i].imshow(img, cmap="gray")
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])

    axs[1, i].imshow(pred_class, cmap="tab10", vmin=-0.5, vmax=9.5)
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])

    axs[2, i].imshow(pred_prob, cmap="viridis", vmin=0, vmax=1)
    axs[2, i].set_xticks([])
    axs[2, i].set_yticks([])

# add colorbar for the whole figure
fig.subplots_adjust(right=0.8)
prob_cbar = fig.add_axes([1.005, 0.365, 0.01, 0.27])
fig.colorbar(axs[1, 0].get_images()[0], cax=prob_cbar, ticks=np.arange(10))

prob_cbar = fig.add_axes([1.005, 0.04, 0.01, 0.28])
fig.colorbar(axs[2, 0].get_images()[0], cax=prob_cbar, ticks=np.linspace(0, 1, 6))


plt.tight_layout()
plt.show()
