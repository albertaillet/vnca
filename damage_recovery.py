# %%
# This script demonstrates the damage recovery capabilities of the NonDoublingVNCA model
from jax import random
from models import NonDoublingVNCA
from matplotlib import pyplot as plt
from equinox import tree_deserialise_leaves
from log_utils import to_grid, to_PIL_img


# %%
# Load the model
vnca_model = NonDoublingVNCA(latent_size=128, key=random.PRNGKey(0))
vnca_model: NonDoublingVNCA = tree_deserialise_leaves('models/NonDoublingVNCA_gstep100000.eqx', vnca_model)


# %%
# Generate 100 steps and damage the 50th
ih, iw = 10, 10
T = ih * iw
damage_idx = {50}
key = random.PRNGKey(1)
out = vnca_model.nca_stages(n_channels=1, T=T, damage_idx=damage_idx, key=key)


# %%
# Convert to a Image grid
to_PIL_img(to_grid(out, ih=ih, iw=iw))


# %%
# Recover the damage and save the original, damaged and recovered images
T = 36 * 2
damage_idx = {36}
original = []
damaged = []
recovered = []
key = random.PRNGKey(0)
keys = random.split(key, num=10)
for key in keys:
    out = vnca_model.nca_stages(n_channels=1, T=T, damage_idx=damage_idx, key=key)
    original.append(out[35])
    damaged.append(out[36])
    recovered.append(out[T - 1])


# %%
# Plot the results
ih = 1
iw = len(keys)
original_img = to_PIL_img(to_grid(original, ih=ih, iw=iw))
damaged_img = to_PIL_img(to_grid(damaged, ih=ih, iw=iw))
recovered_img = to_PIL_img(to_grid(recovered, ih=ih, iw=iw))
plt.figure(figsize=(20, 6))
for i, img, title in zip([1, 2, 3], [original_img, damaged_img, recovered_img], ['original', 'damaged', 'recovered']):
    plt.subplot(3, 1, i)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=30, fontname='serif', y=-0.35)
    plt.axis('off')
plt.tight_layout()
plt.show()
