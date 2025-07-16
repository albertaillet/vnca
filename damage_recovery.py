# %%
# This script demonstrates the damage recovery capabilities of the NonDoublingVNCA model
from jax import vmap
from jax.random import PRNGKey, split
from models import NonDoublingVNCA
from matplotlib import pyplot as plt
from log_utils import restore_model, to_grid, to_PIL_img
from functools import partial

# %%
# Load the model
vnca_model = NonDoublingVNCA(latent_size=128, key=PRNGKey(0))
vnca_model = restore_model(vnca_model, 'NonDoublingVNCA_gstep100000.eqx', run_path='dladv-vnca/vnca/runs/1mmyyzbu')


# %%
# Generate 80 steps and damage the 40th
ih, iw = 10, 8
T = ih * iw
damage_idx = {T // 2}
key = PRNGKey(2)
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
key = PRNGKey(0)
keys = split(key, num=11)
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

# %%
# Generate 80 steps and damage the 40th
ih, iw = 10, 8
T = 140
damage_idx = {30, 70}
N = ih * iw
key = split(PRNGKey(2), num=N)
out = vmap(partial(vnca_model.nca_stages, n_channels=1, T=T, damage_idx=damage_idx))(key=key)

# %%
array_of_grids = vmap(partial(to_grid, ih=ih, iw=iw), in_axes=1)(out)
# %%
images = [to_PIL_img(grid) for grid in array_of_grids]
images[0].save('images/recovery.gif', save_all=True, append_images=images, loop=0)

# %%
