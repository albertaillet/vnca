# %%
# Imports
import equinox as eqx
from data import get_data
from jax.random import PRNGKey, PRNGKeyArray, split
from jax.nn import sigmoid
from jax import Array, jit
from models import AutoEncoder, BaselineVAE, DoublingVNCA, sample_gaussian, pad, double
from einops import rearrange
from functools import partial

SAMPLE_KEY = PRNGKey(0)


# %%
# Load the test set
_, test_data = get_data(dataset='binarized_mnist')


# %%
# Load and restore model
vnca_model = DoublingVNCA(key=SAMPLE_KEY, latent_size=256)
vnca_model = eqx.tree_deserialise_leaves('DoublingVNCA_gstep100000.eqx', vnca_model)


# %%
# Sample from the model, and show the growth stages
def growth_stages(model: DoublingVNCA, *, key: PRNGKeyArray) -> Array:
    mean = np.zeros(model.latent_size)
    logvar = np.zeros(model.latent_size)
    z = sample_gaussian(mean, logvar, (model.latent_size,), key=key)

    # Add height and width dimensions
    z = rearrange(z, 'c -> c 1 1')

    def process(z: Array) -> Array:
        '''Process a latent sample by taking the image channels, applying sigmoid and padding.'''
        logits = z[:1]
        probs = sigmoid(logits)
        while probs.shape[1] != 32:
            probs = double(probs)
        return probs

    # Decode the latent sample and save the processed image channels
    stages_probs = []
    for _ in range(model.K):
        z = model.double(z)
        stages_probs.append(process(z))
        for _ in range(model.N_nca_steps):
            z = z + model.step(z)
            stages_probs.append(process(z))

    return np.array(stages_probs)

# %%
ih, iw = 5, 10
keys = split(SAMPLE_KEY, ih * iw)
stages = np.array([growth_stages(vnca_model, key=key) for key in keys])


# %%
stages = rearrange(stages, '(ih iw) t c h w -> t c (ih h) (iw w)', ih=ih, iw=iw)

# %%
# make a gif:
import numpy as np
from PIL import Image

def to_PIL_img(x: np.ndarray) -> Image:
    '''Converts an array of shape (c, h, w) to a PIL Image'''
    x = np.clip(x, 0, 1)
    return Image.fromarray(np.array(255 * x, dtype=np.uint8).squeeze())

images = [to_PIL_img(stage) for stage in stages]


# %%
# make the gif that loops forever
images[0].save('vnca.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
# %%
