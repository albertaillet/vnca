# %%
# Imports
import equinox as eqx
import jax.numpy as np
from numpy import uint8
from jax import Array, vmap, jit
from jax.random import PRNGKey, PRNGKeyArray, split, normal
from jax.nn import sigmoid
from models import AutoEncoder, BaselineVAE, DoublingVNCA, sample_gaussian, pad, double
from einops import rearrange
from IPython.display import Image
from tqdm import tqdm
from PIL import Image as PILImage
from functools import partial

def to_PIL_img(x: np.ndarray) -> PILImage:
    '''Converts an array of shape (c, h, w) to a PIL Image'''
    x = np.clip(x, 0, 1)
    return PILImage.fromarray(uint8(255 * x).squeeze())
SAMPLE_KEY = PRNGKey(0)


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
stages_rearranged = rearrange(stages, '(ih iw) t c h w -> t c (ih h) (iw w)', ih=ih, iw=iw, c=1, h=32, w=32)
images = [to_PIL_img(s) for s in stages_rearranged]
images[0].save('vnca_stages.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
Image(open('vnca_stages.gif','rb').read())

# %%
# Interpolate between two random latent vectors and plot
def get_interpolation(n: int, *, key: PRNGKeyArray) -> Array:
    z_1, z_2 = normal(key, (2, 256))  # Sample two latent vectors
    d = (z_2 - z_1) / n
    interpolate = lambda i: z_1 + i * d

    z = vmap(interpolate)(np.arange(n))  # Interpolate the latent vectors
    logits = vmap(vnca_model.decoder)(z)  # Decode
    logits = logits[:, :1]  # Take the image channels
    probs = sigmoid(logits)  # Sigmoid to get pixel values
    return probs


# %%
ih, iw, n = 5, 10, 10
keys = split(SAMPLE_KEY, ih * iw)
interpolate = jit(partial(get_interpolation, n=n))
interpolations = np.array([interpolate(key=key) for key in tqdm(keys)])

# %%
# make it loop back and forth
interpolations_rearranged = rearrange(interpolations, '(ih iw) n c h w -> n c (ih h) (iw w)', ih=ih, iw=iw, n=n, c=1, h=32, w=32)
images = [to_PIL_img(i) for i in interpolations_rearranged]
images[0].save('vnca_interpolation.gif', save_all=True, append_images=images[1:] + images[::-1], duration=100, loop=0)
Image(open('vnca_interpolation.gif','rb').read())

# %%
