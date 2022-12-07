# %%
from jax import lax
from jax import numpy as np
from distrax import Bernoulli
from functools import partial

from data import binarized_mnist
from keras.datasets import mnist

# typing
from jax import Array
from typing import Tuple, Any


# %%
# Load the binarized MNIST data
train_data_binarized, test_data_binarized = binarized_mnist.get_data(pad=0)
train_data_binarized = train_data_binarized.squeeze()
test_data_binarized = test_data_binarized.squeeze()


# %%
# Load the non-binarized MNIST data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data / 255.0
test_data = test_data / 255.0
train_data = np.array(train_data, dtype=np.float32)
test_data = np.array(test_data, dtype=np.float32)


# %%
assert train_data_binarized.shape == train_data.shape
assert test_data_binarized.shape == test_data.shape
train_data_binarized.shape, train_data.shape


# %%
def get_label(_, binary_image: Array, non_binarized_images: Array) -> Tuple[Any, Array]:
    log_probs = Bernoulli(probs=non_binarized_images).log_prob(value=binary_image)
    log_prob = np.sum(log_probs, axis=(1, 2))
    i = np.argmax(log_prob)
    return (_, i)


# %%
_, train_idx = lax.scan(partial(get_label, non_binarized_images=train_data), 0, train_data_binarized)
np.save('./data/raw/binarized_mnist/labels/train_label_indices', train_idx)
train_labels[train_idx]


# %%
train_idx = np.load('./data/raw/binarized_mnist/labels/train_label_indices.npy')
binarized_train_labels = train_labels[train_idx]
np.save('./data/raw/binarized_mnist/labels/binarized_train_labels', binarized_train_labels)


# %%
_, test_idx = lax.scan(partial(get_label, non_binarized_images=test_data), 0, test_data_binarized)
np.save('./data/raw/binarized_mnist/labels/test_label_indices', test_idx)
test_labels[test_idx]


# %%
test_idx = np.load('./data/raw/binarized_mnist/labels/test_label_indices.npy')
binarized_test_labels = test_labels[test_idx]
np.save('./data/raw/binarized_mnist/labels/binarized_test_labels', binarized_test_labels)


# %%
import numpy as onp
from PIL import Image
from tqdm import tqdm
from einops import rearrange


# %%
# Save the images side by side to disk for inspection
def save_imgs(data_binarized: Array, data: Array, labels: Array, idx: Array, split: str):
    n = len(idx)
    image_pairs = np.stack([data_binarized, data[idx]], axis=1)
    image_pairs = rearrange(image_pairs, 'n p h w -> n h (p w)')  # p = 2, n = 10000, h = 28, w = 28

    for i, image_pair, label in tqdm(zip(range(n), image_pairs, labels[idx])):
        image_pair = onp.array(image_pair * 255, dtype=onp.uint8)
        image_pair = Image.fromarray(image_pair)
        image_pair.save(f'./data/raw/binarized_mnist/labels/images/{split}/{i:05d}_{label}.png')


# %%
save_imgs(train_data_binarized, train_data, train_labels, train_idx, 'train')


# %%
save_imgs(test_data_binarized, test_data, test_labels, test_idx, 'test')
