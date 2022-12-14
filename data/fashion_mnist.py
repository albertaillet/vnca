import jax.numpy as np
from jax.random import PRNGKey, bernoulli
from einops import rearrange
import numpy as onp

from datasets import load_dataset  # HuggingFace datasets

# typing
from jax import Array
from typing import Tuple


BINARIZATION_KEY = PRNGKey(42)


def get_data(binarized: bool, pad: int = 2) -> Tuple[Array, Array]:
    '''Get FASHION-MNIST dataset or a binarized FASHION-MNIST dataset.
    The data is downloaded from HuggingFace datasets.
    The dataset is padded with zeros.
    If binarized is True, the images are binarized using bernoulli sampling with a set seed of 42.'''

    dataset = load_dataset('fashion_mnist')

    train_data = np.array([onp.array(img) for img in dataset['train']['image']])
    test_data = np.array([onp.array(img) for img in dataset['test']['image']])

    # Pad the images with zeros
    train_data = np.pad(train_data, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    test_data = np.pad(test_data, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Reshape the images to (n, c, h, w)
    train_data = rearrange(train_data, '(n c) h w -> n c h w', n=60_000, c=1, h=32, w=32).astype(np.float32)
    test_data = rearrange(test_data, '(n c) h w -> n c h w', n=10_000, c=1, h=32, w=32).astype(np.float32)

    # Normalize the images to [0, 1]
    train_data = train_data / 255.
    test_data = test_data / 255.

    if binarized:
        # Binarize the images using bernoulli sampling
        train_data = bernoulli(BINARIZATION_KEY, p=train_data).astype(np.float32)
        test_data = bernoulli(BINARIZATION_KEY, p=test_data).astype(np.float32)

    return train_data, test_data
