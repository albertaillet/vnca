import jax.numpy as np

# typing
from jax import Array
from typing import Tuple


def get_data() -> Tuple[Array, Array]:
    '''Get CelebA dataset.
    The data is downloaded using Torchvision.
    The dataset is resized to 64x64.'''

    # dummy data
    train_data = np.zeros((162_770, 3, 64, 64), dtype=np.float32)
    test_data = np.zeros((39_829, 3, 64, 64), dtype=np.float32)

    return train_data, test_data
