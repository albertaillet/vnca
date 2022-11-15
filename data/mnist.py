import jax.numpy as np
from jax import Array
from jax.random import split, PRNGKeyArray, permutation, PRNGKey
from einops import rearrange

from tensorflow.keras.datasets import mnist

# typing
from typing import Iterator, Tuple


def load_mnist(batch_size: int, key: PRNGKeyArray) -> Tuple[Iterator, Iterator]:
    (train_dataset, _), (test_dataset, _) = mnist.load_data()

    train_dataset = np.float32(train_dataset) / 255.0
    test_dataset = np.float32(test_dataset) / 255.0

    train_dataset = rearrange(train_dataset, 'n h w -> n 1 h w')
    test_dataset = rearrange(test_dataset, 'n h w -> n 1 h w')

    def get_indices(n: int, key: PRNGKeyArray) -> Array:
        indices = np.arange(n)  # [0, 1, 2, ..., len(dataset)]
        indices = permutation(key, indices)  # shuffle the indices
        indices = indices[: (n // batch_size) * batch_size]  # drop the last few samples not in a batch
        return indices

    def dataset_iterator(dataset: Array, key: PRNGKeyArray) -> Iterator:
        n = len(dataset)
        while True:
            key, subkey = split(key)
            for batch_indices in rearrange(get_indices(n, subkey), '(n b) -> n b', b=batch_size):
                yield dataset[batch_indices]

    return dataset_iterator(train_dataset, key), dataset_iterator(test_dataset, key)


def load_mnist_tpu():
    (train_dataset, _) = mnist.load_data()

    train_dataset = np.float32(train_dataset) / 255.0
    train_dataset = rearrange(train_dataset, 'n h w -> n 1 h w')

    from jax import devices, device_put_replicated

    return device_put_replicated(train_dataset, devices())


if __name__ == '__main__':
    train_dataset, test_dataset = load_mnist(32, PRNGKey(0))
    print(next(train_dataset).shape)
    print(next(test_dataset).shape)
