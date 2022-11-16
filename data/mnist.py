import jax.numpy as np
from jax import Array
from jax.random import split, PRNGKeyArray, permutation, PRNGKey
from einops import rearrange
from torchvision.datasets import MNIST


ROOT = './data'

# typing
from typing import Iterator, Tuple


def get_indices(n: int, batch_size: int, key: PRNGKeyArray) -> Array:
    '''Get random indices for a batch.'''
    indices = np.arange(n)  # [0, 1, 2, ..., len(dataset)]
    indices = permutation(key, indices)  # shuffle the indices
    indices = indices[: (n // batch_size) * batch_size]  # drop the last few samples not in a batch
    return indices  # reshape into (n_batches, batch_size)


def load_mnist(batch_size: int, key: PRNGKeyArray) -> Tuple[Iterator, Iterator]:
    '''Load MNIST dataset.'''
    train_dataset = MNIST(root=ROOT, train=True, download=True, transform=None)
    test_dataset = MNIST(root=ROOT, train=False, download=True, transform=None)

    train_dataset = np.float32(train_dataset) / 255.0  # rescale to [0, 1]
    test_dataset = np.float32(test_dataset) / 255.0

    train_dataset = rearrange(train_dataset, 'n h w -> n 1 h w')  # add channel dimension
    test_dataset = rearrange(test_dataset, 'n h w -> n 1 h w')

    def dataset_iterator(dataset: Array, batch_size: int, key: PRNGKeyArray) -> Iterator:
        n = len(dataset)
        while True:
            key, subkey = split(key)
            for batch_indices in get_indices(n, batch_size, subkey):
                yield dataset[batch_indices]

    return dataset_iterator(train_dataset, batch_size, key), dataset_iterator(test_dataset, batch_size, key)


def load_mnist_train_on_tpu(devices: list) -> Array:
    '''Load MNIST dataset to TPU.'''
    from jax import device_put_replicated

    train_dataset = MNIST(root=ROOT, train=True, download=True, transform=None)

    train_dataset = np.float32(train_dataset.data) / 255.0  # rescale to [0, 1]

    train_dataset = rearrange(train_dataset, 'n h w -> n 1 h w')  # add channel dimension

    return device_put_replicated(train_dataset, devices)


def indicies_tpu_iterator(n_tpus: int, batch_size: int, dataset_size: int, gradient_steps: int, key: PRNGKeyArray, device_iterations: int):
    '''Get random indices for a batch on TPU.'''
    for _ in range(gradient_steps):
        n_batches_per_device_iteration = (device_iterations // (dataset_size // (batch_size * n_tpus))) + 1

        key, *keys = split(key, n_batches_per_device_iteration + 1)

        indices = np.concatenate([get_indices(dataset_size, batch_size * n_tpus, index_key) for index_key in keys])

        yield rearrange(indices[: batch_size * n_tpus * device_iterations], '(t l b) -> t l b', b=batch_size, t=n_tpus, l=device_iterations)


if __name__ == '__main__':
    train_dataset, test_dataset = load_mnist(32, PRNGKey(0))
    print(next(train_dataset).shape)
    print(next(test_dataset).shape)
