import jax.numpy as np
from jax.random import split, PRNGKeyArray, permutation
from jax import device_put_replicated, device_put_sharded
from einops import rearrange


# typing
from jax import Array
from typing import Iterator, Tuple


def get_data(dataset: str = 'binarized_mnist', *args, **kwargs) -> Tuple[Array, Array]:
    if dataset == 'binarized_mnist':
        import data.binarized_mnist as binarized_mnist

        return binarized_mnist.get_data(*args, **kwargs)
    elif dataset == 'fashion_mnist':
        import data.fashion_mnist as fashion_mnist

        return fashion_mnist.get_data(binarized=False, *args, **kwargs)
    elif dataset == 'binarized_fashion_mnist':
        import data.fashion_mnist as fashion_mnist

        return fashion_mnist.get_data(binarized=True, *args, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {dataset}')


def get_indices(n: int, batch_size: int, key: PRNGKeyArray) -> Array:
    '''Get random indices for a batch.'''
    indices = np.arange(n)  # [0, 1, 2, ..., len(dataset)]
    indices = permutation(key, indices)  # shuffle the indices
    indices = indices[: (n // batch_size) * batch_size]  # drop the last few samples not in a batch
    return indices


def load_data(batch_size: int, dataset: str = 'binarized_mnist', *, key: PRNGKeyArray) -> Tuple[Iterator, Iterator]:
    '''Load binarized MNIST dataset.'''
    train_dataset, test_dataset = get_data(dataset, pad=2)

    def dataset_iterator(dataset: Array, batch_size: int, key: PRNGKeyArray) -> Iterator:
        n = len(dataset)
        while True:
            key, subkey = split(key)
            indices = get_indices(n, batch_size, subkey),
            for batch_indices in rearrange(indices, '(n b) -> n b', b=batch_size):  # reshape into (n_batches, batch_size)
                yield dataset[batch_indices]

    return dataset_iterator(train_dataset, batch_size, key), dataset_iterator(test_dataset, batch_size, key)


def load_data_on_tpu(devices: list, dataset: str = 'binarized_mnist', *, key: PRNGKeyArray) -> Tuple[Array, Array]:
    '''Load binarized MNIST dataset to TPU.
    The training set is replicated across all devices and the test set is sharded across all devices.
    '''
    train_dataset, test_dataset = get_data(dataset, pad=2)

    test_dataset = permutation(key, test_dataset, axis=0)

    shard = [*rearrange(test_dataset, '(t s) c h w -> t s c h w', t=len(devices))]

    return device_put_replicated(train_dataset, devices), device_put_sharded(shard, devices)


def indicies_tpu_iterator(n_tpus: int, batch_size: int, dataset_size: int, gradient_steps: int, key: PRNGKeyArray, device_iterations: int):
    '''Get random indices for a batch on TPU.'''
    for _ in range(gradient_steps):
        n_batches_per_device_iteration = (device_iterations // (dataset_size // (batch_size * n_tpus))) + 1

        key, *keys = split(key, n_batches_per_device_iteration + 1)

        indices = np.concatenate([get_indices(dataset_size, batch_size * n_tpus, index_key) for index_key in keys])

        yield rearrange(indices[: batch_size * n_tpus * device_iterations], '(t l b) -> t l b', b=batch_size, t=n_tpus, l=device_iterations)
