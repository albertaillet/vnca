import jax.numpy as np
from jax.random import split, PRNGKeyArray, permutation, PRNGKey
from einops import rearrange
from pathlib import Path
from urllib.request import urlopen
from numpy import genfromtxt, save, load

# typing
from jax import Array
from typing import Iterator, Tuple, List, Dict

ROOT = Path('./data/raw/binarized_mnist')
URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/'
LINKS = {'train': 'binarized_mnist_train.amat', 'val': 'binarized_mnist_valid.amat', 'test': 'binarized_mnist_test.amat'}


def download_mnist(dir: Path) -> None:
    '''Download binarized MNIST dataset to directory.'''

    dir.mkdir(parents=True)
    for s, file in LINKS.items():
        amat_path = dir / file
        print(f'Downloading {s}...', end='\r')

        with urlopen(URL + file) as r:
            with open(amat_path, 'wb') as f:
                f.write(r.read())

        npz_path = dir / s
        split_data = genfromtxt(amat_path, delimiter=' ')
        split_data = rearrange(split_data, 'n (h w c) -> n c h w', h=28, w=28, c=1)
        save(npz_path, split_data)

        print(f'Downloaded {s} to {npz_path}')


def get_mnist() -> Tuple[Array, Array]:
    '''Get binarized MNIST dataset.'''

    if not ROOT.exists():
        download_mnist(ROOT)

    train_data = load(ROOT / 'train.npy')
    val_data = load(ROOT / 'val.npy')
    test_data = load(ROOT / 'test.npy')

    train_data = np.concatenate([train_data, val_data], axis=0)
    return train_data, test_data


def get_indices(n: int, batch_size: int, key: PRNGKeyArray) -> Array:
    '''Get random indices for a batch.'''
    indices = np.arange(n)  # [0, 1, 2, ..., len(dataset)]
    indices = permutation(key, indices)  # shuffle the indices
    indices = indices[: (n // batch_size) * batch_size]  # drop the last few samples not in a batch
    return indices  # reshape into (n_batches, batch_size)


def load_mnist(batch_size: int, key: PRNGKeyArray) -> Tuple[Iterator, Iterator]:
    '''Load binarized MNIST dataset.'''
    train_dataset, test_dataset = get_mnist()

    def dataset_iterator(dataset: Array, batch_size: int, key: PRNGKeyArray) -> Iterator:
        n = len(dataset)
        while True:
            key, subkey = split(key)
            for batch_indices in get_indices(n, batch_size, subkey):
                yield dataset[batch_indices]

    return dataset_iterator(train_dataset, batch_size, key), dataset_iterator(test_dataset, batch_size, key)


def load_mnist_train_on_tpu(devices: list) -> Array:
    '''Load binarized MNIST dataset to TPU.'''
    from jax import device_put_replicated

    train_dataset, _ = get_mnist()

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
