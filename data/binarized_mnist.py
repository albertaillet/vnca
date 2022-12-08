import jax.numpy as np
from einops import rearrange
from pathlib import Path
from urllib.request import urlopen
from numpy import genfromtxt, save, load

# typing
from jax import Array
from typing import Tuple

ROOT = Path('./data/raw/binarized_mnist')
URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/'
LINKS = {'train': 'binarized_mnist_train.amat', 'val': 'binarized_mnist_valid.amat', 'test': 'binarized_mnist_test.amat'}


def download_data(dir: Path) -> None:
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


def get_data(pad: int = 2) -> Tuple[Array, Array]:
    '''Get binarized MNIST dataset.'''

    if not ROOT.exists():
        download_data(ROOT)

    train_data = load(ROOT / 'train.npy')
    val_data = load(ROOT / 'val.npy')
    test_data = load(ROOT / 'test.npy')

    train_data = np.concatenate([train_data, val_data], axis=0)

    test_data = np.pad(test_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    train_data = np.pad(train_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    return train_data, test_data
