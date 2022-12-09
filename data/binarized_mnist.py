import jax.numpy as np
from einops import rearrange
from pathlib import Path
from urllib.request import urlopen
from numpy import genfromtxt, save, load

# typing
from jax import Array
from typing import Tuple


# Hugo Larochelle's Binary Static MNIST
URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/'
LINKS = {'train': 'binarized_mnist_train.amat', 'val': 'binarized_mnist_valid.amat', 'test': 'binarized_mnist_test.amat'}
ROOT = Path('./data/raw/binarized_mnist')


def download_data(dir: Path) -> None:
    '''Download Hugo Larochelle's Binary Static MNIST dataset to directory.'''
    # Could also be done using https://twitter.com/alemi/status/1042658244609499137

    dir.mkdir(parents=True)
    for s, file in LINKS.items():
        amat_path = dir / file
        print(f'Downloading {s}...', end='\r')

        with urlopen(URL + file) as r:
            with open(amat_path, 'wb') as f:
                f.write(r.read())

        npz_path = dir / s
        split_data = genfromtxt(amat_path, delimiter=' ')
        split_data = rearrange(split_data, 'n (h w c) -> n c h w', c=1, h=28, w=28)
        save(npz_path, split_data)

        print(f'Downloaded {s} to {npz_path}')


def get_data(pad: int = 2) -> Tuple[Array, Array]:
    '''Get Hugo Larochelle's Binary Static MNIST dataset.'''

    if not ROOT.exists():
        download_data(ROOT)

    train_data = load(ROOT / 'train.npy')
    val_data = load(ROOT / 'val.npy')
    test_data = load(ROOT / 'test.npy')

    train_data = np.concatenate([train_data, val_data], axis=0)

    test_data = np.pad(test_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    train_data = np.pad(train_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    return train_data, test_data
