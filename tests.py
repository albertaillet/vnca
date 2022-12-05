import equinox as eqx
from jax import numpy as np
from jax.random import PRNGKey
from jax.tree_util import tree_leaves
from models import Module, BaselineVAE, DoublingVNCA, NonDoublingVNCA, Double
from pytest import fixture

from typing import Tuple


@fixture
def latent_sizes():
    return (1, 64, 128, 256)


def output_shape(Model: Module, latent_size: int) -> Tuple[int, int, int]:
    key = PRNGKey(0)
    x = np.zeros((1, 32, 32))
    model = Model(latent_size=latent_size, key=key)
    recon_x, _, _ = model(x, key=key)
    return recon_x.shape


def test_baseline_shape(latent_sizes):
    for latent_size in latent_sizes:
        assert output_shape(BaselineVAE, latent_size) == (1, 1, 32, 32)


def test_doubling_vnca_shape(latent_sizes):
    for latent_size in latent_sizes:
        assert output_shape(DoublingVNCA, latent_size) == (1, 1, 32, 32)


def test_non_doubling_vnca_shape(latent_sizes):
    for latent_size in latent_sizes:
        assert output_shape(NonDoublingVNCA, latent_size) == (1, 1, 32, 32)


def test_doubling_vnca_num_parameters():
    key = PRNGKey(0)
    model = DoublingVNCA(latent_size=256, key=key)
    n_params = sum(x.size for x in tree_leaves(eqx.filter(model, eqx.is_array)))
    assert n_params == 6_585_088  # Number of parameters in the original model


@fixture
def img():
    return np.array(
        [
            [
                [1, 2],
                [3, 4],
            ]
        ]
    )


@fixture
def doubled_img():
    return np.array(
        [
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]
        ]
    )


def test_double_shape(img, doubled_img):
    assert np.all(Double(img) == doubled_img)
