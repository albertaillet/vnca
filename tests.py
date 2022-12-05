from jax import numpy as np
from jax.random import PRNGKey
from models import Module, BaselineVAE, DoublingVNCA, NonDoublingVNCA
from pytest import fixture

from typing import Tuple


@fixture
def latent_sizes():
    return (1, 64, 128, 256)


def output_shape(Model: Module, latent_size: int) -> Tuple[int, int, int]:
    key = PRNGKey(0)
    x = np.zeros((1, 28, 28))
    model = Model(latent_size=latent_size, key=key)
    recon_x, _, _ = model(x, key=key)
    return recon_x.shape


def test_baseline_shape(latent_sizes):
    for latent_size in latent_sizes:
        assert output_shape(BaselineVAE, latent_size) == (1, 1, 28, 28)


def test_doubling_vnca_shape(latent_sizes):
    for latent_size in latent_sizes:
        assert output_shape(DoublingVNCA, latent_size) == (1, latent_size, 32, 32)


def test_non_doubling_vnca_shape(latent_sizes):
    for latent_size in latent_sizes:
        assert output_shape(NonDoublingVNCA, latent_size) == (1, latent_size, 28, 28)
