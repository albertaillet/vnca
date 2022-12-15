import equinox as eqx
from jax import numpy as np
from jax.random import PRNGKey
from jax.tree_util import tree_leaves
from models import BaselineVAE, DoublingVNCA, NonDoublingVNCA, Double
from pytest import fixture


def test_shapes():
    '''Test that the models output the correct shapes for different batch sizes'''
    key = PRNGKey(0)
    x = np.zeros((1, 32, 32))

    for Model in (BaselineVAE, DoublingVNCA, NonDoublingVNCA):
        for latent_size in (1, 64, 128, 256):
            model = Model(latent_size=latent_size, key=key)
            for m in (1, 2):
                recon_x, mean, logvar = model(x, key=key, M=m)
                assert recon_x.shape == (m, 1, 32, 32)
                assert mean.shape == logvar.shape == (latent_size,)
                assert np.any(np.isnan(recon_x)) == False
                assert np.any(np.isnan(mean)) == False
                assert np.any(np.isnan(logvar)) == False


def test_doubling_vnca_num_parameters():
    '''Test that the DoublingVNCA model has the correct number of parameters'''
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
    '''Test that the Double module doubles the image size'''
    assert np.all(Double(img) == doubled_img)
