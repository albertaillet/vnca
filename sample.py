# %%
# Imports
from data import get_data
from jax.random import PRNGKey, bernoulli
from models import AutoEncoder, BaselineVAE, DoublingVNCA, NonDoublingVNCA
from log_utils import restore_model, to_PIL_img, log_samples, log_reconstructions, log_growth_stages

SAMPLE_KEY = PRNGKey(0)


# %%
# Load the test set
_, test_data = get_data(dataset='binarized_mnist')


# %%
# Load and restore model
vnca_model = DoublingVNCA(key=SAMPLE_KEY, latent_size=256)
vnca_model = restore_model(vnca_model, 'DoublingVNCA_gstep100000.eqx', run_path='dladv-vnca/vnca/runs/14c2aulc')


# %%
# Reconstruct the test set
p = log_reconstructions(vnca_model, test_data, ih=4, iw=8, key=SAMPLE_KEY)
samples = bernoulli(SAMPLE_KEY, p=p)
to_PIL_img(samples)


# %%
# Sample from the model
p = log_samples(vnca_model, ih=4, iw=8, key=SAMPLE_KEY)
samples = bernoulli(SAMPLE_KEY, p=p)
to_PIL_img(samples)


# %%
# Sample from the model, and show the growth stages
p = log_growth_stages(vnca_model, key=SAMPLE_KEY)  # or PRNGKey(11)
to_PIL_img(p)
