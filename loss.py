import jax.numpy as np
from jax.random import split
from jax.scipy.special import logsumexp
from distrax import Normal, Bernoulli
import equinox as eqx
from equinox import filter_vmap
from einops import reduce

# Typing
from jax import Array
from jax.random import PRNGKeyArray
from equinox import Module


@eqx.filter_value_and_grad
def iwelbo_loss(model: Module, x: Array, key: PRNGKeyArray, M: int = 1) -> float:

    # Split the key to have one for each sample
    keys = split(key, x.shape[0])

    # Vmap over the batch, we need filter since model is a Module
    recon_x, mean, logvar = filter_vmap(model)(x, key=keys, M=M)

    # Posterior p_{\theta}(z|x)
    post = Normal(loc=np.zeros_like(mean), scale=np.ones_like(logvar))

    # Approximate posterior q_{\phi}(z|x)
    latent = Normal(mean, np.exp(1 / 2 * logvar))

    # Likelihood p_{\theta}(x|z)
    likelihood = Bernoulli(logits=recon_x)

    # KL divergence
    kl_div = reduce(latent.kl_divergence(post), 'b n -> b', 'sum')

    # Log-likelihood or reconstruction loss
    like = reduce(likelihood.log_prob(x), 'b m c h w -> b m', 'sum')

    # Importance weights
    iw_loss = -np.mean(reduce(like - kl_div, "b m -> b", logsumexp) - np.log(M))
    return iw_loss
