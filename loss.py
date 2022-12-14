import jax.numpy as np
from jax.random import split
from jax.scipy.special import logsumexp
from distrax import Normal, Bernoulli
import equinox as eqx
from equinox import filter_vmap
from einops import reduce, repeat

# Typing
from jax import Array
from jax.random import PRNGKeyArray
from equinox import Module


def forward(model: Module, x: Array, key: PRNGKeyArray, M: int = 1, beta: int = 1) -> float:
    # Split the key to have one for each sample
    keys = split(key, x.shape[0])

    # Vmap over the batch, we need filter since model is a Module
    recon_x, mean, logvar = filter_vmap(model)(x, key=keys, M=M)

    return iwelbo_loss(recon_x, x, mean, logvar, M, beta=beta)


def iwelbo_loss(recon_x: Array, x: Array, mean: Array, logvar: Array, M: int = 1, beta: int = 1) -> float:
    '''Compute the IWELBO loss.'''

    # Posterior p_{\theta}(z|x)
    post = Normal(np.zeros_like(mean), np.ones_like(logvar))

    # Approximate posterior q_{\phi}(z|x)
    latent = Normal(mean, np.exp(1 / 2 * logvar))

    # Likelihood p_{\theta}(x|z)
    likelihood = Bernoulli(logits=recon_x)

    # KL divergence
    kl_div = reduce(latent.kl_divergence(post), 'b n -> b', 'sum')

    # Repeat samples for broadcasting
    kl_div = repeat(kl_div, 'b -> b m', m=M)
    xs = repeat(x, 'b c h w -> b m c h w', m=M)

    # Log-likelihood or reconstruction loss
    like = reduce(likelihood.log_prob(xs), 'b m c h w -> b m', 'sum')

    # Importance weights
    iw_loss = reduce(like - beta * kl_div, 'b m -> b', logsumexp) - np.log(M)

    # Mean over the batch
    return -np.mean(iw_loss)
