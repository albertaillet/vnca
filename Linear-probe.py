# %%
from log_utils import restore_model
from models import NonDoublingVNCA,sample_gaussian
from jax.random import PRNGKey, split, randint
from jax import jit
import data
import jax.numpy as np
from jax.lax import scan
import wandb
import equinox as eqx
from einops import repeat
import pickle
import matplotlib.pyplot as plt
MODEL_KEY = PRNGKey(0)


# %%
#Load and restore model
model = NonDoublingVNCA(key=MODEL_KEY)


# %%
wandb.restore("NonDoublingVNCA_gstep100000.eqx", run_path="dladv-vnca/vnca/runs/35rolimv")
model = eqx.tree_deserialise_leaves("NonDoublingVNCA_gstep100000.eqx", model)


# %%
train, test = data.get_data()


# %%
enc = model.encoder
dec = model.decoder
@eqx.filter_jit
def forward(x,key):
    m,l = enc(x)
    z = sample_gaussian(m, l, (256,), key=key)
    return dec(z)

@eqx.filter_jit
def s_forward(key,x):
    return key,forward(x,key)


_, a = scan(s_forward,MODEL_KEY,train[:10])


# %%
a = forward(model,train[0],MODEL_KEY)


# %%


