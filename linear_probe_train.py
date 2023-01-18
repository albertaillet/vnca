# %%
from IPython import get_ipython

get_ipython().system('git clone https://github.com/albertaillet/vnca.git')


# %%
get_ipython().system('pip install equinox wandb einops optax')


# %%
get_ipython().run_line_magic('cd', '/kaggle/working/vnca')


# %%
import jax
from jax import jit, vmap
from jax.lax import scan
from jax import numpy as np
from jax.random import PRNGKey, split, permutation, PRNGKeyArray, randint
import equinox as eqx
from einops import repeat, rearrange
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

import data
from log_utils import restore_model
from models import NonDoublingVNCA, sample_gaussian


# typing
from jax import Array


MODEL_KEY = PRNGKey(0)
SAMPLE_KEY = PRNGKey(1)
SPLIT = 8000


# %%
# Load and restore model
model = NonDoublingVNCA(key=MODEL_KEY, latent_size=128)
model = restore_model(model, "NonDoublingVNCA_gstep100000.eqx", run_path="dladv-vnca/vnca/runs/3k9mouaj")


# %%
_, test = data.get_data()


# %%
enc = model.encoder
dec = model.decoder


@jit
def s_forward(key, x):
    key, subkey = split(key)
    m, l = enc(x)
    z = sample_gaussian(m, l, (128,), key=key)
    return subkey, dec(z)


@jit
def mass_decoder(train):
    _, a = scan(s_forward, MODEL_KEY, train)
    return a


# %%
tot = mass_decoder(test)


# %%
np.save("unprocessed_latent_space_output", tot)


# %%
tot = np.load("unprocessed_latent_space_output.npy")


# %%
@jit
def fast_rearrange(x):
    return rearrange(x, "b c h w -> b (h w) c", h=32, w=32, c=128, b=10_000)


# %%
flattened_tot = fast_rearrange(tot)
del tot


# %%
labels = np.load("/kaggle/input/test-labels/binarized_test_labels.npy")


# %%
rlabels = repeat(labels, "b -> b n", n=1024, b=10_000)


# %%
x = rearrange(flattened_tot, "b n c -> (b n) c")
y = rearrange(rlabels, "b n-> (b n) ")


# %%
np.save("x", x)
np.save("y", y)


# %%
del flattened_tot, rlabels, labels


# %%
x = np.load("x.npy")


# %%
x = permutation(MODEL_KEY, np.load("x.npy"))
y = permutation(MODEL_KEY, np.load("y.npy"))


# %%
size = int(0.8 * len(x))
train_input, test_input = x[:size], x[size:]
del x
train_label, test_label = y[:size], y[size:]
del y


# %%
probe = eqx.nn.Linear(128, 10, key=MODEL_KEY)
batch_size = 4096 * 16


def loss_fn(model, x, labels):
    return np.mean(optax.softmax_cross_entropy_with_integer_labels(vmap(model)(x), labels))


def get_indices(n: int, batch_size: int, key: PRNGKeyArray) -> Array:
    '''Get random indices for a batch.'''
    return randint(key, (batch_size,), 0, n)


def generator(dataset, labels, key):
    def dataset_iterator(batch_size: int, key: PRNGKeyArray):
        n = len(dataset)
        while True:
            key, subkey = split(key)
            indices = (get_indices(n, batch_size, subkey),)
            yield dataset[indices], labels[indices]

    return dataset_iterator(batch_size, key)


test_loss = [30]
opt = optax.adam(1e-4)
# params, static = eqx.partition(model, eqx.is_array)
opt_state = opt.init(probe)
pbar = tqdm(zip(generator(train_input, train_label, MODEL_KEY), generator(test_input, test_label, MODEL_KEY)))
for (input, labels), (i_test, l_test) in pbar:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(probe, input, labels)
    test_loss.append(loss_fn(probe, i_test, l_test))
    pbar.set_postfix({'loss': f'{loss:.3}', 'test_loss': f'{test_loss[-1]:.3}'})
    # if test_loss[-1] - test_loss[-2] > 0:
    #    break
    updates, opt_state = opt.update(grads, opt_state)
    probe = eqx.apply_updates(probe, updates)


# %%
plt.imshow(rearrange(vmap(model)(train_input[: 32 * 32]), "(h w) c-> c h w ", h=32, w=32, c=256)[0])


# %%
a = s_forward(None, test[1234])


# %%
img_size = 32 * 32
place = img_size * 9999

plt.imshow(rearrange(jax.nn.sigmoid(x[place - img_size : place]), "(h w) c-> c h w ", h=32, w=32, c=128)[0], cmap="gray")
plt.axis('off')

plt.show()

plt.imshow(rearrange(np.argmax(vmap(probe)(x[place - img_size : place]), axis=1), "(h w)-> h w ", h=32, w=32), cmap="tab10")
plt.colorbar(
    ticks=np.arange(0, 10),
)
plt.axis('off')
plt.show()
plt.show()


# %%
rearrange(np.argmax(vmap(probe)(x[: 32 * 32]), axis=1), "(h w)-> h w ", h=32, w=32)


# %%
loss_fn(probe, b, 7 * np.ones((32 * 32,), dtype=int))


# %%
eqx.tree_serialise_leaves("128-probe", probe)


# %%
