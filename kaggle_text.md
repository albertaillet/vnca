TODO:
The notebook uploaded to Kaggle should include:
- A summary of the report & findings (Reproducibility Summary)
- An overview of the data with any helpful charts & visualizations from the report and ideally directly using the dataset in the notebook (100GB limit)
- An overview of the methodology & experiments run, ideally with executable code examples
- An summary of the key results
- A references section & any relevant licenses or restrictions
- The notebook should also link out to any important or useful artifacts for readers to learn more, including a GitHub repository, the associated OpenReview forum, and a PDF of the Reproducibility Report
- If the dataset used is open source and less than Kaggle’s 100GB limit, it is encouraged to upload it as a Kaggle dataset so users can directly interact with the data through the notebook


# Variational Neural Cellular Automata

Welcome to this kaggle notebook

## Reproducibility Summary

The main claim of the paper being reproduced in this study is that the proposed Variational Neural Cellular Automata (VNCA) architecture, composed of a convolutional encoder and a Neural Cellular Automata (NCA)-based decoder, is able to generate high-quality samples from the binarized MNIST dataset.
The paper presents two variants of this VNCA decoder: the doubling VNCA variant that is claimed to have a simple latent space, and the non-doubling VNCA variant that is claimed to be optimized for damage recovery and stability over many steps.

To reproduce the results, we re-implemented all of the VNCA models and a fully-convolutional baseline in JAX, by using the descriptions given in the paper. We then followed the same experimental setup and hyperparameter choices as in the original paper. 

All of the models were trained on a TPU v3-8 provided by Kaggle, with a total budget of around 4 TPU hours, not counting unreported experiments.


```python
import jax.numpy as np
from pathlib import Path
from numpy import load


def get_data(root: Path, pad: int = 2) -> Tuple[Array, Array]:
    '''Get Hugo Larochelle's Binary Static MNIST dataset.'''

    train_data = load(root / 'train.npy')
    val_data = load(root / 'val.npy')
    test_data = load(root / 'test.npy')

    train_data = np.concatenate([train_data, val_data], axis=0)

    test_data = np.pad(test_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    train_data = np.pad(train_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    return train_data, test_data
```


```python
def load_data_on_tpu(devices: list, dataset: str = 'binarized_mnist', *, key: PRNGKeyArray) -> Tuple[Array, Array]:
    '''Load binarized MNIST dataset to TPU.
    The training set is replicated across all devices and the test set is sharded across all devices.
    '''
    train_dataset, test_dataset = get_data(root, pad=2)

    test_dataset = permutation(key, test_dataset, axis=0)

    shard = [*rearrange(test_dataset, '(t s) c h w -> t s c h w', t=len(devices))]

    return device_put_replicated(train_dataset, devices), device_put_sharded(shard, devices)

```

The dataset used by Palm et al. is the publicly available statically binarized version of the MNIST dataset by Larochelle et al., which contains binary images of size $28 \times 28$. However, to account for the fact that the doubling VNCA only can produce outputs of size with powers of $2$, the dataset is padded with zeros to become $32 \times 32$. Similarly, as the original MNIST dataset, the binarized MNIST dataset contains $60,000$ training samples and $10,000$ testing samples. As this version of the dataset is usually used for generative tasks, labels are not provided.

## Key results:

The convolutional baseline achieved $\log p(x) \geq -84.64$ nats evaluated with 128 importance-weighted samples on the entire test set, and cannot be compared with the original experiment since it is not presented in the paper. For the doubling VNCA, it achieved $\log p(x) \geq -84.15$ nats compared with the $\log p(x) \geq -84.23$ from the original paper. The non-doubling VNCA achieved $\log p(x) \geq -89.3$ nats compared with the $\log p(x) \geq -90.97$. These results are very similar to those from the original paper and therefore support the first claim.

Test set reconstructions and unconditional samples from the prior $\mathcal{N}(0, I)$. In figure \ref{fig:DoublingVNCA_growth} a visualization of the growing process in the NCA decoder of the doubling VNCA is presented. 


```python
Test set reconstructions here
```


An exploration of the latent space of the doubling VNCA is presented. This exploration includes linear interpolations between samples from the prior and the t-SNE reduction of $5,000$ encoded test set digits. 

```python
exploration of the latent space of the doubling VNCA
```

The damage recovery properties of the non-doubling VNCA are presented. 

```python
damage recovery properties of the non-doubling VNCA are presented. 
```

## Links:

[Github repository](https://github.com/albertaillet/vnca)

[OpenReview forum](https://openreview.net/forum?id=d7-ns6SZqp)

[PDF of the Reproducibility Report](https://openreview.net/pdf?id=d7-ns6SZqp)

## References:

[1] R. B. Palm, M. G. Duque, S. Sudhakaran, and S. Risi. Variational Neural Cellular Automata. ICLR 2022

[2] H Larochelle and I Murray. The neural autoregressive distribution estimator.