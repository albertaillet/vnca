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

## Reproducibility Summary

The main claim of the paper being reproduced in this study is that the proposed Variational Neural Cellular Automata (VNCA) architecture, composed of a convolutional encoder and a Neural Cellular Automata (NCA)-based decoder, is able to generate high-quality samples from the binarized MNIST dataset.
The paper presents two variants of this VNCA decoder: the doubling VNCA variant that is claimed to have a simple latent space, and the non-doubling VNCA variant that is claimed to be optimized for damage recovery and stability over many steps.

To reproduce the results, we re-implemented all of the VNCA models and a fully-convolutional baseline in JAX, by using the descriptions given in the paper. We then followed the same experimental setup and hyperparameter choices as in the original paper. 

All of the models were trained on a TPU v3-8 provided by Kaggle, with a total budget of around 4 TPU hours, not counting unreported experiments.

The dataset used by Palm et al. [1] is the publicly available statically binarized version of the MNIST dataset by Larochelle et al. [2], which contains binary images of size $28 \times 28$. However, to account for the fact that the doubling VNCA only can produce outputs of size with powers of $2$, the dataset is padded with zeros to become $32 \times 32$. Similarly, as the original MNIST dataset, the binarized MNIST dataset contains $60,000$ training samples and $10,000$ testing samples. As this version of the dataset is usually used for generative tasks, labels are not provided.

We also load the dataset on the TPUs by replicating the training dataset on each device and sharding the sharing the test dataset over all device using `device_put_replicated` and `device_put_sharded` respectively.

```python
# Code for get_data(), get_indices(), load_data_on_tpu(), indicies_tpu_iterator()
```

The first variant of the VNCA model is the doubling VNCA variant. This model is loosely inspired by the process of cellular growth. The decoder consists of K doubling steps, each followed by a number of NCA steps, the shape of the multidimensional array between each step is shown on the arrows. The doubling operation repeats the grid in each spatial dimension, meaning that each cell is repeated four times. It is to be noted that the NCA in all the steps contain the same parameters. The NCA step is defined by a local-only communication function implemented by a 2D convolution with kernel size 3 followed by residuals and 2D convolutions with kernel size 1, meaning that they operate in papallel on each cell.

As the model is a VAE, the encoder is a convolutional neural network that outputs the mean and the log-variance of the latent distribution. The encoder is composed of 5 convolutional layers with kernel size 5 and stride 1, followed by a fully-connected layer that outputs the mean and the log-variance of the latent distribution.

```python
# double(), sample_gaussian(), crop(), Elu, Double, AutoEncoder, Residual, Conv2dZeroInit, NCAStep, DoublingVNCA.
```

The forward function is defined and the importance weighted evidence lower bound (IWELBO) is computed. The IWELBO is computed by sampling $S$ times from the latent distribution and averaging the log-likelihood of the data given the latent samples minus the KL divergence between the latent distribution and the prior. The IWELBO is then averaged over the batch.

```python
# forward(), vae_loss(), iwae_loss(), 
```

Now that the model and loss is defined, let's defined how to make an optimization step in `make_step`.


```python
# make_step(), test_iwelbo() 
```




## Key results:

The model achieves the following performance on :

### [Image Generation on Binarized MNIST](https://paperswithcode.com/sota/image-generation-on-binarized-mnist)

| Model name         | IWELBO evaluated on the test set using 128 importance weighted samples. |
| --------------- |----------- |
| BaselineVAE     | -84.64 nats  |
| DoublingVNCA    | -84.15 nats  |
| NonDoublingVNCA | -89.3 nats |

Test set reconstructions and unconditional samples from the prior $\mathcal{N}(0, I)$. 

A visualization of the growing process in the NCA decoder of the doubling VNCA is presented.

```python
growing process in the NCA decoder
```


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