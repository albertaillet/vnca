# Re-implementation of Variational Neural Cellular Automata

The repository contains code for the reproduction of the results from "Variational Neural Cellular Automata" [1] for the course [DD2412 Deep Learning, Advanced. KTH (Royal Institute of Technology), Stockholm, Sweden](https://www.kth.se/student/kurser/kurs/DD2412?l=en).

The autograd engine [JAX](https://github.com/google/jax), the neural network library [equinox](https://github.com/patrick-kidger/equinox), the optimization library [optax](https://github.com/deepmind/optax) and the tensor operation library [einops](https://github.com/arogozhnikov/einops) are used.

The results using the binarized MNIST dataset [2] are the main points of the paper reproduced.

<img src="./images/damage_full.png" alt="drawing" width="400"/>

## Requirements

To install requirements locally, run the following command:

```setup
pip install -r requirements.txt
```

## Training

To train a model using 8 v3 TPUs available on [Kaggle](https://www.kaggle.com/), import the script `main-train.py` as a kaggle notebook under:

+ Create -> New Notebook -> File -> Import Notebook

<img src="./images/kaggle_import.png" alt="drawing" width="400"/>

Then select the TPU accelerator:

<img src="./images/kaggle_accelerator.png" alt="drawing" width="400"/>

The script can then be run as a notebook.

## Evaluation

To evaluate a trained model, the script to be used is `eval.py`. The script should be loaded onto Kaggle in the same way as the training script.


## Results

Our model achieves the following performance on :

### [Image Generation on Binarized MNIST](https://paperswithcode.com/sota/image-generation-on-binarized-mnist)

| Model name         | ELBO evaluated on the test set using 128 importance weighted samples. |
| --------------- |----------- |
| BaselineVAE     | -83.88 nats  |
| DoublingVNCA    | -83.56 nats  |
| NonDoublingVNCA | -87.92 nats |

<!-- 📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.--> 

## Figures

The different figures from the paper can be reproduced using the following scripts:

| Figure         | Script to reproduce the figure |
| --------------- |----------- |
| Figure 2     | `sample.py`  |
| Figure 3    | `sample.py`  |
| Figure 4 | `latent_interpolate.py` and `t-sne.py` |
| Figure 5 | `damage_recovery.py` |
| Figure 6 | `linear_probe_figure.py` |
| Figure 7 | `latent_viz.py` |

## Tests

To run a few tests that test the model output shape and the doubling operation, run the following command:

```pytonn
pytest tests.py
```


## References


[1] R. B. Palm, M. G. Duque, S. Sudhakaran, and S. Risi. Variational Neural Cellular Automata. ICLR 2022

[2] H Larochelle and I Murray. The neural autoregressive distribution estimator.
