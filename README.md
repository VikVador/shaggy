![Shaggy's banner](assets/banner.svg)

# Shaggy - Autoencoders in PyTorch

Shaggy is a lightweight Python package that implements autoencoder models in [PyTorch](https://pytorch.org). It provides modular encoder-decoder architectures, losses tailored to latent representations, the [SOAP](https://arxiv.org/abs/2409.11321) and [Muon](https://kellerjordan.github.io/posts/muon/) optimizers, gradient-checkpointing utilities, and more. Basically everything needed to go from raw data to a trained latent representation, at scale, with minimal boilerplate.

> In the [Scooby-Doo](https://wikipedia.org/wiki/Scooby-Doo) cartoon, [Shaggy](https://wikipedia.org/wiki/Shaggy_Rogers) is the always scared, always hungry best friend of a Great Dane 👻

## Installation

The `shaggy` package is installable directly from the repository.

```
pip install git+https://github.com/VikVador/shaggy
```

## Getting started

Shaggy's autoencoders are the composition of an encoder and a decoder. The convolutional autoencoder works on 1D, 2D or 3D data, and can be asymmetric, with a light encoder and a heavy decoder.

```python
from shaggy.loss import loss_reconstruction
from shaggy.models import create_ConvAE
from shaggy.optimizers import SOAP, safe_gradient_step

# Build a 2D convolutional autoencoder
autoencoder = create_ConvAE(
    in_channels=3,
    out_channels=3,
    lat_channels=8,
    hid_channels=[64, 128, 256],
    hid_blocks=[2, 2, 2],
    spatial=2,
)

latent, factor = autoencoder.compression((3, 64, 64))  # (8, 16, 16), 6

# Train it to reconstruct its input
optimizer = SOAP(autoencoder.parameters(), lr=1e-3)

for x in train_loader:
    z, y = autoencoder(x)

    loss = loss_reconstruction(y, x)
    loss.backward()

    safe_gradient_step(optimizer, grad_clip=1.0)
```

Trained models are saved with their configuration, and rebuilt from it.

```python
from shaggy.tools import load, save

save(autoencoder, config, "runs/cae")  # config: the arguments given to create_ConvAE

autoencoder = load("runs/cae", create_ConvAE)
```

## Tutorials

Tutorials live on the [`master-demo`](https://github.com/VikVador/shaggy/tree/master-demo) branch, a copy of `master` that also holds the datasets they use.

* [`demo-optimizers.ipynb`](https://github.com/VikVador/shaggy/blob/master-demo/notebook/demo-optimizers.ipynb) compares the available optimizers (SOAP, HybridMA with Muon) against AdamW.
* [`demo.ipynb`](https://github.com/VikVador/shaggy/blob/master-demo/notebook/demo.ipynb) trains an asymmetric 3D `ConvAE` on real ocean data and compares its reconstruction error to the dataset mean.

## Contributing

For a local editable install with all optional dependencies (training, notebooks, linting):

```
conda create -n shaggy python=3.11
conda activate shaggy
pip install --editable '.[all]' --index-url https://download.pytorch.org/whl/cu126
```

Then install the pre-commit hooks, which detect code issues before each commit, and run the tests.

```
pre-commit install --config pre-commit.yml
pytest tests
```
