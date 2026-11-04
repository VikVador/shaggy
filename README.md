![Shaggy's banner](assets/banner.svg)

# Shaggy - Distributed training in PyTorch

Shaggy is a lightweight Python package of tools to train deep learning models in [PyTorch](https://pytorch.org), at scale, on many GPUs. It provides distributed training utilities, losses, the [SOAP](https://arxiv.org/abs/2409.11321) and [Muon](https://kellerjordan.github.io/posts/muon/) optimizers, learning rate schedulers, gradient checkpointing, tools to stop and resume a training, and models that work either in the ambient space of the data or in a latent space. Autoencoders map one space to the other; transformers, coming next, will run in either, on the data itself or on the latent states of an autoencoder. Basically everything needed to go from raw data to a trained model, at scale, with minimal boilerplate.

> In the [Scooby-Doo](https://wikipedia.org/wiki/Scooby-Doo) cartoon, [Shaggy](https://wikipedia.org/wiki/Shaggy_Rogers) is the always scared, always hungry best friend of a Great Dane 👻

## Installation

The `shaggy` package is installable directly from the repository.

```
pip install git+https://github.com/VikVador/shaggy
```

## Getting started

Shaggy's autoencoders are the composition of an encoder and a decoder, from the ambient space of the data to a latent space and back. The convolutional autoencoder works on 1D, 2D or 3D data, and can be asymmetric, with a light encoder and a heavy decoder.

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

The same loop scales to many GPUs when launched with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). Shaggy sets up the processes, schedules the learning rate, cycles over the data and saves the state of the training, so that a job can stop and resume.

```python
from pathlib import Path

from shaggy.dataloader import infinite_dataloader
from shaggy.distributed import reduce_mean, setup_distributed
from shaggy.schedulers import warmup_cosine_decay
from shaggy.training import save_state

rank, local_rank, world_size, device, is_distributed = setup_distributed()

scheduler = warmup_cosine_decay(
    optimizer, lr_start=1e-5, lr_peak=1e-3, lr_end=1e-5, warmup_steps=1_000, total_steps=50_000
)

for step, x in enumerate(infinite_dataloader(train_loader, batches=50_000, is_distributed=is_distributed)):
    ...  # forward, backward and safe_gradient_step, as above
    scheduler.step()

    if step % 1_000 == 0:
        mean_loss = reduce_mean(loss.item(), device)  # averaged over every process
        save_state(Path("runs/cae/state.pt"), {"model": autoencoder.state_dict(), "step": step})
```

For more information, check out the documentation at [shaggy.readthedocs.io](https://shaggy.readthedocs.io).

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
