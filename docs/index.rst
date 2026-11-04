.. image:: ../assets/banner.svg

Shaggy
======

Shaggy is a lightweight package of tools to train deep learning models in `PyTorch <https://pytorch.org>`_, at scale, on many GPUs. It provides distributed training utilities, losses, the `SOAP <https://arxiv.org/abs/2409.11321>`_ and `Muon <https://kellerjordan.github.io/posts/muon/>`_ optimizers, learning rate schedulers, gradient checkpointing, tools to stop and resume a training, and models that work either in the ambient space of the data or in a latent space. Autoencoders map one space to the other; transformers, coming next, will run in either, on the data itself or on the latent states of an autoencoder. Basically everything needed to go from raw data to a trained model, at scale, with minimal boilerplate.

Installation
------------

The :mod:`shaggy` package is installable from its repository via `pip`.

.. code-block:: console

   pip install git+https://github.com/VikVador/shaggy

For a local editable install with all optional dependencies (training, notebooks, linting, documentation), create an environment

.. code-block:: console

   conda create -n shaggy python=3.11
   conda activate shaggy

then install the package from the root of the repository.

.. code-block:: console

   pip install --editable '.[all]' --index-url https://download.pytorch.org/whl/cu126

Optionally, install the pre-commit hooks to automatically detect code issues before each commit.

.. code-block:: console

   pre-commit install --config pre-commit.yml

Getting started
---------------

Shaggy's autoencoders are the composition of an encoder and a decoder, from the ambient space of the data to a latent space and back. The convolutional autoencoder works on 1D, 2D or 3D data, and can be asymmetric, with a light encoder and a heavy decoder.

.. code-block:: python

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

Trained models are saved with their configuration, and rebuilt from it.

.. code-block:: python

    from shaggy.tools import load, save

    save(autoencoder, config, "runs/cae")  # config: the arguments given to create_ConvAE

    autoencoder = load("runs/cae", create_ConvAE)

The same loop scales to many GPUs when launched with `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_. Shaggy sets up the processes, schedules the learning rate, cycles over the data and saves the state of the training, so that a job can stop and resume.

.. code-block:: python

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

Overview
--------

Shaggy is organized in a few sub-packages.

* :mod:`shaggy.layers` holds the building blocks of the models: a modulated RMS normalization and (convolutional) residual blocks, with optional gradient checkpointing.
* :mod:`shaggy.models` holds the models. Today, the autoencoders: a base :class:`~shaggy.models.AutoEncoder` class, convolutional encoders and decoders assembled into a :class:`~shaggy.models.ConvAE`, and a :class:`~shaggy.models.FusionAE` combining several trained encoder-decoder pairs.
* :mod:`shaggy.loss` holds the training losses: reconstruction, Gromov-Monge embedding (GME) and continuous ranked probability score (CRPS).
* :mod:`shaggy.optimizers` holds the SOAP, Muon and HybridMA optimizers, as well as a safe gradient step.
* :mod:`shaggy.schedulers` holds the learning rate schedulers: a linear warmup followed by a cosine decay.
* :mod:`shaggy.distributed` sets up a training on several processes launched with ``torchrun``, and averages values across them.
* :mod:`shaggy.dataloader` cycles over a dataloader for a given number of batches, and picks fixed samples of a dataset.
* :mod:`shaggy.training` saves and loads the state of a training, and tells when its loss stopped improving.
* :mod:`shaggy.tools` saves and loads models with their configuration.

For more information, check out the :doc:`tutorials <../tutorials>` or the :doc:`API <../api>`.

.. toctree::
    :caption: shaggy
    :hidden:
    :maxdepth: 2

    tutorials.rst
    api.rst

.. toctree::
    :caption: Development
    :hidden:
    :maxdepth: 1

    Source <https://github.com/VikVador/shaggy>
    Changelog <https://github.com/VikVador/shaggy/commits/master>
    License <https://github.com/VikVador/shaggy/blob/master/LICENSE>
