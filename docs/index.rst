.. image:: images/banner.svg

Shaggy
======

Shaggy is a lightweight package that implements autoencoder models in `PyTorch <https://pytorch.org>`_. It provides modular encoder-decoder architectures, the `SOAP <https://arxiv.org/abs/2409.11321>`_ and `Muon <https://kellerjordan.github.io/posts/muon/>`_ optimizers, gradient-checkpointing utilities, and more. Basically everything needed to go from raw data to a trained latent representation with minimal boilerplate.

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

Shaggy's autoencoders are the composition of an encoder and a decoder. The convolutional autoencoder works on 1D, 2D or 3D data, and can be asymmetric, with a light encoder and a heavy decoder.

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

Overview
--------

Shaggy is organized in a few sub-packages.

* :mod:`shaggy.layers` holds the building blocks of the models: a modulated RMS normalization and (convolutional) residual blocks and trunks.
* :mod:`shaggy.models` holds the autoencoders: a base :class:`~shaggy.models.AutoEncoder` class, convolutional encoders and decoders assembled into a :class:`~shaggy.models.ConvAE`, and a :class:`~shaggy.models.FusionAE` combining several trained encoder-decoder pairs.
* :mod:`shaggy.loss` holds the training losses: reconstruction, Gromov-Monge embedding (GME) and continuous ranked probability score (CRPS).
* :mod:`shaggy.optimizers` holds the SOAP, Muon and HybridMA optimizers, as well as a safe gradient step.
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
