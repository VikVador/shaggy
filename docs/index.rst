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

Shaggy is organized in a few sub-packages.

* :mod:`shaggy.layers` holds the building blocks of the models: a modulated RMS normalization and (convolutional) residual blocks, groups and trunks.
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
