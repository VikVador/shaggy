<img src="assets/header.gif" width="100%">
<hr style="color:#808080;">

**Shaggy** is a lightweight package that implements autoencoder models in [PyTorch](https://pytorch.org). It provides modular encoder–decoder architectures, the [SOAP](https://arxiv.org/abs/2409.11321) and [Muon](https://kellerjordan.github.io/posts/muon/) optimizers, gradient-checkpointing utilities, and more. Basically everything needed to go from raw data to a trained latent representation with minimal boilerplate.

<hr style="color:#808080;">
<p align="center"><b>T U T O R I A L S</b></p>
<hr style="color:#808080;">

- [`notebook/demo-optimizers.ipynb`](notebook/demo-optimizers.ipynb) compares the available optimizers (SOAP, HybridMA with Muon) against AdamW.

- [`notebook/demo.ipynb`](notebook/demo.ipynb) trains an asymmetric 3D `ConvAE` on real ocean data, encoder then decoder, and compares its reconstruction error to the dataset mean.

<hr style="color:#808080;">
<p align="center"><b>I N S T A L L A T I O N</b></p>
<hr style="color:#808080;">

- If you want the **latest version**, install it directly from GitHub:

    ```
    pip install git+https://github.com/VikVador/shaggy
    ```

- If you want a **local editable** install with all optional dependencies (training, notebooks, linting):

    ```
    conda create -n shaggy python=3.11
    conda activate shaggy
    ```

    then

    ```
    pip install --editable '.[all]' --index-url https://download.pytorch.org/whl/cu126
    ```
    Optionally, install the pre-commit hooks to automatically detect code issues before each commit:

    ```
    pre-commit install --config pre-commit.yml
    ```
