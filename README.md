<img src="assets/header.gif" width="100%">
<hr style="color:#808080;">

**Shaggy** is a lightweight package that implements autoencoder models in [PyTorch](https://pytorch.org). It provides modular encoder–decoder architectures, the [SOAP](https://arxiv.org/abs/2409.11321) optimizer, gradient-checkpointing utilities, and save/load tools. Basically everything needed to go from raw data to a trained latent representation with minimal boilerplate.

<hr style="color:#808080;">
<p align="center"><b>C O N T R I B U T O R S</b></p>
<hr style="color:#808080;">

We build on the work of [François Rozet](https://francois-rozet.github.io), [Gerome Andry](https://gerome-andry.github.io), and [Sacha Lewin](https://isach.be) as well as to the entire Science with AI Laboratory ([SAIL](https://glouppe.github.io/sail/)) team. Thanks ! 

<hr style="color:#808080;">
<p align="center"><b>T U T O R I A L</b></p>
<hr style="color:#808080;">

A self-contained tutorial is available as a Jupyter notebook. It walks through dataset loading, model configuration, training with a live loss plot, and reconstruction visualization on CIFAR-10.

➜ [`notebook/demo.ipynb`](notebook/demo.ipynb)

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
    pip install --editable '.[all]' --extra-index-url https://download.pytorch.org/whl/cu121
    ```
    Optionally, install the pre-commit hooks to automatically detect code issues before each commit:

    ```
    pre-commit install --config pre-commit.yml
    ```