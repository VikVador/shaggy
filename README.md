<img src="assets/header.gif" style="width:100%">
<hr style="color:#808080;">
<p align="center">
    <b style="font-size:3vw; color:#808080; font-weight:bold;">
    <center>S H A G G Y</center>
    </b>
</p>
<hr style="color:#808080;">

Shaggy is a lightweight package that implements autoencoder models in [PyTorch](https://pytorch.org). It provides modular encoder–decoder architectures, the [SOAP](https://arxiv.org/abs/2409.11321) optimizer, gradient-checkpointing utilities, and save/load tools. Basically everything needed to go from raw data to a trained latent representation with minimal boilerplate.

<hr style="color:#808080;">
<p align="center">
    <b style="font-size:1.5vw; color:#808080; font-weight:bold;">
    <center>C O N T R I B U T O R S</center>
    </b>
</p>
<hr style="color:#808080;">

Shaggy builds on the work and the code of [François Rozet](https://francois-rozet.github.io), [Gerome Andry](https://gerome-andry.github.io), and [Sacha Lewin](https://isach.be) as well as to the entire [SAIL](https://glouppe.github.io/sail/) team.

<hr style="color:#808080;">
<p align="center">
    <b style="font-size:1.5vw; color:#808080; font-weight:bold;">
    <center>I N S T A L L A T I O N</center>
    </b>
</p>
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

<hr style="color:#808080;">
<p align="center">
    <b style="font-size:1.5vw; color:#808080; font-weight:bold;">
    <center>T U T O R I A L</center>
    </b>
</p>
<hr style="color:#808080;">

A self-contained tutorial is available as a Jupyter notebook. It walks through dataset loading, model configuration, training with a live loss plot, and reconstruction visualization on CIFAR-10.

➜ [`notebook/demo.ipynb`](notebook/demo.ipynb)

<hr style="color:#808080;">
<p align="center">
    <b style="font-size:1.5vw; color:#808080; font-weight:bold;">
    <center>P A P E R S</center>
    </b>
</p>
<hr style="color:#808080;">

This package is based / contains the code of the following works:

```bibtex
@article{vyas2024soap,
  title   = {SOAP: Improving and Stabilizing Shampoo using Adam},
  author  = {Vyas, Nikhil and Morwani, Depen and Zhao, Rosie and Shapira, Itai and Brandfonbrener, David and Janson, Lucas and Kakade, Sham},
  journal = {arXiv preprint arXiv:2409.11321},
  year    = {2024},
  url     = {https://arxiv.org/abs/2409.11321},
}
```

```bibtex
@article{andry2025appa,
  title={Appa: Bending Weather Dynamics with Latent Diffusion Models for Global Data Assimilation},
  author={Gérôme Andry and Sacha Lewin and François Rozet and Omer Rochman and Victor Mangeleer and Matthias Pirlet and Elise Faulx and Marilaure Grégoire and Gilles Louppe},
  booktitle={Machine Learning and the Physical Sciences Workshop (NeurIPS)},
  year={2025},
  url={https://arxiv.org/abs/2504.18720},
}
```
