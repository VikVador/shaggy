![Shaggy's banner](assets/banner.svg)

[Shaggy](https://shaggy.readthedocs.io) is a lightweight Python package of tools to train deep learning models in [PyTorch](https://pytorch.org), at scale, on many GPUs. It provides distributed training utilities, losses, the [SOAP](https://arxiv.org/abs/2409.11321) and [Muon](https://kellerjordan.github.io/posts/muon/) optimizers, learning rate schedulers, gradient checkpointing, tools to stop and resume a training, and models that work either in the ambient space of the data or in a latent space. Autoencoders map one space to the other; transformers, coming next, will run in either, on the data itself or on the latent states of an autoencoder. Basically everything needed to go from raw data to a trained model, at scale, with minimal boilerplate.

> In the [Scooby-Doo](https://wikipedia.org/wiki/Scooby-Doo) cartoon, [Shaggy](https://wikipedia.org/wiki/Shaggy_Rogers) is the always scared, always hungry best friend of a Great Dane 👻
