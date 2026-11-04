Training an autoencoder
=======================

This tutorial trains a 3D convolutional autoencoder on the Black Sea state (temperature, salinity, oxygen and chlorophyll, each on 5 depth levels) in two stages: first the encoder alone, with a loss that preserves the geometry of the data, then the decoder alone, with a reconstruction loss.

.. code-block:: python

    import matplotlib.pyplot as plt
    import torch

    from shaggy.loss import loss_geometry_embedding, loss_reconstruction
    from shaggy.models.cae import create_ConvAE
    from shaggy.optimizers import SOAP, safe_gradient_step

Configuration
-------------

All the parameters that influence the tutorial are grouped in the dictionaries below.

.. code-block:: python

    config = {
        "data": {
            "x": "data/x.pt",
            "mask": "data/m.pt",
            "variables": ["temperature", "salinity", "oxygen", "chlorophyll"],
            "n_val": 32,
            "seed": 0,
        },
        "model": {
            "lat_channels": 8,
            "spatial": 3,
            "stride": (2, 2, 1),  # The depth axis (5 levels) is too small to be downsampled
            "config_encoder": {
                "hid_channels": [8, 16, 32],
                "hid_blocks": [2, 2, 2],
            },
            "config_decoder": {
                "hid_channels": [16, 32, 64],
                "hid_blocks": [2, 2, 2],
            },
        },
        "train": {
            "encoder": {"steps": 2_000, "lr": 8e-4, "batch_size": 16},
            "decoder": {"steps": 2_000, "lr": 8e-4, "batch_size": 16},
            "grad_clip": 2.0,
        },
        "plot": {
            "sample": 0,  # Validation sample to display
            "level": 0,   # Depth level to display (0 = surface)
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dataset
-------

The land values (NaN) are replaced by zeros, and each sample is reshaped into a 3D volume ``(variables, Y, X, depth)``, so that depth becomes a spatial axis. The land mask gets the same shape, and a few samples are held out for validation.

.. code-block:: python

    x = torch.load(config["data"]["x"])        # (N, C, Y, X)
    mask = torch.load(config["data"]["mask"])  # (C, Y, X)

    N, C, Y, X = x.shape
    V = len(config["data"]["variables"])
    Z = C // V

    # The channels (V * Z) are split into a variable axis and a trailing depth axis
    x = torch.nan_to_num(x, nan=0.0).view(N, V, Z, Y, X).movedim(2, -1).to(device)  # (N, V, Y, X, Z)
    mask = mask.view(V, Z, Y, X).movedim(1, -1).to(device)                           # (V, Y, X, Z)

    # Random split of the samples into training and validation
    generator = torch.Generator().manual_seed(config["data"]["seed"])
    permutation = torch.randperm(N, generator=generator)
    val, train = permutation[: config["data"]["n_val"]], permutation[config["data"]["n_val"] :]

    print(f"Samples | train: {len(train)} | validation: {len(val)} | shape: {tuple(x.shape[1:])}")

Model
-----

The autoencoder is asymmetric: a small encoder and a larger decoder, since compressing is assumed easier than reconstructing. The cell prints the number of parameters on each side and the size of the latent code.

.. code-block:: python

    autoencoder = create_ConvAE(in_channels=V, out_channels=V, **config["model"]).to(device)

    n_encoder = sum(p.numel() for p in autoencoder.encoder.parameters())
    n_decoder = sum(p.numel() for p in autoencoder.decoder.parameters())
    latent, factor = autoencoder.compression((V, Y, X, Z))

    print(f"Parameters | encoder: {n_encoder:,} | decoder: {n_decoder:,}")
    print(f"Latent     | shape: {latent} | compression: x{factor}")

Encoder training
----------------

The encoder is trained alone with the Gromov-Monge embedding (GME) loss, which asks the distances between latent codes to match the distances between the samples they encode. The loss curve is drawn at the end.

.. code-block:: python

    def plot_loss(losses: list, title: str) -> None:
        r"""Draws a training loss curve."""
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(range(1, len(losses) + 1), losses)
        ax.set(title=title, xlabel="Step", ylabel="Loss", yscale="log")
        plt.show()


    steps = config["train"]["encoder"]["steps"]
    batch_size = config["train"]["encoder"]["batch_size"]
    optimizer = SOAP(autoencoder.encoder.parameters(), lr=config["train"]["encoder"]["lr"])
    losses = []

    for step in range(1, steps + 1):
        batch = train[torch.randperm(len(train))[:batch_size]]

        loss = loss_geometry_embedding(x[batch], autoencoder.encode(x[batch]))
        loss.backward()
        safe_gradient_step(optimizer, grad_clip=config["train"]["grad_clip"])
        losses.append(loss.item())

    plot_loss(losses, "Encoder | GME loss")

Encoding
--------

The trained encoder is frozen and the whole dataset is encoded once. The decoder is then trained on these stored latent codes, so the encoder never has to run again.

.. code-block:: python

    autoencoder.encoder.requires_grad_(False).eval()

    with torch.no_grad():
        z = torch.cat([autoencoder.encode(x_chunk) for x_chunk in x.split(batch_size)])

    print(f"Latent codes: {tuple(z.shape)}")

Decoder training
----------------

The decoder is trained alone to rebuild each sample from its latent code, with a mean squared error where the mask removes the land points.

.. code-block:: python

    steps = config["train"]["decoder"]["steps"]
    batch_size = config["train"]["decoder"]["batch_size"]
    optimizer = SOAP(autoencoder.decoder.parameters(), lr=config["train"]["decoder"]["lr"])
    losses = []

    for step in range(1, steps + 1):
        batch = train[torch.randperm(len(train))[:batch_size]]

        loss = loss_reconstruction(autoencoder.decode(z[batch]), x[batch], weights=mask)
        loss.backward()
        safe_gradient_step(optimizer, grad_clip=config["train"]["grad_clip"])
        losses.append(loss.item())

    plot_loss(losses, "Decoder | Reconstruction loss")

Comparison
----------

On the validation samples, the reconstruction error of the autoencoder is compared to a naive baseline that always predicts the mean of the training set. The error is a root mean squared error (RMSE) over sea points, per variable (lower is better).

.. code-block:: python

    def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computes the RMSE over sea points, one value per variable (V)."""
        error = ((prediction - target).square() * mask).sum(dim=(0, 2, 3, 4))
        return (error / (mask.sum(dim=(1, 2, 3)) * len(target))).sqrt()


    autoencoder.decoder.eval()

    with torch.no_grad():
        reconstruction = torch.cat([autoencoder.decode(z_val) for z_val in z[val].split(batch_size)])

    errors = {
        "Mean": rmse(x[train].mean(dim=0, keepdim=True), x[val]).cpu(),
        "Autoencoder": rmse(reconstruction, x[val]).cpu(),
    }

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (name, error) in enumerate(errors.items()):
        ax.bar(torch.arange(V) + (i - 0.5) * 0.4, error, width=0.4, label=name)
    ax.set_xticks(range(V), config["data"]["variables"])
    ax.set(title="Validation", ylabel="RMSE")
    ax.legend()
    plt.tight_layout()
    plt.show()

.. code-block:: python

    sample, level = config["plot"]["sample"], config["plot"]["level"]

    truth = x[val][sample, ..., level].cpu()               # (V, Y, X)
    prediction = reconstruction[sample, ..., level].cpu()  # (V, Y, X)
    sea = mask[..., level].cpu().bool()                    # (V, Y, X)

    fig, axes = plt.subplots(V, 3, figsize=(11, 2.4 * V), squeeze=False, layout="constrained")

    for v, name in enumerate(config["data"]["variables"]):
        error = prediction[v] - truth[v]

        # Ground truth and reconstruction share one color scale, the error is centered on 0
        vmin, vmax = torch.nanquantile(truth[v][sea[v]], torch.tensor([0.05, 0.95]))
        emax = error[sea[v]].abs().max().item()

        panels = [
            ("Ground truth", truth[v], dict(cmap="inferno", vmin=vmin, vmax=vmax)),
            ("Reconstruction", prediction[v], dict(cmap="inferno", vmin=vmin, vmax=vmax)),
            ("Error", error, dict(cmap="RdBu_r", vmin=-emax, vmax=emax)),
        ]

        for ax, (title, field, style) in zip(axes[v], panels):
            field = torch.flipud(field.masked_fill(~sea[v], float("nan")))  # Land hidden, map upright
            image = ax.imshow(field.numpy(), **style)
            ax.set(xticks=[], yticks=[], facecolor="0.85")
            if v == 0:
                ax.set_title(title)

        axes[v, 0].set_ylabel(name)
        fig.colorbar(axes[v, 1].images[0], ax=axes[v, :2], shrink=0.9)
        fig.colorbar(image, ax=axes[v, 2], shrink=0.9)

    fig.suptitle(f"Validation sample {sample} | depth level {level}")
    plt.show()
