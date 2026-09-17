r"""Building blocks for Autoencoders."""

__all__ = [
    "FRMSNorm",
    "ResidualBlock",
    "ResidualGroup",
    "ResidualTrunk",
]

import torch.nn as nn

from azula.nn.layers import ConvNd, RMSNorm
from azula.nn.utils import checkpoint
from collections.abc import Sequence
from torch import Tensor
from typing import Optional


class FRMSNorm(RMSNorm):
    r"""Creates a functional RMS normalization layer.

    The features are normalized, then scaled and shifted by a modulation vector, as
    gamma(z) * x / rms(x) + beta(z), where gamma and beta are linear functions of z. With
    mod_features = 0, the layer is a plain RMS normalization and ignores z.

    References:
        | Scalable Diffusion Models with Transformers (Peebles et al., 2022)
        | https://arxiv.org/abs/2212.09748
        | Skillful joint probabilistic weather forecasting from marginals (Alet et al., 2025)
        | https://arxiv.org/abs/2506.10772

    Arguments:
        channels: Number of channels C.
        mod_features: Number of modulating features D, or 0 to disable the modulation.
        spatial: Number of spatial dimensions N.
        eps: Numerical stability term.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        spatial: int = 2,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(dim=-spatial - 1, eps=eps)

        self.spatial = spatial

        if mod_features > 0:
            self.proj = nn.Linear(mod_features, 2 * channels)

            # Identity initialization
            self.proj.weight.data.mul_(1e-2)
            self.proj.bias.data.zero_()
        else:
            self.proj = None

    def forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, L_1, ..., L_N).
            mod: Modulation vector (B, D), shared by all spatial positions.

        Returns:
            Output tensor (B, C, L_1, ..., L_N).
        """

        x = super().forward(x)

        if self.proj is None or mod is None:
            return x

        # One value per channel, broadcast over every spatial position
        gamma, beta = self.proj(mod).unflatten(-1, (2, -1)).unbind(-2)
        gamma = gamma.reshape(*gamma.shape, *(1,) * self.spatial)
        beta = beta.reshape(*beta.shape, *(1,) * self.spatial)

        return (1 + gamma) * x + beta


class ResidualBlock(nn.Module):
    r"""Creates a (convolutional) residual block module.

    Arguments:
        channels: Number of channels C.
        ffn_factor: Channel expansion factor in the feed-forward network.
        spatial: Number of spatial dimensions N.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to convolutional layers.
    """

    def __init__(
        self,
        channels: int,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.checkpointing = checkpointing
        self.norm = RMSNorm(dim=-spatial - 1)
        self.ffn = nn.Sequential(
            ConvNd(
                channels,
                channels * ffn_factor,
                spatial=spatial,
                **kwargs,
            ),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(
                channels * ffn_factor,
                channels,
                spatial=spatial,
                **kwargs,
            ),
        )

        # Identity initialization
        self.ffn[-1].weight.data.mul_(1e-2)
        self.ffn[-1].bias.data.zero_()

    def _forward(self, x: Tensor) -> Tensor:
        r"""Checkpointable forward pass."""

        return x + self.ffn(self.norm(x))

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, L_1, ..., L_N).

        Returns:
            Output tensor (B, C, L_1, ..., L_N).
        """

        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(x)
        else:
            return self._forward(x)


class ResidualGroup(nn.Module):
    r"""Creates a long-range residual blocks module.

    References:
        | Image Super-Resolution Using Very Deep Residual Channel Attention Networks (Zhang et al., 2018)
        | https://arxiv.org/abs/1807.02758

    Arguments:
        channels: Number of channels C.
        blocks: Blocks to wrap, applied in order.
        spatial: Number of spatial dimensions N.
        kwargs: Keyword arguments passed to convolutional layers.
    """

    def __init__(
        self,
        channels: int,
        blocks: Sequence[nn.Module],
        spatial: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(*blocks)

        # Fusing convolution after long-range skip
        self.fuse = ConvNd(channels, channels, spatial=spatial, **kwargs)

        # Identity initialization
        self.fuse.weight.data.mul_(1e-2)
        self.fuse.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, L_1, ..., L_N).

        Returns:
            Output tensor (B, C, L_1, ..., L_N).
        """
        return x + self.fuse(self.blocks(x))


class ResidualTrunk(nn.Module):
    r"""Creates a (convolutional) residuals within residuals block module.

    Arguments:
        channels: Number of channels C.
        num_blocks: Number of residual blocks, per group when grouped.
        num_groups: Number of residual groups, or 0 for a flat stack.
        spatial: Number of spatial dimensions N.
        ffn_factor: Channel expansion factor in the feed-forward networks.
        dropout: Dropout rate in [0, 1].
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to convolutional layers.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_groups: int,
        spatial: int = 2,
        ffn_factor: int = 1,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        block = lambda: ResidualBlock(
            channels,
            ffn_factor=ffn_factor,
            spatial=spatial,
            dropout=dropout,
            checkpointing=checkpointing,
            **kwargs,
        )

        # Simple stack of residual blocks
        if num_groups == 0:
            self.blocks = nn.Sequential(
                *[block() for _ in range(num_blocks)],
            )

        # Nested residual groups of residual blocks
        else:
            groups = [
                ResidualGroup(
                    channels,
                    [block() for _ in range(num_blocks)],
                    spatial=spatial,
                    **kwargs,
                )
                for _ in range(num_groups)
            ]

            # Adding a final skip connection to the trunk
            self.blocks = ResidualGroup(channels, groups, spatial=spatial, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: Input tensor (B, C, L_1, ..., L_N).

        Returns:
            Output tensor (B, C, L_1, ..., L_N).
        """
        return self.blocks(x)
