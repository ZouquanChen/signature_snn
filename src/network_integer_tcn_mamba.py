"""Integer TCN + continuous Mamba-2-style long-context mixer (portable backend).

Keeps the eight original integer-count nodes, then adds continuous selective
SSMs before statistics pooling. This is not a fully spiking/quantized Mamba.
"""

import math

import torch
from torch import nn

try:
    from .network_integer_tcn import IntegerTCN
    from .mamba2_torch import Mamba2ResidualBlock
except ImportError:
    from network_integer_tcn import IntegerTCN
    from mamba2_torch import Mamba2ResidualBlock


class IntegerTCNMamba2(IntegerTCN):
    def __init__(self, n_in, n_classes, mamba_layers=2, mamba_state_dim=16,
                 mamba_expand=1, mamba_head_dim=32, mamba_conv_size=4,
                 mamba_chunk_size=64, mamba_residual_scale=.1, **kwargs):
        super().__init__(n_in, n_classes, **kwargs)
        if isinstance(mamba_layers, bool) or not isinstance(mamba_layers, int) or mamba_layers < 1:
            raise ValueError("mamba_layers must be a positive integer")
        if not math.isfinite(mamba_residual_scale) or mamba_residual_scale <= 0:
            raise ValueError("mamba_residual_scale must be finite and positive")
        self.mamba_config = dict(mamba_layers=mamba_layers, mamba_state_dim=mamba_state_dim,
                                mamba_expand=mamba_expand, mamba_head_dim=mamba_head_dim,
                                mamba_conv_size=mamba_conv_size, mamba_chunk_size=mamba_chunk_size,
                                mamba_residual_scale=float(mamba_residual_scale))
        self.mamba_blocks = nn.ModuleList([
            self._make_mamba_block(width=self.width, state_dim=mamba_state_dim, expand=mamba_expand,
                                   head_dim=mamba_head_dim, conv_size=mamba_conv_size,
                                   chunk_size=mamba_chunk_size, residual_scale=mamba_residual_scale)
            for _ in range(mamba_layers)
        ])

    def _make_mamba_block(self, **options):
        return Mamba2ResidualBlock(**options)

    @property
    def local_receptive_field(self):
        return super().receptive_field

    @property
    def receptive_field(self):
        # The TCN is finite-window; the SSM can depend on the whole causal prefix.
        return None

    def get_extra_state(self):
        state = super().get_extra_state()
        state.update(version=2, model_name="integer-tcn-mamba2", mamba_backend="torch-ssd-prefix-v1",
                     **self.mamba_config)
        return state

    @classmethod
    def from_state_dict_config(cls, state_dict, n_in):
        config = dict(state_dict.get("_extra_state", {}))
        if (config.pop("version", None) != 2
                or config.pop("model_name", None) != "integer-tcn-mamba2"
                or config.pop("mamba_backend", None) != "torch-ssd-prefix-v1"):
            raise ValueError("missing or unsupported integer-tcn-mamba2 metadata")
        if config.pop("sim_steps", None) != 1 or config.pop("spike_gradient", None) != "clipped-ste":
            raise ValueError("unsupported simulation/gradient metadata")
        if config.get("n_in") != n_in:
            raise ValueError("evaluation feature dimension differs from training")
        return cls(**config)

    def forward_sequence(self, sig, mask=None, lens=None):
        hidden, valid = super().forward_sequence(sig, mask=mask, lens=lens)
        for block in self.mamba_blocks:
            if getattr(self, "checkpoint_blocks", False) and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                hidden = checkpoint(block, hidden, valid, use_reentrant=False)
            else:
                hidden = block(hidden, valid)
        return hidden, valid
