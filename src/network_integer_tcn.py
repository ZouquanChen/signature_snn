"""Single-step, integer-preactivation residual TCN for dynamic signatures.

The quantizer follows SpikeYOLO's round(clamp(x, 0, D)) + clipped STE.
S=1: there is NO membrane recurrence, simulated-time expansion, or loop over
signature positions. Temporal context comes from causal convolutions. The
continuous stem, normalization, residual stream and readout make this a hybrid
model, not a claim of integer arithmetic or event-driven execution on a GPU.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .integer_tcn_config import parse_dilations
    from .optimAP import LossAugmentedInferenceAP
except ImportError:
    from integer_tcn_config import parse_dilations
    from optimAP import LossAugmentedInferenceAP


class _IntegerCount(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, maximum):
        ctx.save_for_backward(value)
        ctx.maximum = maximum
        # torch.round uses ties-to-even, matching the referenced implementation.
        return value.clamp(0, maximum).round()

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        valid = (value >= 0) & (value <= ctx.maximum)
        return grad_output * valid.to(grad_output.dtype), None


class IntegerSpike(nn.Module):
    """Stateless integer counts; the backward slope is 1, not D."""

    def __init__(self, maximum=4):
        super().__init__()
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("integer_max must be a positive integer")
        self.maximum = maximum

    def forward(self, value, mask=None):
        if mask is not None:
            value = value.masked_fill(~mask[:, None, :], 0)
        return _IntegerCount.apply(value, self.maximum)


class CausalConv1d(nn.Conv1d):
    def forward(self, value):
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(value, (padding, 0)))


class IntegerResidualBlock(nn.Module):
    def __init__(self, width, kernel_size, dilation, maximum, residual_scale):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.spike1 = IntegerSpike(maximum)
        self.depthwise = CausalConv1d(
            width, width, kernel_size, dilation=dilation, groups=width, bias=False
        )
        self.norm2 = nn.LayerNorm(width)
        self.spike2 = IntegerSpike(maximum)
        self.pointwise = nn.Conv1d(width, width, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.full((1, width, 1), residual_scale))

    def forward(self, value, mask):
        # LayerNorm is per timestamp, across channels only. Its affine parameters
        # provide learned pre-quantization scale and offset.
        current = self.norm1(value.transpose(1, 2)).transpose(1, 2)
        current = self.depthwise(self.spike1(current, mask))
        current = self.norm2(current.transpose(1, 2)).transpose(1, 2)
        current = self.pointwise(self.spike2(current, mask))
        return (value + self.residual_scale * current).masked_fill(
            ~mask[:, None, :], 0
        )


class MaskedStatisticsPooling(nn.Module):
    def forward(self, value, mask):
        valid = mask[:, None, :].to(value.dtype)
        clean = value.masked_fill(~mask[:, None, :], 0)
        count = valid.sum(dim=2).clamp_min(1)
        mean = clean.sum(dim=2) / count
        centered = (clean - mean[:, :, None]).masked_fill(~mask[:, None, :], 0)
        variance = centered.square().sum(dim=2) / count
        std = (variance + 1e-5).sqrt()
        pooled = torch.cat((mean, std), dim=1)
        return pooled.masked_fill(~mask.any(dim=1, keepdim=True), 0)


class IntegerTCN(nn.Module):
    def __init__(
        self, n_in, n_classes, width=128, kernel_size=5, dilations=(1, 2, 4, 8),
        embedding_dim=256, integer_max=4, residual_scale=0.1,
        n_task=4, n_shot_g=5, n_shot_f=10, APAlpha=5.0,
    ):
        super().__init__()
        for name, value in dict(n_in=n_in, n_classes=n_classes, width=width,
                                kernel_size=kernel_size, embedding_dim=embedding_dim,
                                n_task=n_task, n_shot_g=n_shot_g, n_shot_f=n_shot_f).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        self.n_in, self.n_classes = n_in, n_classes
        self.width, self.embedding_dim = width, embedding_dim
        self.kernel_size, self.dilations = kernel_size, parse_dilations(dilations)
        self.integer_max, self.sim_steps = integer_max, 1
        self.initial_residual_scale = float(residual_scale)
        self.n_task, self.n_shot_g, self.n_shot_f = n_task, n_shot_g, n_shot_f
        self.APAlpha, self.epsilon, self.positive_update = APAlpha, 1.0, True
        self.LAI = LossAugmentedInferenceAP(n_shot_g, n_shot_f, 1.0, True)
        self.stem = CausalConv1d(n_in, width, 7, bias=False)
        self.blocks = nn.ModuleList([
            IntegerResidualBlock(width, kernel_size, dilation, integer_max, residual_scale)
            for dilation in self.dilations
        ])
        # The common exporter hooks sp2 to capture the full-resolution sequence.
        self.sp2 = MaskedStatisticsPooling()
        self.projection = nn.Linear(2 * width, embedding_dim)
        self.cls = nn.Linear(embedding_dim, n_classes, bias=False)

    @property
    def spike_neurons(self):
        # Do not register aliases: each module appears only once in state_dict.
        return tuple(neuron for block in self.blocks
                     for neuron in (block.spike1, block.spike2))

    @property
    def classifier(self):
        return self.cls

    @property
    def receptive_field(self):
        # One temporal convolution per block; the other convolution is 1x1.
        return 7 + (self.kernel_size - 1) * sum(self.dilations)

    def get_extra_state(self):
        return dict(
            version=1, model_name="integer-tcn", sim_steps=1,
            spike_gradient="clipped-ste", n_in=self.n_in, n_classes=self.n_classes,
            width=self.width, kernel_size=self.kernel_size,
            dilations=list(self.dilations), embedding_dim=self.embedding_dim,
            integer_max=self.integer_max, residual_scale=self.initial_residual_scale,
        )

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("integer-tcn checkpoint architecture/quantizer mismatch")

    @classmethod
    def from_state_dict_config(cls, state_dict, n_in):
        config = dict(state_dict.get("_extra_state", {}))
        if config.pop("version", None) != 1 or config.pop("model_name", None) != "integer-tcn":
            raise ValueError("missing or unsupported integer-tcn checkpoint metadata")
        if config.pop("sim_steps", None) != 1 or config.pop("spike_gradient", None) != "clipped-ste":
            raise ValueError("unsupported integer-tcn simulation/gradient metadata")
        if config.get("n_in") != n_in:
            raise ValueError("evaluation feature dimension differs from training")
        return cls(**config)

    @staticmethod
    def getOutputMask(lens):
        lengths = np.asarray(lens)
        if lengths.ndim != 1 or not np.isfinite(lengths).all():
            raise ValueError("lens must be a finite vector")
        if (lengths < 0).any() or (lengths != np.floor(lengths)).any():
            raise ValueError("lens must contain nonnegative integers")
        maximum = int(lengths.max()) if lengths.size else 0
        return (np.arange(maximum)[None, :] < lengths[:, None]).astype(np.float32)

    def spike_valid_lengths(self, raw_lengths, layer_idx, max_steps):
        if not 0 <= layer_idx < len(self.spike_neurons):
            raise ValueError("invalid spike layer index")
        return np.clip(np.asarray(raw_lengths, dtype=np.int64), 0, max_steps)

    def _prepare_input(self, sig, mask, lens):
        if sig.ndim != 3 or not sig.is_floating_point():
            raise ValueError("sig must be a floating [B,T,C] or [B,C,T] tensor")
        first, last = sig.shape[1] == self.n_in, sig.shape[2] == self.n_in
        if first == last:
            raise ValueError("ambiguous or invalid signature feature dimension")
        signal = sig if first else sig.transpose(1, 2)
        batch, _, steps = signal.shape
        valid = torch.ones(batch, steps, device=sig.device, dtype=torch.bool)
        if lens is not None:
            lengths = torch.as_tensor(lens, device=sig.device)
            if lengths.shape != (batch,) or not torch.isfinite(lengths).all():
                raise ValueError("lens must have shape [B] and be finite")
            if ((lengths < 0) | (lengths > steps) | (lengths != lengths.floor())).any():
                raise ValueError("lens must be integers within the padded sequence length")
            valid = torch.arange(steps, device=sig.device)[None, :] < lengths[:, None]
        if mask is not None:
            supplied = torch.as_tensor(mask, device=sig.device)
            if supplied.ndim != 2 or supplied.shape[0] != batch or supplied.shape[1] > steps:
                raise ValueError("mask must have shape [B,T] (or max(lens) with lens)")
            if not torch.isfinite(supplied).all() or ((supplied != 0) & (supplied != 1)).any():
                raise ValueError("mask must be binary and finite")
            if supplied.shape[1] != steps:
                if lens is None:
                    raise ValueError("a shorter mask requires lens")
                supplied = F.pad(supplied, (0, steps - supplied.shape[1]))
            supplied = supplied.bool()
            if lens is not None and not torch.equal(supplied, valid):
                raise ValueError("mask and lens disagree")
            valid = supplied
        # masked_fill removes even NaNs/infs in padding, unlike multiplication.
        return signal.masked_fill(~valid[:, None, :], 0), valid

    def forward_sequence(self, sig, mask=None, lens=None):
        signal, valid = self._prepare_input(sig, mask, lens)
        if signal.shape[2] == 0:
            return signal.new_zeros(signal.shape[0], self.width, 0), valid
        hidden = self.stem(signal).masked_fill(~valid[:, None, :], 0)
        for block in self.blocks:  # loop over depth only, never over time L
            if getattr(self, "checkpoint_blocks", False) and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                hidden = checkpoint(block, hidden, valid, use_reentrant=False)
            else:
                hidden = block(hidden, valid)
        return hidden, valid

    def forward(self, sig, mask=None, lens=None):
        hidden, valid = self.forward_sequence(sig, mask, lens)
        pooled = self.sp2(hidden, valid)
        embedding = F.normalize(self.projection(pooled), p=2, dim=1, eps=1e-8)
        embedding = embedding.masked_fill(~valid.any(dim=1, keepdim=True), 0)
        return embedding, self.cls(embedding)

    def tau_diversity_regularization(self):
        # Compatibility with the common AP+CE trainer: this model has no tau.
        return self.cls.weight.new_zeros(())

    def smoothCEloss(self, logit, target, eps=0.1):
        if logit.shape[1] == 1:
            return F.cross_entropy(logit, target)
        one_hot = torch.zeros_like(logit).scatter(1, target[:, None], 1)
        smoothed = one_hot * (1 - eps) + (1 - one_hot) * eps / (logit.shape[1] - 1)
        return -(smoothed * F.log_softmax(logit, dim=1)).sum(dim=1).mean()

    def APLoss_DLM(self, embedding, n_task=None):
        """Same episode layout and DLM AP objective as the existing Sig2Vec SNN."""
        tasks = self.n_task if n_task is None else n_task
        if not isinstance(tasks, int) or tasks < 1:
            raise ValueError("n_task must be a positive integer")
        step = 1 + self.n_shot_g + self.n_shot_f
        if embedding.ndim != 2 or embedding.shape[0] != tasks * step:
            raise ValueError("feature batch does not match episode layout")
        score_std, score_aug, mean_ap = (embedding.new_zeros(()) for _ in range(3))
        normalized = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)
        for start in range(0, embedding.shape[0], step):
            anchor = normalized[start]
            positive = normalized[start + 1:start + 1 + self.n_shot_g] @ anchor
            negative = normalized[start + 1 + self.n_shot_g:start + step] @ anchor
            positive = positive.sort(descending=True).values
            negative = negative.sort(descending=True).values
            difference = positive[:, None] - negative[None, :]
            score_std = score_std + (difference.sign() * difference).mean()
            direction = self.LAI(positive.detach().cpu(), negative.detach().cpu())
            mean_ap = mean_ap + self.LAI.AP(direction).to(embedding)
            score_aug = score_aug + (-direction[1:, 1:].to(embedding) * difference).mean()
        return (self.APAlpha * score_aug - score_std) / tasks, mean_ap / tasks
