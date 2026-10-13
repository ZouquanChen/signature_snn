#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, functional
from torch.autograd import Function

try:
    from .optimAP import LossAugmentedInferenceAP
except ImportError:
    from optimAP import LossAugmentedInferenceAP


class SurrogateSpike(Function):
    """Hard threshold with a selectable backward-only gradient estimator.

    ``surrogate`` preserves the historical triangular surrogate. ``ste`` keeps
    the exact same binary forward pass but sends the upstream gradient through
    unchanged (the standard identity straight-through estimator).
    """

    sigma = 1.0
    gradient_mode = "surrogate"

    @staticmethod
    def forward(ctx, membrane):
        mode = str(SurrogateSpike.gradient_mode).lower()
        sigma = float(SurrogateSpike.sigma)
        if mode not in ("surrogate", "ste"):
            raise ValueError("unsupported spike gradient mode: %s" % mode)
        if sigma <= 0.0:
            raise ValueError("surrogate sigma must be positive")

        # Capture the configuration used by this forward call so changing the
        # process-wide mode cannot alter an already constructed autograd graph.
        ctx.gradient_mode = mode
        ctx.sigma = sigma
        if mode == "surrogate":
            ctx.save_for_backward(membrane)
        return (membrane > 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.gradient_mode == "ste":
            return grad_output
        (membrane,) = ctx.saved_tensors
        scale = torch.clamp(1.0 - (membrane.abs() / ctx.sigma), min=0.0)
        return grad_output * scale


def configure_spike_gradient(mode="surrogate", sigma=1.0):
    """Configure the backward estimator shared by all project spike neurons."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in ("surrogate", "ste"):
        raise ValueError(
            "spike gradient mode must be 'surrogate' or 'ste', got %r" % mode
        )
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("surrogate sigma must be positive")
    SurrogateSpike.gradient_mode = normalized_mode
    SurrogateSpike.sigma = sigma


def delta_encode(signal, threshold):
    if signal.dim() != 3:
        raise ValueError("signal must have shape (batch, channels, time)")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    delta = signal.new_zeros(signal.shape)
    delta[:, :, 1:] = signal[:, :, 1:] - signal[:, :, :-1]
    pos = (delta >= threshold).to(signal.dtype)
    neg = (delta <= -threshold).to(signal.dtype)
    return pos - neg


class DeltaEventEncoder(nn.Module):
    def __init__(self, event_steps, threshold, normalize_delta=True):
        super(DeltaEventEncoder, self).__init__()
        if event_steps <= 0:
            raise ValueError("event_steps must be positive")
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        self.event_steps = int(event_steps)
        self.threshold = float(threshold)
        self.normalize_delta = bool(normalize_delta)

    def forward(self, signal, mask=None):
        if signal.dim() != 3:
            raise ValueError("signal must have shape (batch, channels, time)")

        valid_mask = None
        if mask is not None:
            if mask.dim() == 3 and mask.size(1) == 1:
                mask = mask.squeeze(1)
            if mask.dim() != 2:
                raise ValueError("mask must have shape (batch, time)")
            if mask.size(0) != signal.size(0) or mask.size(1) != signal.size(2):
                raise ValueError("mask shape must match signal batch and time")
            valid_mask = mask.to(dtype=signal.dtype, device=signal.device)

        delta = signal.new_zeros(signal.shape)
        step_delta = signal[:, :, 1:] - signal[:, :, :-1]
        if valid_mask is not None:
            contiguous_valid = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(1)
            step_delta = step_delta * contiguous_valid

        if self.normalize_delta:
            scale = step_delta.detach().abs().mean(dim=2, keepdim=True).clamp_min(1e-6)
            step_delta = step_delta / scale

        delta[:, :, 1:] = step_delta

        eps = torch.finfo(signal.dtype).eps if signal.dtype.is_floating_point else 0.0
        on_counts = torch.floor(torch.clamp(delta, min=0.0) / self.threshold + eps)
        off_counts = torch.floor(torch.clamp(-delta, min=0.0) / self.threshold + eps)
        on_counts = on_counts.clamp(max=self.event_steps).to(dtype=torch.int64)
        off_counts = off_counts.clamp(max=self.event_steps).to(dtype=torch.int64)

        step_ids = torch.arange(self.event_steps, device=signal.device).view(
            -1, 1, 1, 1
        )
        on_events = (step_ids < on_counts.unsqueeze(0)).to(dtype=signal.dtype)
        off_events = (step_ids < off_counts.unsqueeze(0)).to(dtype=signal.dtype)
        events = torch.cat([on_events, off_events], dim=2)

        if valid_mask is not None:
            events = events * valid_mask.view(1, signal.size(0), 1, signal.size(2))

        return events


class PLIFLTNode(base.MemoryModule):
    def __init__(self, tau=2.0, v_threshold=0.25):
        super(PLIFLTNode, self).__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.tau_raw = nn.Parameter(self._inverse_softplus(tau - 1.0))
        self.threshold_raw = nn.Parameter(self._inverse_softplus(v_threshold))
        self.register_memory("v", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        tau = 1.0 + F.softplus(self.tau_raw)
        threshold = F.softplus(self.threshold_raw)
        return tau, threshold

    def single_step_forward(self, x):
        tau, threshold = self.current_parameters()

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        self.v = self.v + (x - self.v) / tau
        spike = SurrogateSpike.apply(self.v - threshold)
        self.v = self.v - spike * threshold
        return spike


class PLIFTauNode(base.MemoryModule):
    def __init__(self, tau=2.0, v_threshold=0.25):
        super(PLIFTauNode, self).__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.tau_raw = nn.Parameter(self._inverse_softplus(tau - 1.0))
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        tau = 1.0 + F.softplus(self.tau_raw)
        threshold = tau.new_tensor(self.v_threshold)
        return tau, threshold

    def effective_taus(self):
        tau, _ = self.current_parameters()
        return tau.view(1)

    def single_step_forward(self, x):
        tau, threshold = self.current_parameters()

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        self.v = self.v + (x - self.v) / tau
        spike = SurrogateSpike.apply(self.v - threshold)
        self.v = self.v - spike * threshold
        return spike


class ScalarPLIFComparisonNode(PLIFTauNode):
    """PLIF baseline with one learnable time constant shared by a layer."""


class ResidualCausalTemporalMix(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        alpha_init=0.01,
        ema_decay=0.6,
        padding_mode="zero",
    ):
        super(ResidualCausalTemporalMix, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be in (0, 1)")
        if not 0.0 < ema_decay <= 1.0:
            raise ValueError("ema_decay must be in (0, 1]")
        if padding_mode not in ("zero", "replicate"):
            raise ValueError("padding_mode must be 'zero' or 'replicate'")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding_mode = padding_mode
        self.conv = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            groups=self.channels,
            bias=False,
        )
        self.gate = nn.Parameter(self._logit(alpha_init))
        self._init_ema_kernel(ema_decay)

    @staticmethod
    def _logit(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(value_tensor / (1.0 - value_tensor))

    def _init_ema_kernel(self, decay):
        with torch.no_grad():
            exponents = torch.arange(self.kernel_size - 1, -1, -1, dtype=torch.float32)
            weight = torch.pow(
                torch.tensor(float(decay), dtype=torch.float32), exponents
            )
            weight = weight / weight.sum()
            self.conv.weight.copy_(
                weight.view(1, 1, self.kernel_size).repeat(self.channels, 1, 1)
            )

    def _causal_pad(self, x):
        pad = self.kernel_size - 1
        if self.padding_mode == "zero":
            return F.pad(x, (pad, 0))
        left = x[:, :, :1].repeat(1, 1, pad)
        return torch.cat([left, x], dim=2)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        mixed = self.conv(self._causal_pad(x))
        alpha = torch.sigmoid(self.gate).to(device=x.device, dtype=x.dtype)
        return x + alpha * (mixed - x)


class PLIFTauChannelwiseNode(base.MemoryModule):
    def __init__(self, channels, tau=2.0, v_threshold=0.25):
        super(PLIFTauChannelwiseNode, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.channels = int(channels)
        self.tau_raw = nn.Parameter(
            self._inverse_softplus(tau - 1.0).repeat(self.channels)
        )
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        taus = 1.0 + F.softplus(self.tau_raw)
        threshold = taus.new_tensor(self.v_threshold)
        return taus, threshold

    def effective_taus(self):
        taus, _ = self.current_parameters()
        return taus

    def single_step_forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )

        taus, threshold = self.current_parameters()
        tau_per_channel = taus.to(device=x.device, dtype=x.dtype).view(
            1, self.channels, 1
        )

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        self.v = self.v + (x - self.v) / tau_per_channel
        spike = SurrogateSpike.apply(self.v - threshold)
        self.v = self.v - spike * threshold
        return spike

    def _single_step_burst(self, x, K=2, normalize=True, thresholds=None):
        taus, threshold = self.current_parameters()
        tau_per_channel = taus.to(device=x.device, dtype=x.dtype).view(
            1, self.channels, 1
        )
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)
        self.v = self.v + (x - self.v) / tau_per_channel
        spike_count = torch.zeros_like(self.v)
        for k in range(1, K + 1):
            th = k * threshold
            if thresholds is not None and k - 1 < len(thresholds):
                th = float(thresholds[k - 1]) * threshold
            spike_count = spike_count + SurrogateSpike.apply(self.v - th)
        self.v = self.v - spike_count * threshold
        return spike_count / float(K) if normalize else spike_count

    def _single_step_signed_burst(self, x, K=2, normalize=True, thresholds=None):
        taus, threshold = self.current_parameters()
        tau_per_channel = taus.to(device=x.device, dtype=x.dtype).view(
            1, self.channels, 1
        )
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)
        self.v = self.v + (x - self.v) / tau_per_channel
        pos_count = torch.zeros_like(self.v)
        neg_count = torch.zeros_like(self.v)
        for k in range(1, K + 1):
            th = k * threshold
            if thresholds is not None and k - 1 < len(thresholds):
                th = float(thresholds[k - 1]) * threshold
            pos_count = pos_count + SurrogateSpike.apply(self.v - th)
            neg_count = neg_count + SurrogateSpike.apply(-self.v - th)
        signed_count = pos_count - neg_count
        self.v = self.v - pos_count * threshold + neg_count * threshold
        return signed_count / float(K) if normalize else signed_count

    def _single_step_burst_slots(self, x, K=2):
        taus, threshold = self.current_parameters()
        tau_per_channel = taus.to(device=x.device, dtype=x.dtype).view(
            1, self.channels, 1
        )
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)
        self.v = self.v + (x - self.v) / tau_per_channel
        slots = []
        buf = self.v.clone()
        for k in range(1, K + 1):
            sk = SurrogateSpike.apply(buf - k * threshold)
            slots.append(sk)
            buf = buf - sk * threshold
        total = torch.stack(slots).sum(dim=0)
        self.v = self.v - total * threshold
        return (
            torch.stack(slots).permute(1, 2, 0, 3).reshape(x.size(0), self.channels, -1)
        )


class BurstPLIFTauChannelwiseNode(PLIFTauChannelwiseNode):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        burst_k=2,
        normalize=True,
        thresholds=None,
    ):
        super().__init__(channels=channels, tau=tau, v_threshold=v_threshold)
        self.burst_k = int(burst_k)
        self.normalize = bool(normalize)
        self.thresholds = thresholds

    def single_step_forward(self, x):
        return self._single_step_burst(
            x, K=self.burst_k, normalize=self.normalize, thresholds=self.thresholds
        )


class SignedBurstPLIFTauChannelwiseNode(BurstPLIFTauChannelwiseNode):
    def single_step_forward(self, x):
        return self._single_step_signed_burst(
            x, K=self.burst_k, normalize=self.normalize, thresholds=self.thresholds
        )


class BinaryBurstSlotsPLIFTauChannelwiseNode(PLIFTauChannelwiseNode):
    def __init__(self, channels, tau=2.0, v_threshold=0.25, burst_k=2):
        super().__init__(channels=channels, tau=tau, v_threshold=v_threshold)
        self.burst_k = int(burst_k)

    def single_step_forward(self, x):
        return self._single_step_burst_slots(x, K=self.burst_k)


class _TemporalComparisonNode(base.MemoryModule):
    """Shape-preserving base for optional neuron-only comparison baselines."""

    def _validate(self, x):
        if x.dim() != 3:
            raise ValueError("comparison neuron expects (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError("input channel dimension does not match configured channels")

    def forward(self, x):
        self._validate(x)
        if x.size(2) == 0:
            return x.new_empty(x.shape)
        return torch.stack(
            [self.single_step_forward(x[:, :, t]) for t in range(x.size(2))], dim=2
        )


class MLFK3ComparisonNode(_TemporalComparisonNode):
    """Three independent membrane states with fixed multi-level thresholds."""

    def __init__(self, channels, decay=0.25, thresholds=(0.6, 1.6, 2.6)):
        super().__init__()
        self.channels = int(channels)
        self.decay = float(decay)
        self.thresholds = tuple(float(v) for v in thresholds)
        if len(self.thresholds) != 3:
            raise ValueError("MLF-K3 requires exactly three thresholds")
        self.register_buffer("tau_values", torch.full((self.channels,), 1.0 / self.decay))
        self.register_memory("v", None)
        self.register_memory("prev_spike", None)

    def effective_taus(self):
        return self.tau_values

    def single_step_forward(self, x):
        if self.v is None or self.v.shape != (3,) + x.shape:
            self.v = torch.zeros((3,) + x.shape, device=x.device, dtype=x.dtype)
            self.prev_spike = torch.zeros_like(self.v)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)
            self.prev_spike = self.prev_spike.to(device=x.device, dtype=x.dtype)
        self.v = self.decay * self.v * (1.0 - self.prev_spike) + x.unsqueeze(0)
        thresholds = x.new_tensor(self.thresholds).view(3, 1, 1)
        spikes = SurrogateSpike.apply(self.v - thresholds)
        self.prev_spike = spikes
        return spikes.sum(dim=0)


class ILIFComparisonNode(_TemporalComparisonNode):
    """Temporal inhibitory LIF baseline adapted to the 1-D Sig2Vec backbone."""

    def __init__(self, channels, tau=2.0, v_threshold=0.25, inhibition_scale=0.03):
        super().__init__()
        self.channels = int(channels)
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.inhibition_scale = float(inhibition_scale)
        self.register_buffer("tau_values", torch.full((self.channels,), self.tau))
        for name in ("v", "prev_input", "prev_spike", "inhibition", "inhibitory_memory"):
            self.register_memory(name, None)

    def effective_taus(self):
        return self.tau_values

    def single_step_forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
            self.prev_input = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)
            self.inhibition = torch.zeros_like(x)
            self.inhibitory_memory = torch.zeros_like(x)
        self.inhibition = self.inhibition_scale * (
            self.inhibition + self.prev_input * self.prev_spike
        )
        current = x - torch.clamp(self.inhibition, min=0.0)
        self.v = self.v * (1.0 - 1.0 / self.tau) + current
        spike = SurrogateSpike.apply(self.v - self.v_threshold)
        self.prev_input = x
        self.prev_spike = spike
        self.v = self.v - spike * self.v_threshold
        self.inhibitory_memory = self.inhibitory_memory + spike * self.v
        self.v = self.v - spike * torch.sigmoid(self.inhibitory_memory)
        return spike


class GLIFComparisonNode(_TemporalComparisonNode):
    """Channel-wise gated LIF without the source model's tdBatchNorm."""

    def __init__(self, channels, tau=2.0, v_threshold=0.25):
        super().__init__()
        self.channels = int(channels)
        if tau <= 1.0:
            raise ValueError("GLIF tau must be greater than 1")
        logit = lambda value: math.log(value / (1.0 - value))
        tau_init = math.log(math.expm1(tau - 1.0))
        self.tau_raw = nn.Parameter(torch.full((self.channels,), tau_init))
        self.threshold_raw = nn.Parameter(torch.full((self.channels,), logit(v_threshold)))
        self.leak_raw = nn.Parameter(torch.full((self.channels,), logit(v_threshold / 8.0)))
        self.conduct_raw = nn.Parameter(torch.zeros(self.channels))
        self.alpha_raw = nn.Parameter(torch.full((self.channels,), logit(0.6)))
        self.beta_raw = nn.Parameter(torch.full((self.channels,), logit(0.8)))
        self.gamma_raw = nn.Parameter(torch.full((self.channels,), logit(0.6)))
        self.register_memory("v", None)
        self.register_memory("prev_spike", None)

    def effective_taus(self):
        return 1.0 + F.softplus(self.tau_raw)

    def single_step_forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)
        tau = self.effective_taus().to(x).view(1, -1)
        threshold = torch.sigmoid(self.threshold_raw).to(x).view(1, -1)
        leak = torch.sigmoid(self.leak_raw).to(x).view(1, -1)
        conduct = torch.sigmoid(self.conduct_raw).to(x).view(1, -1)
        alpha = torch.sigmoid(self.alpha_raw).to(x).view(1, -1)
        beta = torch.sigmoid(self.beta_raw).to(x).view(1, -1)
        gamma = torch.sigmoid(self.gamma_raw).to(x).view(1, -1)
        decay = 1.0 - alpha / tau
        current = x * (1.0 - beta * (1.0 - conduct))
        self.v = decay * self.v * (1.0 - gamma * self.prev_spike) - (1.0 - alpha) * leak + current
        spike = SurrogateSpike.apply(self.v - threshold)
        self.v = self.v - (1.0 - gamma) * threshold * spike
        self.prev_spike = spike
        return spike


class KLIFComparisonNode(_TemporalComparisonNode):
    """K-based LIF with learnable membrane scaling and hard reset."""

    def __init__(self, channels, tau=2.0, v_threshold=0.25):
        super().__init__()
        if tau <= 1.0:
            raise ValueError("KLIF tau must be greater than 1")
        self.channels = int(channels)
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.k = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("tau_values", torch.full((self.channels,), self.tau))
        self.register_memory("v", None)

    def effective_taus(self):
        return self.tau_values

    def single_step_forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)
        self.v = self.v + (x - self.v) / self.tau
        self.v = F.relu(self.k.to(x) * self.v)
        spike = SurrogateSpike.apply(self.v - self.v_threshold)
        self.v = self.v * (1.0 - spike)
        return spike


class SlidingPSNComparisonNode(_TemporalComparisonNode):
    """Causal SlidingPSN with a learnable finite-order temporal kernel."""

    def __init__(self, channels, order=2, v_threshold=1.0):
        super().__init__()
        self.channels = int(channels)
        self.order = int(order)
        if self.order <= 0:
            raise ValueError("PSN order must be positive")
        weights = torch.ones(self.order)
        for i in range(self.order - 2, -1, -1):
            weights[i] = weights[i + 1] / 2.0
        self.weight = nn.Parameter(weights)
        self.threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.register_buffer("tau_values", torch.ones(self.channels))

    def effective_taus(self):
        return self.tau_values

    def single_step_forward(self, x):
        raise RuntimeError("SlidingPSN requires the complete time sequence")

    def forward(self, x):
        self._validate(x)
        if x.size(2) == 0:
            return x.new_empty(x.shape)
        flat = x.reshape(-1, 1, x.size(2))
        padded = F.pad(flat, (self.order - 1, 0))
        kernel = self.weight.to(x).view(1, 1, self.order)
        membrane = F.conv1d(padded, kernel).reshape_as(x)
        return SurrogateSpike.apply(membrane - self.threshold.to(x))


class SDNComparisonNode(_TemporalComparisonNode):
    """Frozen learned membrane-dynamics surrogate from the SDN baseline."""

    def __init__(self, channels, checkpoint):
        super().__init__()
        if not checkpoint:
            raise ValueError("SDN comparison requires an explicit checkpoint path")
        self.channels = int(channels)
        self.checkpoint = str(checkpoint)
        self.model = torch.jit.load(self.checkpoint, map_location="cpu").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("tau_values", torch.ones(self.channels))

    def effective_taus(self):
        return self.tau_values

    def single_step_forward(self, x):
        raise RuntimeError("SDN comparison requires the complete time sequence")

    def forward(self, x):
        self._validate(x)
        if x.size(2) == 0:
            return x.new_empty(x.shape)
        shape = x.shape
        self.model.eval()
        with torch.no_grad():
            membrane = self.model(x.detach().reshape(-1, 1, shape[-1]))
        membrane = membrane.reshape(shape).to(device=x.device, dtype=x.dtype)
        return SurrogateSpike.apply(membrane + x - 1.0)


class PLIFTauChannelwiseDynamicThresholdNode(base.MemoryModule):
    def __init__(self, channels, tau=2.0, v_threshold=0.25):
        super(PLIFTauChannelwiseDynamicThresholdNode, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.channels = int(channels)
        self.tau_raw = nn.Parameter(
            self._inverse_softplus(tau - 1.0).repeat(self.channels)
        )
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        taus = 1.0 + F.softplus(self.tau_raw)
        threshold = taus.new_tensor(self.v_threshold)
        return taus, threshold

    def effective_taus(self):
        taus, _ = self.current_parameters()
        return taus

    def single_step_forward(self, x, threshold=None):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )

        taus, base_threshold = self.current_parameters()
        tau_per_channel = taus.to(device=x.device, dtype=x.dtype).view(
            1, self.channels, 1
        )

        if threshold is None:
            threshold_to_use = base_threshold
        else:
            if threshold.shape != x.shape:
                raise ValueError("threshold must have the same shape as x")
            threshold_to_use = threshold.to(device=x.device, dtype=x.dtype)

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        self.v = self.v + (x - self.v) / tau_per_channel
        spike = SurrogateSpike.apply(self.v - threshold_to_use)
        self.v = self.v - spike * threshold_to_use
        return spike

    def forward(self, x, threshold=None):
        return self.single_step_forward(x, threshold=threshold)


class SpikeGatedResetMessage(nn.Module):
    def __init__(
        self,
        channels,
        k=3,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.02,
        gamma_max=0.20,
        decay=0.7,
    ):
        super(SpikeGatedResetMessage, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")

        self.channels = int(channels)
        self.k = int(k)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        delay_weights = torch.tensor(
            [decay**delay for delay in range(self.k)], dtype=torch.float32
        )
        delay_weights = delay_weights / delay_weights.sum()
        self.register_buffer("delay_weight", delay_weights.view(1, 1, self.k))

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def _shift_past(self, x, delay):
        T = x.size(2)
        if T == 0:
            return x
        return F.pad(x, (delay, 0))[:, :, :T]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        reset_msg = torch.zeros_like(x)
        for idx in range(self.k):
            delay = idx + 1
            reset_msg = reset_msg + self.delay_weight[:, :, idx].view(1, 1, 1).to(
                device=x.device, dtype=x.dtype
            ) * self._shift_past(p, delay)

        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return x - gamma * self.threshold * reset_msg


class SGRChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        k=3,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.02,
        gamma_max=0.20,
        decay=0.7,
    ):
        super(SGRChannelwisePLIF, self).__init__()
        self.sgr = SpikeGatedResetMessage(
            channels=channels,
            k=k,
            threshold=threshold,
            beta=beta,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            decay=decay,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class FullHistorySpikeGatedResetMessage(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        decay=0.7,
        max_kernel_len=64,
    ):
        super(FullHistorySpikeGatedResetMessage, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if max_kernel_len <= 0:
            raise ValueError("max_kernel_len must be positive")

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.max_kernel_len = int(max_kernel_len)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        base_kernel = torch.tensor(
            [decay**delay for delay in range(self.max_kernel_len)], dtype=torch.float32
        )
        base_kernel = base_kernel / base_kernel.sum()
        self.register_buffer("base_kernel", base_kernel)

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def _build_kernel(self, T, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        kernel = self.base_kernel[:kernel_len].to(device=device, dtype=dtype)
        kernel = kernel / kernel.sum()
        kernel = torch.flip(kernel, dims=[0])
        return kernel.view(1, 1, kernel_len).repeat(self.channels, 1, 1)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        kernel = self._build_kernel(x.size(2), x.dtype, x.device)
        pad = kernel.size(2)
        p_pad = F.pad(p, (pad, 0))
        reset_msg = F.conv1d(p_pad, kernel, groups=self.channels)[:, :, : x.size(2)]
        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return x - gamma * self.threshold * reset_msg


class FullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        decay=0.7,
        max_kernel_len=64,
    ):
        super(FullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = FullHistorySpikeGatedResetMessage(
            channels=channels,
            threshold=threshold,
            beta=beta,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            decay=decay,
            max_kernel_len=max_kernel_len,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class STSGRFullHistoryThreshold(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        decay=0.7,
        max_kernel_len=64,
    ):
        super(STSGRFullHistoryThreshold, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if max_kernel_len <= 0:
            raise ValueError("max_kernel_len must be positive")

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.max_kernel_len = int(max_kernel_len)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        base_kernel = torch.tensor(
            [decay**delay for delay in range(self.max_kernel_len)], dtype=torch.float32
        )
        base_kernel = base_kernel / base_kernel.sum()
        self.register_buffer("base_kernel", base_kernel)

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def _build_kernel(self, T, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        kernel = self.base_kernel[:kernel_len].to(device=device, dtype=dtype)
        kernel = kernel / kernel.sum()
        kernel = torch.flip(kernel, dims=[0])
        return kernel.view(1, 1, kernel_len).repeat(self.channels, 1, 1)

    def _causal_depthwise_conv(self, x, kernel):
        pad = kernel.size(2)
        x_pad = F.pad(x, (pad, 0))
        return F.conv1d(x_pad, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        kernel = self._build_kernel(x.size(2), x.dtype, x.device)
        reset_msg = self._causal_depthwise_conv(p, kernel)
        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return self.threshold * (1.0 + gamma * reset_msg)


class STSGRFullHistoryChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        decay=0.7,
        max_kernel_len=64,
    ):
        super(STSGRFullHistoryChannelwisePLIF, self).__init__()
        self.threshold_builder = STSGRFullHistoryThreshold(
            channels=channels,
            threshold=threshold,
            beta=beta,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            decay=decay,
            max_kernel_len=max_kernel_len,
        )
        self.neuron = PLIFTauChannelwiseDynamicThresholdNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_taus(self):
        return self.neuron.effective_taus()

    def forward(self, x):
        threshold = self.threshold_builder(x)
        return self.neuron(x, threshold=threshold)


class MultiStateFullHistorySGR(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        rho_fast=0.5,
        rho_slow=0.9,
        max_kernel_len=64,
    ):
        super(MultiStateFullHistorySGR, self).__init__()
        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.max_kernel_len = int(max_kernel_len)
        kernel_fast = self._make_kernel(rho_fast, max_kernel_len)
        kernel_slow = self._make_kernel(rho_slow, max_kernel_len)
        self.register_buffer("kernel_fast", kernel_fast)
        self.register_buffer("kernel_slow", kernel_slow)
        init_ratio = gamma_init / gamma_max
        gv = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gv.item()))
        rr = (rho_init - rho_min) / (rho_max - rho_min)
        self.rho_raw = nn.Parameter(
            torch.logit(torch.tensor(rr, dtype=torch.float32)).view(())
        )
        self.a_fast = nn.Parameter(torch.zeros(1, self.channels, 1))
        self.a_slow = nn.Parameter(torch.zeros(1, self.channels, 1))

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def effective_rho(self):
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(
            self.rho_raw
        )

    @staticmethod
    def _make_kernel(rho, D):
        d = torch.arange(D, dtype=torch.float32)
        k = rho**d
        k = k / k.sum()
        k = torch.flip(k, dims=[0])
        return k.view(1, 1, D)

    def _build_kernel(self, T, static_kernel, dtype, device):
        kl = min(T, self.max_kernel_len)
        k = static_kernel.to(device=device, dtype=dtype)[:, :, -kl:]
        return k.repeat(self.channels, 1, 1)

    def _build_rho_kernel(self, T, dtype, device):
        kl = min(T, self.max_kernel_len)
        rho = self.effective_rho().to(device=device, dtype=dtype)
        exponents = torch.arange(kl, device=device, dtype=dtype)
        k = torch.pow(rho, exponents)
        k = k / k.sum()
        k = torch.flip(k, dims=[0])
        return k.view(1, 1, kl).repeat(self.channels, 1, 1)

    def _cc(self, x, kernel):
        pad = kernel.size(2)
        xp = F.pad(x, (pad, 0))
        return F.conv1d(xp, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3 or x.size(1) != self.channels:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(2) == 0:
            return x.new_empty(x.shape)
        p = torch.sigmoid(self.beta * (x - self.threshold))
        kf = self._build_kernel(x.size(2), self.kernel_fast, x.dtype, x.device)
        ks = self._build_kernel(x.size(2), self.kernel_slow, x.dtype, x.device)
        i_f = self._cc(p, kf)
        i_s = self._cc(p, ks)
        p2 = torch.sigmoid(
            self.beta * (x + self.a_fast * i_f + self.a_slow * i_s - self.threshold)
        )
        kr = self._build_rho_kernel(x.size(2), x.dtype, x.device)
        r = self._cc(p2, kr)
        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return x - gamma * self.threshold * r


class MultiStateFullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(self, channels, tau=2.0, v_threshold=0.25, **sgr_kwargs):
        super(MultiStateFullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = MultiStateFullHistorySGR(channels=channels, **sgr_kwargs)
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def effective_rho(self):
        return self.sgr.effective_rho()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class LearnableDecayFullHistorySpikeGatedResetMessage(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        max_kernel_len=64,
    ):
        super(LearnableDecayFullHistorySpikeGatedResetMessage, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < rho_min < rho_init < rho_max < 1.0:
            raise ValueError(
                "rho_min, rho_init, and rho_max must satisfy 0 < rho_min < rho_init < rho_max < 1"
            )
        if max_kernel_len <= 0:
            raise ValueError("max_kernel_len must be positive")

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.max_kernel_len = int(max_kernel_len)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        rho_ratio = (rho_init - rho_min) / (rho_max - rho_min)
        rho_value = torch.logit(torch.tensor(rho_ratio, dtype=torch.float32))
        self.rho_raw = nn.Parameter(rho_value.view(()))

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def effective_rho(self):
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(
            self.rho_raw
        )

    def _build_kernel(self, T, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        rho = self.effective_rho().to(device=device, dtype=dtype)
        exponents = torch.arange(kernel_len, device=device, dtype=dtype)
        kernel = torch.pow(rho, exponents)
        kernel = kernel / kernel.sum()
        kernel = torch.flip(kernel, dims=[0])
        return kernel.view(1, 1, kernel_len).repeat(self.channels, 1, 1)

    def _causal_depthwise_conv(self, x, kernel):
        pad = kernel.size(2)
        x_pad = F.pad(x, (pad, 0))
        return F.conv1d(x_pad, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        kernel = self._build_kernel(x.size(2), x.dtype, x.device)
        reset_msg = self._causal_depthwise_conv(p, kernel)
        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return x - gamma * self.threshold * reset_msg


class PerChannelLearnableDecayFullHistorySGR(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        max_kernel_len=64,
    ):
        super(PerChannelLearnableDecayFullHistorySGR, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < rho_min < rho_init < rho_max < 1.0:
            raise ValueError(
                "rho_min, rho_init, and rho_max must satisfy 0 < rho_min < rho_init < rho_max < 1"
            )
        if max_kernel_len <= 0:
            raise ValueError("max_kernel_len must be positive")

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.max_kernel_len = int(max_kernel_len)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        rho_ratio = (rho_init - rho_min) / (rho_max - rho_min)
        rho_value = torch.logit(torch.tensor(rho_ratio, dtype=torch.float32))
        self.rho_raw = nn.Parameter(torch.full((1, self.channels, 1), rho_value.item()))

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def effective_rho(self):
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(
            self.rho_raw
        )

    def _build_kernel(self, T, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        rho = self.effective_rho().to(device=device, dtype=dtype)
        exponents = torch.arange(kernel_len, device=device, dtype=dtype)
        kernel = torch.pow(rho, exponents)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        kernel = torch.flip(kernel, dims=[-1])
        return kernel.view(self.channels, 1, kernel_len)

    def _causal_depthwise_conv(self, x, kernel):
        pad = kernel.size(2)
        x_pad = F.pad(x, (pad, 0))
        return F.conv1d(x_pad, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        kernel = self._build_kernel(x.size(2), x.dtype, x.device)
        reset_msg = self._causal_depthwise_conv(p, kernel)
        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return x - gamma * self.threshold * reset_msg


class PerChannelLearnableDecayFullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(self, channels, tau=2.0, v_threshold=0.25, **sgr_kwargs):
        super(PerChannelLearnableDecayFullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = PerChannelLearnableDecayFullHistorySGR(
            channels=channels, **sgr_kwargs
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def effective_rho(self):
        return self.sgr.effective_rho()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class LearnableDecayFullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        max_kernel_len=64,
    ):
        super(LearnableDecayFullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = LearnableDecayFullHistorySpikeGatedResetMessage(
            channels=channels,
            threshold=threshold,
            beta=beta,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            rho_init=rho_init,
            rho_min=rho_min,
            rho_max=rho_max,
            max_kernel_len=max_kernel_len,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def effective_rho(self):
        return self.sgr.effective_rho()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class GatedModeFullHistorySGR(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        gate_init=0.8,
        max_kernel_len=64,
    ):
        super(GatedModeFullHistorySGR, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < rho_min < rho_init < rho_max < 1.0:
            raise ValueError(
                "rho_min, rho_init, and rho_max must satisfy 0 < rho_min < rho_init < rho_max < 1"
            )
        if max_kernel_len <= 0:
            raise ValueError("max_kernel_len must be positive")

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.max_kernel_len = int(max_kernel_len)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        rho_ratio = (rho_init - rho_min) / (rho_max - rho_min)
        rho_value = torch.logit(torch.tensor(rho_ratio, dtype=torch.float32))
        self.rho_raw = nn.Parameter(rho_value.view(()))

        mode_init = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
        self.mode_gate = nn.Parameter(
            torch.full((1, self.channels, 1), mode_init.item())
        )

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def effective_rho(self):
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(
            self.rho_raw
        )

    def _build_kernel(self, T, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        rho = self.effective_rho().to(device=device, dtype=dtype)
        exponents = torch.arange(kernel_len, device=device, dtype=dtype)
        kernel = torch.pow(rho, exponents)
        kernel = kernel / kernel.sum()
        kernel = torch.flip(kernel, dims=[0])
        return kernel.view(1, 1, kernel_len).repeat(self.channels, 1, 1)

    def _causal_depthwise_conv(self, x, kernel):
        pad = kernel.size(2)
        x_pad = F.pad(x, (pad, 0))
        return F.conv1d(x_pad, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        kernel = self._build_kernel(x.size(2), x.dtype, x.device)
        reset_msg = self._causal_depthwise_conv(p, kernel)
        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)

        x_sub = x - gamma * self.threshold * reset_msg

        gain = torch.clamp(1.0 - gamma * reset_msg, min=0.0, max=1.0)
        x_mul = x * gain

        g = torch.sigmoid(self.mode_gate)
        return g * x_sub + (1.0 - g) * x_mul


class GatedModeFullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        gate_init=0.8,
        max_kernel_len=64,
    ):
        super(GatedModeFullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = GatedModeFullHistorySGR(
            channels=channels,
            threshold=threshold,
            beta=beta,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            rho_init=rho_init,
            rho_min=rho_min,
            rho_max=rho_max,
            gate_init=gate_init,
            max_kernel_len=max_kernel_len,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def effective_rho(self):
        return self.sgr.effective_rho()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class DoubleTimescaleFullHistorySGR(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        rho_fast=0.5,
        rho_slow=0.9,
        gamma_fast_init=0.01,
        gamma_slow_init=0.003,
        gamma_fast_max=0.20,
        gamma_slow_max=0.10,
        max_kernel_len=64,
    ):
        super(DoubleTimescaleFullHistorySGR, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 < rho_fast < rho_slow < 1.0:
            raise ValueError(
                "rho_fast and rho_slow must satisfy 0 < rho_fast < rho_slow < 1"
            )

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_fast_max = float(gamma_fast_max)
        self.gamma_slow_max = float(gamma_slow_max)
        self.max_kernel_len = int(max_kernel_len)

        kernel_fast = self._make_kernel_static(rho_fast, max_kernel_len)
        kernel_slow = self._make_kernel_static(rho_slow, max_kernel_len)
        self.register_buffer("kernel_fast", kernel_fast)
        self.register_buffer("kernel_slow", kernel_slow)

        self.gate_fast = nn.Parameter(
            torch.full(
                (1, self.channels, 1),
                torch.logit(
                    torch.tensor(gamma_fast_init / gamma_fast_max, dtype=torch.float32)
                ).item(),
            )
        )
        self.gate_slow = nn.Parameter(
            torch.full(
                (1, self.channels, 1),
                torch.logit(
                    torch.tensor(gamma_slow_init / gamma_slow_max, dtype=torch.float32)
                ).item(),
            )
        )

    @staticmethod
    def _make_kernel_static(rho, D):
        d = torch.arange(D, dtype=torch.float32)
        k = rho**d
        k = k / k.sum()
        k = torch.flip(k, dims=[0])
        return k.view(1, 1, D)

    def effective_gammas_fast(self):
        return self.gamma_fast_max * torch.sigmoid(self.gate_fast)

    def effective_gammas_slow(self):
        return self.gamma_slow_max * torch.sigmoid(self.gate_slow)

    def _build_kernel_for_t(self, T, static_kernel, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        k = static_kernel.to(device=device, dtype=dtype)[:, :, -kernel_len:]
        return k.repeat(self.channels, 1, 1)

    def _causal_depthwise_conv(self, x, kernel):
        pad = kernel.size(2)
        x_pad = F.pad(x, (pad, 0))
        return F.conv1d(x_pad, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))

        kf = self._build_kernel_for_t(x.size(2), self.kernel_fast, x.dtype, x.device)
        ks = self._build_kernel_for_t(x.size(2), self.kernel_slow, x.dtype, x.device)
        r_fast = self._causal_depthwise_conv(p, kf)
        r_slow = self._causal_depthwise_conv(p, ks)

        gamma_fast = self.effective_gammas_fast().to(device=x.device, dtype=x.dtype)
        gamma_slow = self.effective_gammas_slow().to(device=x.device, dtype=x.dtype)

        reset = gamma_fast * r_fast + gamma_slow * r_slow
        return x - self.threshold * reset


class DoubleTimescaleFullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        threshold=0.25,
        beta=5.0,
        rho_fast=0.5,
        rho_slow=0.9,
        gamma_fast_init=0.01,
        gamma_slow_init=0.003,
        gamma_fast_max=0.20,
        gamma_slow_max=0.10,
        max_kernel_len=64,
    ):
        super(DoubleTimescaleFullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = DoubleTimescaleFullHistorySGR(
            channels=channels,
            threshold=threshold,
            beta=beta,
            rho_fast=rho_fast,
            rho_slow=rho_slow,
            gamma_fast_init=gamma_fast_init,
            gamma_slow_init=gamma_slow_init,
            gamma_fast_max=gamma_fast_max,
            gamma_slow_max=gamma_slow_max,
            max_kernel_len=max_kernel_len,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas_fast(self):
        return self.sgr.effective_gammas_fast()

    def effective_gammas_slow(self):
        return self.sgr.effective_gammas_slow()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class CrossChannelFullHistorySGR(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        groups=4,
        max_kernel_len=64,
    ):
        super(CrossChannelFullHistorySGR, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < rho_min < rho_init < rho_max < 1.0:
            raise ValueError(
                "rho_min, rho_init, and rho_max must satisfy 0 < rho_min < rho_init < rho_max < 1"
            )
        if max_kernel_len <= 0:
            raise ValueError("max_kernel_len must be positive")

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.groups = min(int(groups), self.channels)
        if self.channels % self.groups != 0:
            self.groups = 1
        self.max_kernel_len = int(max_kernel_len)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        rho_ratio = (rho_init - rho_min) / (rho_max - rho_min)
        rho_value = torch.logit(torch.tensor(rho_ratio, dtype=torch.float32))
        self.rho_raw = nn.Parameter(rho_value.view(()))

        self.mix = nn.Conv1d(
            self.channels, self.channels, kernel_size=1, groups=self.groups, bias=False
        )
        nn.init.zeros_(self.mix.weight)

        self.alpha = nn.Parameter(torch.zeros(1, self.channels, 1))

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def effective_rho(self):
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(
            self.rho_raw
        )

    def _build_kernel(self, T, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        rho = self.effective_rho().to(device=device, dtype=dtype)
        exponents = torch.arange(kernel_len, device=device, dtype=dtype)
        kernel = torch.pow(rho, exponents)
        kernel = kernel / kernel.sum()
        kernel = torch.flip(kernel, dims=[0])
        return kernel.view(1, 1, kernel_len).repeat(self.channels, 1, 1)

    def _causal_depthwise_conv(self, x, kernel):
        pad = kernel.size(2)
        x_pad = F.pad(x, (pad, 0))
        return F.conv1d(x_pad, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        kernel = self._build_kernel(x.size(2), x.dtype, x.device)
        reset_msg = self._causal_depthwise_conv(p, kernel)

        mixed = reset_msg + torch.tanh(self.alpha) * self.mix(reset_msg)

        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return x - gamma * self.threshold * mixed


class CrossChannelFullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        rho_init=0.7,
        rho_min=0.3,
        rho_max=0.98,
        groups=4,
        max_kernel_len=64,
    ):
        super(CrossChannelFullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = CrossChannelFullHistorySGR(
            channels=channels,
            threshold=threshold,
            beta=beta,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            rho_init=rho_init,
            rho_min=rho_min,
            rho_max=rho_max,
            groups=groups,
            max_kernel_len=max_kernel_len,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def effective_rho(self):
        return self.sgr.effective_rho()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class VisibleNormalizedFullHistorySpikeGatedResetMessage(nn.Module):
    def __init__(
        self,
        channels,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        decay=0.7,
        max_kernel_len=64,
        eps=1e-6,
    ):
        super(VisibleNormalizedFullHistorySpikeGatedResetMessage, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < gamma_init < gamma_max < 1.0:
            raise ValueError(
                "gamma_init and gamma_max must satisfy 0 < gamma_init < gamma_max < 1"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if max_kernel_len <= 0:
            raise ValueError("max_kernel_len must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.channels = int(channels)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.gamma_max = float(gamma_max)
        self.max_kernel_len = int(max_kernel_len)
        self.eps = float(eps)

        init_ratio = gamma_init / gamma_max
        gate_value = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((1, self.channels, 1), gate_value.item()))

        base_kernel = torch.tensor(
            [decay**delay for delay in range(self.max_kernel_len)], dtype=torch.float32
        )
        base_kernel = base_kernel / base_kernel.sum()
        self.register_buffer("base_kernel", base_kernel)

    def effective_gammas(self):
        return self.gamma_max * torch.sigmoid(self.gate)

    def _build_kernel(self, T, dtype, device):
        kernel_len = min(T, self.max_kernel_len)
        kernel = self.base_kernel[:kernel_len].to(device=device, dtype=dtype)
        kernel = kernel / kernel.sum()
        kernel = torch.flip(kernel, dims=[0])
        return kernel.view(1, 1, kernel_len).repeat(self.channels, 1, 1)

    def _causal_depthwise_conv(self, x, kernel):
        pad = kernel.size(2)
        x_pad = F.pad(x, (pad, 0))
        return F.conv1d(x_pad, kernel, groups=self.channels)[:, :, : x.size(2)]

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        p = torch.sigmoid(self.beta * (x - self.threshold))
        kernel = self._build_kernel(x.size(2), x.dtype, x.device)
        reset_num = self._causal_depthwise_conv(p, kernel)
        visible_mass = self._causal_depthwise_conv(torch.ones_like(p), kernel)
        reset_msg = reset_num / (visible_mass + self.eps)
        gamma = self.effective_gammas().to(device=x.device, dtype=x.dtype)
        return x - gamma * self.threshold * reset_msg


class VisibleNormalizedFullHistorySGRChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        threshold=0.25,
        beta=5.0,
        gamma_init=0.01,
        gamma_max=0.10,
        decay=0.7,
        max_kernel_len=64,
        eps=1e-6,
    ):
        super(VisibleNormalizedFullHistorySGRChannelwisePLIF, self).__init__()
        self.sgr = VisibleNormalizedFullHistorySpikeGatedResetMessage(
            channels=channels,
            threshold=threshold,
            beta=beta,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            decay=decay,
            max_kernel_len=max_kernel_len,
            eps=eps,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_gammas(self):
        return self.sgr.effective_gammas()

    def effective_taus(self):
        return self.neuron.effective_taus()

    def forward(self, x):
        return self.neuron(self.sgr(x))


class ResidualCausalTemporalMixChannelwisePLIF(nn.Module):
    def __init__(
        self,
        channels,
        tau=2.0,
        v_threshold=0.25,
        kernel_size=3,
        alpha_init=0.01,
        ema_decay=0.6,
        padding_mode="zero",
    ):
        super(ResidualCausalTemporalMixChannelwisePLIF, self).__init__()
        self.mix = ResidualCausalTemporalMix(
            channels=channels,
            kernel_size=kernel_size,
            alpha_init=alpha_init,
            ema_decay=ema_decay,
            padding_mode=padding_mode,
        )
        self.neuron = PLIFTauChannelwiseNode(
            channels=channels, tau=tau, v_threshold=v_threshold
        )

    def effective_taus(self):
        return self.neuron.effective_taus()

    def forward(self, x):
        return self.neuron(self.mix(x))


class PLIFTauTemporalNode(base.MemoryModule):
    def __init__(self, tau=2.0, v_threshold=0.25):
        super(PLIFTauTemporalNode, self).__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.tau_raw = nn.Parameter(self._inverse_softplus(tau - 1.0))
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        tau = 1.0 + F.softplus(self.tau_raw)
        threshold = tau.new_tensor(self.v_threshold)
        return tau, threshold

    def effective_taus(self):
        tau, threshold = self.current_parameters()
        return tau.view(1)

    def single_step_forward(self, x):
        tau, threshold = self.current_parameters()

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        self.v = self.v + (x - self.v) / tau
        spike = SurrogateSpike.apply(self.v - threshold)
        self.v = self.v - spike * threshold
        return spike

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")

        if x.size(2) == 0:
            return x.new_empty(x.shape)

        spikes = []
        for step in range(x.size(2)):
            spike = self.single_step_forward(x[:, :, step])
            spikes.append(spike.unsqueeze(2))

        return torch.cat(spikes, dim=2)


class PLIFTauTemporalChannelwiseNode(base.MemoryModule):
    def __init__(self, channels, tau=2.0, v_threshold=0.25):
        super(PLIFTauTemporalChannelwiseNode, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.channels = int(channels)
        self.tau_raw = nn.Parameter(
            self._inverse_softplus(tau - 1.0).repeat(self.channels)
        )
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        taus = 1.0 + F.softplus(self.tau_raw)
        threshold = taus.new_tensor(self.v_threshold)
        return taus, threshold

    def effective_taus(self):
        taus, _ = self.current_parameters()
        return taus

    def single_step_forward(self, x):
        if x.dim() != 2:
            raise ValueError(
                "PLIFTauTemporalChannelwiseNode expects input shape (batch, channels)"
            )
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )

        taus, threshold = self.current_parameters()
        tau_per_channel = taus.to(device=x.device, dtype=x.dtype).view(1, self.channels)

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        self.v = self.v + (x - self.v) / tau_per_channel
        spike = SurrogateSpike.apply(self.v - threshold)
        self.v = self.v - spike * threshold
        return spike

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )
        if x.size(2) == 0:
            return x.new_empty(x.shape)

        spikes = []
        for step in range(x.size(2)):
            spike = self.single_step_forward(x[:, :, step])
            spikes.append(spike.unsqueeze(2))

        return torch.cat(spikes, dim=2)


class PLIFTauPSNNode(base.MemoryModule):
    def __init__(self, tau=2.0, v_threshold=0.25):
        super(PLIFTauPSNNode, self).__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.tau_raw = nn.Parameter(self._inverse_softplus(tau - 1.0))
        self.v_threshold = float(v_threshold)

        init_time = 2048
        init_weight = self._default_time_weight(init_time)
        self.time_weight = nn.Parameter(init_weight)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        tau = 1.0 + F.softplus(self.tau_raw)
        threshold = tau.new_tensor(self.v_threshold)
        return tau, threshold

    def effective_taus(self):
        tau, _ = self.current_parameters()
        return tau.view(1)

    def _default_time_weight(self, size):
        tau, _ = self.current_parameters()
        gamma = 1.0 - 1.0 / tau
        index = torch.arange(size, dtype=tau.dtype, device=tau.device)
        lag = index.unsqueeze(1) - index.unsqueeze(0)
        lag = (-lag).clamp_min(0)
        weight = torch.pow(gamma, lag)
        return torch.triu(weight).detach()

    def _expand_time_weight(self, new_size):
        old = self.time_weight.data
        old_size = old.size(0)
        if new_size <= old_size:
            return

        expanded = self._default_time_weight(new_size).to(
            device=old.device, dtype=old.dtype
        )
        expanded[:old_size, :old_size] = old
        with torch.no_grad():
            self.time_weight.data = expanded

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")

        T = x.size(2)
        if T == 0:
            return x.new_empty(x.shape)

        current_max = self.time_weight.size(0)
        if T > current_max:
            self._expand_time_weight(T)

        tau, threshold = self.current_parameters()
        charging = x / tau

        weight_slice = self.time_weight[:T, :T].to(device=x.device, dtype=x.dtype)
        mixed = charging @ weight_slice
        spike = SurrogateSpike.apply(mixed - threshold)
        return spike


class PLIFCLIFTauNode(base.MemoryModule):
    def __init__(self, tau=2.0, v_threshold=0.25):
        super(PLIFCLIFTauNode, self).__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.tau_raw = nn.Parameter(self._inverse_softplus(tau - 1.0))
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)
        self.register_memory("m", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        tau = 1.0 + F.softplus(self.tau_raw)
        threshold = tau.new_tensor(self.v_threshold)
        return tau, threshold

    def effective_taus(self):
        tau, _ = self.current_parameters()
        return tau.view(1)

    def single_step_forward(self, x):
        tau, threshold = self.current_parameters()
        gamma = 1.0 - 1.0 / tau

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        if self.m is None or self.m.shape != x.shape:
            self.m = torch.zeros_like(x)
        else:
            self.m = self.m.to(device=x.device, dtype=x.dtype)

        self.v = gamma * self.v + x
        self.m = self.m * torch.sigmoid(self.v / tau)
        spike = SurrogateSpike.apply(self.v - threshold)
        self.m = self.m + spike
        self.v = self.v - spike * threshold
        self.v = self.v - spike * torch.sigmoid(self.m)
        return spike


class PLIFCLIFTemporalNode(base.MemoryModule):
    def __init__(self, tau=2.0, v_threshold=0.25):
        super(PLIFCLIFTemporalNode, self).__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.tau_raw = nn.Parameter(self._inverse_softplus(tau - 1.0))
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)
        self.register_memory("m", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        tau = 1.0 + F.softplus(self.tau_raw)
        threshold = tau.new_tensor(self.v_threshold)
        return tau, threshold

    def effective_taus(self):
        tau, _ = self.current_parameters()
        return tau.view(1)

    def single_step_forward(self, x):
        tau, threshold = self.current_parameters()
        gamma = 1.0 - 1.0 / tau

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        if self.m is None or self.m.shape != x.shape:
            self.m = torch.zeros_like(x)
        else:
            self.m = self.m.to(device=x.device, dtype=x.dtype)

        self.v = gamma * self.v + x
        self.m = self.m * torch.sigmoid(self.v / tau)
        spike = SurrogateSpike.apply(self.v - threshold)
        self.m = self.m + spike
        self.v = self.v - spike * threshold
        self.v = self.v - spike * torch.sigmoid(self.m)
        return spike

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("x must have shape (batch, channels, time)")

        if x.size(2) == 0:
            return x.new_empty(x.shape)

        spikes = []
        for step in range(x.size(2)):
            spike = self.single_step_forward(x[:, :, step])
            spikes.append(spike.unsqueeze(2))

        return torch.cat(spikes, dim=2)


class PLIFGroupTauNode(base.MemoryModule):
    def __init__(self, channels, group_num=4, tau=2.0, v_threshold=0.25):
        super(PLIFGroupTauNode, self).__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if group_num <= 0:
            raise ValueError("group_num must be positive")
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1.0")
        if v_threshold <= 0.0:
            raise ValueError("v_threshold must be positive")

        self.channels = int(channels)
        self.group_num = int(group_num)
        self.tau_raw = nn.Parameter(
            self._inverse_softplus(tau - 1.0).repeat(self.group_num)
        )
        self.v_threshold = float(v_threshold)
        self.register_memory("v", None)

    @staticmethod
    def _inverse_softplus(value):
        value_tensor = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value_tensor))

    def current_parameters(self):
        taus = 1.0 + F.softplus(self.tau_raw)
        threshold = taus.new_tensor(self.v_threshold)
        return taus, threshold

    def effective_taus(self):
        taus, _ = self.current_parameters()
        return taus

    def _expand_group_taus(self, x):
        taus, _ = self.current_parameters()
        chunk_sizes = []
        base = self.channels // self.group_num
        remainder = self.channels % self.group_num
        for index in range(self.group_num):
            chunk_sizes.append(base + (1 if index < remainder else 0))
        repeats = torch.tensor(chunk_sizes, device=taus.device, dtype=torch.long)
        tau_per_channel = torch.repeat_interleave(taus, repeats)
        tau_per_channel = tau_per_channel.to(device=x.device, dtype=x.dtype)
        return tau_per_channel.view(1, self.channels, 1)

    def single_step_forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                "PLIFGroupTauNode expects input shape (batch, channels, length)"
            )
        if x.size(1) != self.channels:
            raise ValueError(
                "input channel dimension does not match configured channels"
            )

        _, threshold = self.current_parameters()
        tau_per_channel = self._expand_group_taus(x)

        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)

        self.v = self.v + (x - self.v) / tau_per_channel
        spike = SurrogateSpike.apply(self.v - threshold)
        self.v = self.v - spike * threshold
        return spike


class LIFBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, decay=0.9, threshold=1.0):
        super(LIFBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.BatchNorm1d(n_out)
        self.decay = decay
        self.threshold = threshold

    def forward(self, x):
        current = self.norm(self.conv(x))
        state = current.new_zeros(current.size(0), current.size(1))
        spikes = []
        spike_fn = SurrogateSpike.apply

        for step in range(current.size(2)):
            state = self.decay * state + current[:, :, step]
            spike = spike_fn(state - self.threshold)
            state = state - spike * self.threshold
            spikes.append(spike.unsqueeze(2))

        spikes = torch.cat(spikes, dim=2)
        firing_rate = spikes.mean()
        return spikes, firing_rate


class selectivePooling(nn.Module):
    def __init__(self, in_dim, head_dim, num_heads, tau=1.0):
        super(selectivePooling, self).__init__()
        self.keys = nn.Parameter(torch.Tensor(num_heads, head_dim), requires_grad=True)
        self.w_q = nn.Conv1d(in_dim, head_dim * num_heads, kernel_size=1)
        self.norm = 1 / head_dim**0.5
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.count = 0
        nn.init.orthogonal_(self.keys, gain=1)
        nn.init.kaiming_normal_(self.w_q.weight, a=1)
        nn.init.zeros_(self.w_q.bias)

    def forward(self, x, mask, save=False):
        n_batch = x.shape[0]
        n_steps = x.shape[2]
        queries = values = (
            self.w_q(x)
            .transpose(1, 2)
            .view(n_batch, n_steps, self.num_heads, self.head_dim)
        )
        atten = F.softmax(
            torch.sum(queries * self.keys[None, None, :, :], dim=-1) * self.norm
            - (1.0 - mask).unsqueeze(2) * 1000,
            dim=1,
        )
        return torch.sum(values * atten.unsqueeze(3), dim=1).view(n_batch, -1)

    def orthoNorm(self):
        keys = F.normalize(self.keys, dim=1)
        corr = torch.mm(keys, keys.transpose(0, 1))
        return torch.sum(torch.triu(corr, 1).abs_())


def build_user_prototypes(embedding, labels):
    if embedding.size(0) != labels.size(0):
        raise ValueError("embedding and labels must have the same batch size")

    prototype_labels = labels.unique(sorted=True)
    prototypes = []
    for label in prototype_labels:
        prototypes.append(embedding[labels == label].mean(dim=0, keepdim=True))
    return torch.cat(prototypes, dim=0), prototype_labels


class LearnableThresholdBurstPLIFTauChannelwiseNode(BurstPLIFTauChannelwiseNode):
    def __init__(self, channels, tau=2.0, v_threshold=1.0, burst_k=10, thresholds=None):
        super().__init__(
            channels=channels,
            tau=tau,
            v_threshold=v_threshold,
            burst_k=burst_k,
            thresholds=thresholds,
            normalize=False,
        )
        _thresholds = (
            thresholds if thresholds is not None else list(range(1, burst_k + 1))
        )
        if len(_thresholds) != self.burst_k:
            raise ValueError(
                "thresholds length %d != burst_k %d" % (len(_thresholds), self.burst_k)
            )
        diffs = numpy.diff([v_threshold] + list(_thresholds))
        init_log_spacings = numpy.log(numpy.maximum(diffs, 1e-3))
        self.log_spacings = nn.Parameter(
            torch.tensor(init_log_spacings, dtype=torch.float32)
        )

    def single_step_forward(self, x):
        spacings = torch.exp(self.log_spacings)
        thresholds = torch.cumsum(spacings, dim=0)

        taus, v_threshold = self.current_parameters()
        tau_per_channel = taus.to(device=x.device, dtype=x.dtype).view(
            1, self.channels, 1
        )
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        else:
            self.v = self.v.to(device=x.device, dtype=x.dtype)
        self.v = self.v + (x - self.v) / tau_per_channel
        spike_count = torch.zeros_like(self.v)
        for k in range(1, self.burst_k + 1):
            th = thresholds[k - 1].to(dtype=x.dtype, device=x.device) * v_threshold
            spike_count = spike_count + SurrogateSpike.apply(self.v - th)
        self.v = self.v - spike_count * v_threshold
        return spike_count


class Sig2VecSNN(nn.Module):
    def __init__(
        self,
        n_in,
        n_classes,
        n_task=1,
        n_shot_g=5,
        n_shot_f=5,
        APAlpha=5.0,
        embedding_dim=1024,
        spike_threshold=0.25,
        sim_steps=None,
        event_steps=4,
        event_threshold=None,
        lif_tau=2.0,
        lif_v_threshold=None,
        lambda_div=1e-5,
        tau_psn=False,
        clif_tau=False,
        temporal_tau=False,
        temporal_tau_channelwise=False,
        tau_channelwise=False,
        rctm_channelwise_plif=False,
        sgr_channelwise_plif=False,
        full_history_sgr_channelwise_plif=False,
        full_history_sgr_visible_norm_channelwise_plif=False,
        full_history_sgr_learnable_decay_channelwise_plif=False,
        st_sgr_full_history_channelwise_plif=False,
        gated_mode_full_history_sgr_channelwise_plif=False,
        cc_full_history_sgr_channelwise_plif=False,
        dt_full_history_sgr_channelwise_plif=False,
        per_channel_learnable_decay_full_history_sgr_channelwise_plif=False,
        multi_state_full_history_sgr_channelwise_plif=False,
        midlevel_forgery_margin=False,
        lambda_midlevel_margin=0.0,
        width_scale=1.0,
        burst_spike=False,
        burst_k=2,
        burst_slots=False,
        burst_raw_count=False,
        burst_signed_l5=False,
        burst_signed_all=False,
        learnable_l5_thresholds=False,
        learnable_l45_thresholds=False,
        learnable_all_thresholds=False,
        l4_tau=None,
        burst_k_per_layer=None,
        burst_thresholds=None,
        burst_thresholds_per_layer=None,
        clif_temporal=False,
        rollout_repeated_input=False,
        snn_readout="rate",
        readout_mode=None,
        dual_branch_readout=False,
        comparison_neuron=None,
        comparison_psn_order=2,
        comparison_sdn_checkpoint=None,
    ):
        super(Sig2VecSNN, self).__init__()
        if embedding_dim != 1024:
            raise ValueError("Task 3 restores the original 1024-dim Sig2Vec head")

        resolved_sim_steps = event_steps if sim_steps is None else sim_steps
        if int(resolved_sim_steps) <= 0:
            raise ValueError("sim_steps must be positive")

        self.n_in = n_in
        self.n_classes = n_classes
        self.n_task = n_task
        self.n_shot_g = n_shot_g
        self.n_shot_f = n_shot_f
        self.positive_update = True
        self.epsilon = 1
        self.APAlpha = APAlpha if self.positive_update else 1.0
        self.embedding_dim = 1024
        self.lif_tau = float(lif_tau)
        self.lif_v_threshold = (
            float(spike_threshold)
            if lif_v_threshold is None
            else float(lif_v_threshold)
        )
        self.lambda_div = float(lambda_div)
        self.comparison_neuron = comparison_neuron
        self.comparison_psn_order = int(comparison_psn_order)
        self.comparison_sdn_checkpoint = comparison_sdn_checkpoint
        comparison_choices = {
            "mlf-k3",
            "ilif",
            "klif",
            "scalar-plif",
            "sliding-psn-k2",
        }
        if self.comparison_neuron is not None:
            if self.comparison_neuron not in comparison_choices:
                raise ValueError("unknown comparison_neuron: %s" % self.comparison_neuron)
            conflicting = {
                "burst_spike": burst_spike,
                "tau_psn": tau_psn,
                "clif_tau": clif_tau,
                "clif_temporal": clif_temporal,
                "temporal_tau": temporal_tau,
                "temporal_tau_channelwise": temporal_tau_channelwise,
                "tau_channelwise": tau_channelwise,
                "rctm_channelwise_plif": rctm_channelwise_plif,
                "sgr_channelwise_plif": sgr_channelwise_plif,
                "full_history_sgr_channelwise_plif": full_history_sgr_channelwise_plif,
                "full_history_sgr_visible_norm_channelwise_plif": full_history_sgr_visible_norm_channelwise_plif,
                "full_history_sgr_learnable_decay_channelwise_plif": full_history_sgr_learnable_decay_channelwise_plif,
                "st_sgr_full_history_channelwise_plif": st_sgr_full_history_channelwise_plif,
                "gated_mode_full_history_sgr_channelwise_plif": gated_mode_full_history_sgr_channelwise_plif,
                "cc_full_history_sgr_channelwise_plif": cc_full_history_sgr_channelwise_plif,
                "dt_full_history_sgr_channelwise_plif": dt_full_history_sgr_channelwise_plif,
                "per_channel_learnable_decay_full_history_sgr_channelwise_plif": per_channel_learnable_decay_full_history_sgr_channelwise_plif,
                "multi_state_full_history_sgr_channelwise_plif": multi_state_full_history_sgr_channelwise_plif,
            }
            enabled = [name for name, value in conflicting.items() if value]
            if enabled:
                raise ValueError(
                    "comparison_neuron cannot be combined with: %s"
                    % ", ".join(enabled)
                )
        self.tau_psn = bool(tau_psn)
        self.clif_tau = bool(clif_tau)
        self.temporal_tau = bool(temporal_tau)
        self.temporal_tau_channelwise = bool(temporal_tau_channelwise)
        self.tau_channelwise = bool(tau_channelwise)
        self.rctm_channelwise_plif = bool(rctm_channelwise_plif)
        self.sgr_channelwise_plif = bool(sgr_channelwise_plif)
        self.full_history_sgr_channelwise_plif = bool(full_history_sgr_channelwise_plif)
        self.full_history_sgr_visible_norm_channelwise_plif = bool(
            full_history_sgr_visible_norm_channelwise_plif
        )
        self.full_history_sgr_learnable_decay_channelwise_plif = bool(
            full_history_sgr_learnable_decay_channelwise_plif
        )
        self.st_sgr_full_history_channelwise_plif = bool(
            st_sgr_full_history_channelwise_plif
        )
        self.gated_mode_full_history_sgr_channelwise_plif = bool(
            gated_mode_full_history_sgr_channelwise_plif
        )
        self.cc_full_history_sgr_channelwise_plif = bool(
            cc_full_history_sgr_channelwise_plif
        )
        self.dt_full_history_sgr_channelwise_plif = bool(
            dt_full_history_sgr_channelwise_plif
        )
        self.per_channel_learnable_decay_full_history_sgr_channelwise_plif = bool(
            per_channel_learnable_decay_full_history_sgr_channelwise_plif
        )
        self.multi_state_full_history_sgr_channelwise_plif = bool(
            multi_state_full_history_sgr_channelwise_plif
        )
        self.midlevel_forgery_margin = bool(midlevel_forgery_margin)
        self.lambda_midlevel_margin = float(lambda_midlevel_margin)
        self.width_scale = float(width_scale)
        self.burst_spike = bool(burst_spike)
        self.burst_k = int(burst_k)
        self.burst_slots = bool(burst_slots)
        self.burst_raw_count = bool(burst_raw_count)
        self.burst_signed_l5 = bool(burst_signed_l5)
        self.burst_signed_all = bool(burst_signed_all)
        self.learnable_l5_thresholds = bool(learnable_l5_thresholds)
        self.learnable_l45_thresholds = bool(learnable_l45_thresholds)
        self.learnable_all_thresholds = bool(learnable_all_thresholds)
        self.l4_tau = None if l4_tau is None else float(l4_tau)
        if burst_k_per_layer is not None:
            if isinstance(burst_k_per_layer, str):
                burst_k_per_layer = [
                    int(k.strip()) for k in burst_k_per_layer.split(",")
                ]
            self.burst_k_per_layer = burst_k_per_layer
        else:
            self.burst_k_per_layer = None
        if burst_thresholds is not None:
            self.burst_thresholds = [float(t) for t in str(burst_thresholds).split(",")]
        else:
            self.burst_thresholds = None
        if burst_thresholds_per_layer is not None:
            self.burst_thresholds_per_layer = [
                [float(t) for t in s.split(",")]
                for s in str(burst_thresholds_per_layer).split(";")
            ]
        else:
            self.burst_thresholds_per_layer = None
        self.clif_temporal = bool(clif_temporal)
        self.rollout_repeated_input = bool(rollout_repeated_input)
        self.readout_mode = snn_readout if readout_mode is None else readout_mode
        self.snn_readout = self.readout_mode
        self.dual_branch_readout = bool(dual_branch_readout)
        self.sim_steps = int(resolved_sim_steps)
        self.event_steps = self.sim_steps
        self.event_threshold = event_threshold

        if self.tau_psn and self.temporal_tau:
            raise ValueError("tau_psn is incompatible with temporal_tau")
        if self.tau_psn and self.temporal_tau_channelwise:
            raise ValueError("tau_psn is incompatible with temporal_tau_channelwise")
        if self.tau_psn and self.clif_tau:
            raise ValueError("tau_psn is incompatible with clif_tau")
        if self.tau_psn and self.clif_temporal:
            raise ValueError("tau_psn is incompatible with clif_temporal")
        if self.tau_psn and self.rollout_repeated_input:
            raise ValueError("tau_psn is incompatible with rollout_repeated_input")
        if self.tau_psn and self.tau_channelwise:
            raise ValueError("tau_psn is incompatible with tau_channelwise")
        if self.tau_psn and self.rctm_channelwise_plif:
            raise ValueError("tau_psn is incompatible with rctm_channelwise_plif")
        if self.tau_psn and self.sgr_channelwise_plif:
            raise ValueError("tau_psn is incompatible with sgr_channelwise_plif")
        if self.tau_psn and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "tau_psn is incompatible with full_history_sgr_channelwise_plif"
            )
        if self.tau_psn and self.full_history_sgr_visible_norm_channelwise_plif:
            raise ValueError(
                "tau_psn is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if self.tau_psn and self.full_history_sgr_learnable_decay_channelwise_plif:
            raise ValueError(
                "tau_psn is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.temporal_tau_channelwise and self.temporal_tau:
            raise ValueError(
                "temporal_tau_channelwise is incompatible with temporal_tau"
            )
        if self.temporal_tau_channelwise and self.clif_tau:
            raise ValueError("temporal_tau_channelwise is incompatible with clif_tau")
        if self.temporal_tau_channelwise and self.clif_temporal:
            raise ValueError(
                "temporal_tau_channelwise is incompatible with clif_temporal"
            )
        if self.temporal_tau_channelwise and self.rollout_repeated_input:
            raise ValueError(
                "temporal_tau_channelwise is incompatible with rollout_repeated_input"
            )
        if self.temporal_tau_channelwise and self.tau_channelwise:
            raise ValueError(
                "temporal_tau_channelwise is incompatible with tau_channelwise"
            )
        if self.temporal_tau_channelwise and self.rctm_channelwise_plif:
            raise ValueError(
                "temporal_tau_channelwise is incompatible with rctm_channelwise_plif"
            )
        if self.temporal_tau_channelwise and self.sgr_channelwise_plif:
            raise ValueError(
                "temporal_tau_channelwise is incompatible with sgr_channelwise_plif"
            )
        if self.temporal_tau_channelwise and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "temporal_tau_channelwise is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.temporal_tau_channelwise
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "temporal_tau_channelwise is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.temporal_tau_channelwise
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "temporal_tau_channelwise is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.temporal_tau and self.clif_tau:
            raise ValueError("temporal_tau is incompatible with clif_tau")
        if self.temporal_tau and self.rollout_repeated_input:
            raise ValueError("temporal_tau is incompatible with rollout_repeated_input")
        if self.temporal_tau and self.tau_channelwise:
            raise ValueError("temporal_tau is incompatible with tau_channelwise")
        if self.temporal_tau and self.rctm_channelwise_plif:
            raise ValueError("temporal_tau is incompatible with rctm_channelwise_plif")
        if self.temporal_tau and self.sgr_channelwise_plif:
            raise ValueError("temporal_tau is incompatible with sgr_channelwise_plif")
        if self.temporal_tau and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "temporal_tau is incompatible with full_history_sgr_channelwise_plif"
            )
        if self.temporal_tau and self.full_history_sgr_visible_norm_channelwise_plif:
            raise ValueError(
                "temporal_tau is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if self.temporal_tau and self.full_history_sgr_learnable_decay_channelwise_plif:
            raise ValueError(
                "temporal_tau is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.clif_temporal and self.clif_tau:
            raise ValueError("clif_temporal is incompatible with clif_tau")
        if self.clif_temporal and self.temporal_tau:
            raise ValueError("clif_temporal is incompatible with temporal_tau")
        if self.clif_temporal and self.rollout_repeated_input:
            raise ValueError(
                "clif_temporal is incompatible with rollout_repeated_input"
            )
        if self.clif_temporal and self.rctm_channelwise_plif:
            raise ValueError("clif_temporal is incompatible with rctm_channelwise_plif")
        if self.clif_temporal and self.sgr_channelwise_plif:
            raise ValueError("clif_temporal is incompatible with sgr_channelwise_plif")
        if self.clif_temporal and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "clif_temporal is incompatible with full_history_sgr_channelwise_plif"
            )
        if self.clif_temporal and self.full_history_sgr_visible_norm_channelwise_plif:
            raise ValueError(
                "clif_temporal is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.clif_temporal
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "clif_temporal is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.clif_temporal and self.tau_channelwise:
            raise ValueError("clif_temporal is incompatible with tau_channelwise")
        if self.clif_tau and self.rctm_channelwise_plif:
            raise ValueError("clif_tau is incompatible with rctm_channelwise_plif")
        if self.clif_tau and self.sgr_channelwise_plif:
            raise ValueError("clif_tau is incompatible with sgr_channelwise_plif")
        if self.clif_tau and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "clif_tau is incompatible with full_history_sgr_channelwise_plif"
            )
        if self.clif_tau and self.full_history_sgr_visible_norm_channelwise_plif:
            raise ValueError(
                "clif_tau is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if self.clif_tau and self.full_history_sgr_learnable_decay_channelwise_plif:
            raise ValueError(
                "clif_tau is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.clif_tau and self.tau_channelwise:
            raise ValueError("clif_tau is incompatible with tau_channelwise")
        if self.rollout_repeated_input and self.tau_channelwise:
            raise ValueError(
                "rollout_repeated_input is incompatible with tau_channelwise"
            )
        if self.tau_channelwise and self.rctm_channelwise_plif:
            raise ValueError(
                "tau_channelwise is incompatible with rctm_channelwise_plif"
            )
        if self.tau_channelwise and self.sgr_channelwise_plif:
            raise ValueError(
                "tau_channelwise is incompatible with sgr_channelwise_plif"
            )
        if self.tau_channelwise and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "tau_channelwise is incompatible with full_history_sgr_channelwise_plif"
            )
        if self.tau_channelwise and self.full_history_sgr_visible_norm_channelwise_plif:
            raise ValueError(
                "tau_channelwise is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.tau_channelwise
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "tau_channelwise is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.rctm_channelwise_plif and self.sgr_channelwise_plif:
            raise ValueError(
                "rctm_channelwise_plif is incompatible with sgr_channelwise_plif"
            )
        if self.rctm_channelwise_plif and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "rctm_channelwise_plif is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.rctm_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "rctm_channelwise_plif is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.rctm_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "rctm_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.rollout_repeated_input and self.sgr_channelwise_plif:
            raise ValueError(
                "rollout_repeated_input is incompatible with sgr_channelwise_plif"
            )
        if self.rollout_repeated_input and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "rollout_repeated_input is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.rollout_repeated_input
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "rollout_repeated_input is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.rollout_repeated_input
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "rollout_repeated_input is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.sgr_channelwise_plif and self.full_history_sgr_channelwise_plif:
            raise ValueError(
                "sgr_channelwise_plif is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.sgr_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "sgr_channelwise_plif is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.sgr_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "sgr_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if (
            self.full_history_sgr_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "full_history_sgr_channelwise_plif is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.full_history_sgr_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "full_history_sgr_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if (
            self.full_history_sgr_visible_norm_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "full_history_sgr_visible_norm_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.st_sgr_full_history_channelwise_plif and self.temporal_tau:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with temporal_tau"
            )
        if self.st_sgr_full_history_channelwise_plif and self.temporal_tau_channelwise:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with temporal_tau_channelwise"
            )
        if self.st_sgr_full_history_channelwise_plif and self.tau_psn:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with tau_psn"
            )
        if self.st_sgr_full_history_channelwise_plif and self.tau_channelwise:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with tau_channelwise"
            )
        if self.st_sgr_full_history_channelwise_plif and self.clif_tau:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with clif_tau"
            )
        if self.st_sgr_full_history_channelwise_plif and self.clif_temporal:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with clif_temporal"
            )
        if self.st_sgr_full_history_channelwise_plif and self.rollout_repeated_input:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with rollout_repeated_input"
            )
        if self.st_sgr_full_history_channelwise_plif and self.rctm_channelwise_plif:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with rctm_channelwise_plif"
            )
        if self.st_sgr_full_history_channelwise_plif and self.sgr_channelwise_plif:
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with sgr_channelwise_plif"
            )
        if (
            self.st_sgr_full_history_channelwise_plif
            and self.full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.st_sgr_full_history_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.st_sgr_full_history_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "st_sgr_full_history_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if self.gated_mode_full_history_sgr_channelwise_plif and self.temporal_tau:
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with temporal_tau"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.temporal_tau_channelwise
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with temporal_tau_channelwise"
            )
        if self.gated_mode_full_history_sgr_channelwise_plif and self.tau_psn:
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with tau_psn"
            )
        if self.gated_mode_full_history_sgr_channelwise_plif and self.tau_channelwise:
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with tau_channelwise"
            )
        if self.gated_mode_full_history_sgr_channelwise_plif and self.clif_tau:
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with clif_tau"
            )
        if self.gated_mode_full_history_sgr_channelwise_plif and self.clif_temporal:
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with clif_temporal"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.rollout_repeated_input
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with rollout_repeated_input"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.rctm_channelwise_plif
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with rctm_channelwise_plif"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.sgr_channelwise_plif
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with sgr_channelwise_plif"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if (
            self.gated_mode_full_history_sgr_channelwise_plif
            and self.st_sgr_full_history_channelwise_plif
        ):
            raise ValueError(
                "gated_mode_full_history_sgr_channelwise_plif is incompatible with st_sgr_full_history_channelwise_plif"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.temporal_tau:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with temporal_tau"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.temporal_tau_channelwise:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with temporal_tau_channelwise"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.tau_psn:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with tau_psn"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.tau_channelwise:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with tau_channelwise"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.clif_tau:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with clif_tau"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.clif_temporal:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with clif_temporal"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.rollout_repeated_input:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with rollout_repeated_input"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.rctm_channelwise_plif:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with rctm_channelwise_plif"
            )
        if self.cc_full_history_sgr_channelwise_plif and self.sgr_channelwise_plif:
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with sgr_channelwise_plif"
            )
        if (
            self.cc_full_history_sgr_channelwise_plif
            and self.full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.cc_full_history_sgr_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.cc_full_history_sgr_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if (
            self.cc_full_history_sgr_channelwise_plif
            and self.st_sgr_full_history_channelwise_plif
        ):
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with st_sgr_full_history_channelwise_plif"
            )
        if (
            self.cc_full_history_sgr_channelwise_plif
            and self.gated_mode_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with gated_mode_full_history_sgr_channelwise_plif"
            )
        if (
            self.cc_full_history_sgr_channelwise_plif
            and self.dt_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "cc_full_history_sgr_channelwise_plif is incompatible with dt_full_history_sgr_channelwise_plif"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.temporal_tau:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with temporal_tau"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.temporal_tau_channelwise:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with temporal_tau_channelwise"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.tau_psn:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with tau_psn"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.tau_channelwise:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with tau_channelwise"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.clif_tau:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with clif_tau"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.clif_temporal:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with clif_temporal"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.rollout_repeated_input:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with rollout_repeated_input"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.rctm_channelwise_plif:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with rctm_channelwise_plif"
            )
        if self.dt_full_history_sgr_channelwise_plif and self.sgr_channelwise_plif:
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with sgr_channelwise_plif"
            )
        if (
            self.dt_full_history_sgr_channelwise_plif
            and self.full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.dt_full_history_sgr_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.dt_full_history_sgr_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if (
            self.dt_full_history_sgr_channelwise_plif
            and self.st_sgr_full_history_channelwise_plif
        ):
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with st_sgr_full_history_channelwise_plif"
            )
        if (
            self.dt_full_history_sgr_channelwise_plif
            and self.gated_mode_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "dt_full_history_sgr_channelwise_plif is incompatible with gated_mode_full_history_sgr_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.temporal_tau
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with temporal_tau"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.temporal_tau_channelwise
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with temporal_tau_channelwise"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.tau_psn
        ):
            raise ValueError("per_channel_learnable_decay is incompatible with tau_psn")
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.tau_channelwise
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with tau_channelwise"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.clif_tau
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with clif_tau"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.clif_temporal
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with clif_temporal"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.rollout_repeated_input
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with rollout_repeated_input"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.rctm_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with rctm_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.sgr_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with sgr_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.st_sgr_full_history_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with st_sgr_full_history_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.gated_mode_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with gated_mode_full_history_sgr_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.cc_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with cc_full_history_sgr_channelwise_plif"
            )
        if (
            self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
            and self.dt_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "per_channel_learnable_decay is incompatible with dt_full_history_sgr_channelwise_plif"
            )
        if self.multi_state_full_history_sgr_channelwise_plif and self.temporal_tau:
            raise ValueError("multi_state is incompatible with temporal_tau")
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.temporal_tau_channelwise
        ):
            raise ValueError(
                "multi_state is incompatible with temporal_tau_channelwise"
            )
        if self.multi_state_full_history_sgr_channelwise_plif and self.tau_psn:
            raise ValueError("multi_state is incompatible with tau_psn")
        if self.multi_state_full_history_sgr_channelwise_plif and self.tau_channelwise:
            raise ValueError("multi_state is incompatible with tau_channelwise")
        if self.multi_state_full_history_sgr_channelwise_plif and self.clif_tau:
            raise ValueError("multi_state is incompatible with clif_tau")
        if self.multi_state_full_history_sgr_channelwise_plif and self.clif_temporal:
            raise ValueError("multi_state is incompatible with clif_temporal")
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.rollout_repeated_input
        ):
            raise ValueError("multi_state is incompatible with rollout_repeated_input")
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.rctm_channelwise_plif
        ):
            raise ValueError("multi_state is incompatible with rctm_channelwise_plif")
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.sgr_channelwise_plif
        ):
            raise ValueError("multi_state is incompatible with sgr_channelwise_plif")
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with full_history_sgr_channelwise_plif"
            )
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.full_history_sgr_visible_norm_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with full_history_sgr_visible_norm_channelwise_plif"
            )
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.full_history_sgr_learnable_decay_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with full_history_sgr_learnable_decay_channelwise_plif"
            )
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.st_sgr_full_history_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with st_sgr_full_history_channelwise_plif"
            )
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.gated_mode_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with gated_mode_full_history_sgr_channelwise_plif"
            )
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.cc_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with cc_full_history_sgr_channelwise_plif"
            )
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.dt_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with dt_full_history_sgr_channelwise_plif"
            )
        if (
            self.multi_state_full_history_sgr_channelwise_plif
            and self.per_channel_learnable_decay_full_history_sgr_channelwise_plif
        ):
            raise ValueError(
                "multi_state is incompatible with per_channel_learnable_decay_full_history_sgr_channelwise_plif"
            )
        if self.rollout_repeated_input and self.rctm_channelwise_plif:
            raise ValueError(
                "rollout_repeated_input is incompatible with rctm_channelwise_plif"
            )

        self.LAI = LossAugmentedInferenceAP(
            n_shot_g, n_shot_f, self.epsilon, self.positive_update
        )
        self.register_buffer(
            "smoothCElossMask",
            torch.zeros(n_task * (1 + n_shot_g + n_shot_f), dtype=torch.float32),
        )

        w = self.width_scale
        channel_layout = tuple(int(c * w) for c in (64, 64, 128, 128, 256, 256))

        def build_spike_neuron(channels):
            if self.comparison_neuron == "mlf-k3":
                return MLFK3ComparisonNode(channels=channels)
            if self.comparison_neuron == "ilif":
                return ILIFComparisonNode(
                    channels=channels, tau=self.lif_tau, v_threshold=self.lif_v_threshold
                )
            if self.comparison_neuron == "klif":
                return KLIFComparisonNode(
                    channels=channels, tau=self.lif_tau, v_threshold=self.lif_v_threshold
                )
            if self.comparison_neuron == "scalar-plif":
                return ScalarPLIFComparisonNode(
                    tau=self.lif_tau, v_threshold=self.lif_v_threshold
                )
            if self.comparison_neuron == "sliding-psn-k2":
                if self.comparison_psn_order != 2:
                    raise ValueError("sliding-psn-k2 requires comparison_psn_order=2")
                return SlidingPSNComparisonNode(
                    channels=channels,
                    order=2,
                )
            if self.tau_psn:
                return PLIFTauPSNNode(
                    tau=self.lif_tau, v_threshold=self.lif_v_threshold
                )
            if self.tau_channelwise:
                return PLIFTauChannelwiseNode(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.sgr_channelwise_plif:
                return SGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.full_history_sgr_channelwise_plif:
                return FullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.full_history_sgr_visible_norm_channelwise_plif:
                return VisibleNormalizedFullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.full_history_sgr_learnable_decay_channelwise_plif:
                return LearnableDecayFullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.st_sgr_full_history_channelwise_plif:
                return STSGRFullHistoryChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.gated_mode_full_history_sgr_channelwise_plif:
                return GatedModeFullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.cc_full_history_sgr_channelwise_plif:
                return CrossChannelFullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.dt_full_history_sgr_channelwise_plif:
                return DoubleTimescaleFullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.per_channel_learnable_decay_full_history_sgr_channelwise_plif:
                return PerChannelLearnableDecayFullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.multi_state_full_history_sgr_channelwise_plif:
                return MultiStateFullHistorySGRChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.rctm_channelwise_plif:
                return ResidualCausalTemporalMixChannelwisePLIF(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.clif_temporal:
                return PLIFCLIFTemporalNode(
                    tau=self.lif_tau, v_threshold=self.lif_v_threshold
                )
            if self.temporal_tau_channelwise:
                return PLIFTauTemporalChannelwiseNode(
                    channels=channels,
                    tau=self.lif_tau,
                    v_threshold=self.lif_v_threshold,
                )
            if self.temporal_tau:
                return PLIFTauTemporalNode(
                    tau=self.lif_tau, v_threshold=self.lif_v_threshold
                )
            if self.clif_tau:
                return PLIFCLIFTauNode(
                    tau=self.lif_tau, v_threshold=self.lif_v_threshold
                )
            return PLIFTauNode(tau=self.lif_tau, v_threshold=self.lif_v_threshold)

        self.spike_neurons = nn.ModuleList(
            [build_spike_neuron(channels) for channels in channel_layout]
        )
        l5_burst_index = len(channel_layout) - 1
        if self.burst_spike and self.comparison_neuron is None:
            signed_l5_applied = False
            for idx, sn in enumerate(self.spike_neurons):
                is_wrapped = hasattr(sn, "neuron") and isinstance(
                    sn.neuron, PLIFTauChannelwiseNode
                )
                is_bare = isinstance(sn, PLIFTauChannelwiseNode)
                if is_wrapped or is_bare:
                    old = sn.neuron if is_wrapped else sn
                    cls = (
                        BinaryBurstSlotsPLIFTauChannelwiseNode
                        if self.burst_slots
                        else SignedBurstPLIFTauChannelwiseNode
                        if (self.burst_signed_l5 and idx == l5_burst_index)
                        or self.burst_signed_all
                        else BurstPLIFTauChannelwiseNode
                    )
                    if cls is SignedBurstPLIFTauChannelwiseNode:
                        signed_l5_applied = True
                    k = self.burst_k
                    if self.burst_k_per_layer is not None and idx < len(
                        self.burst_k_per_layer
                    ):
                        k = self.burst_k_per_layer[idx]
                    layer_tau = (
                        self.l4_tau if idx == 4 and self.l4_tau is not None else 2.0
                    )
                    kw = dict(
                        channels=old.channels,
                        tau=layer_tau,
                        v_threshold=old.v_threshold,
                        burst_k=k,
                    )
                    if not self.burst_slots:
                        kw["normalize"] = not self.burst_raw_count
                    if self.burst_thresholds is not None:
                        kw["thresholds"] = self.burst_thresholds
                    elif self.burst_thresholds_per_layer is not None and idx < len(
                        self.burst_thresholds_per_layer
                    ):
                        kw["thresholds"] = self.burst_thresholds_per_layer[idx]
                    new_burst = cls(**kw)
                    with torch.no_grad():
                        if layer_tau != 2.0:
                            new_burst.tau_raw.copy_(torch.tensor(layer_tau))
                        elif hasattr(old, "tau_raw"):
                            new_burst.tau_raw.copy_(old.tau_raw)
                    if is_wrapped:
                        sn.neuron = new_burst
                    else:
                        self.spike_neurons[idx] = new_burst
            if self.burst_signed_l5 and not signed_l5_applied:
                raise ValueError(
                    "burst_signed_l5 requires the final spike layer to expose a "
                    "PLIFTauChannelwiseNode-backed burst-capable neuron"
                )

        def replace_burst_with_learnable(layer_idx, flag_name):
            layer = self.spike_neurons[layer_idx]
            is_wrapped = hasattr(layer, "neuron") and isinstance(
                layer.neuron, BurstPLIFTauChannelwiseNode
            )
            is_bare = isinstance(layer, BurstPLIFTauChannelwiseNode)
            if not is_wrapped and not is_bare:
                raise ValueError(
                    "%s requires spike layer %d to be a BurstPLIFTauChannelwiseNode"
                    % (flag_name, layer_idx)
                )
            old = layer.neuron if is_wrapped else layer
            learnable = LearnableThresholdBurstPLIFTauChannelwiseNode(
                channels=old.channels,
                tau=2.0,
                v_threshold=old.v_threshold,
                burst_k=old.burst_k,
                thresholds=old.thresholds,
            )
            with torch.no_grad():
                if hasattr(old, "tau_raw"):
                    learnable.tau_raw.copy_(old.tau_raw)
            if is_wrapped:
                layer.neuron = learnable
            else:
                self.spike_neurons[layer_idx] = learnable

        if self.burst_spike and self.learnable_all_thresholds:
            for layer_idx in range(len(self.spike_neurons)):
                replace_burst_with_learnable(layer_idx, "learnable_all_thresholds")
        elif self.burst_spike and self.learnable_l45_thresholds:
            for layer_idx in [l5_burst_index - 1, l5_burst_index]:
                replace_burst_with_learnable(layer_idx, "learnable_l45_thresholds")
        elif self.burst_spike and self.learnable_l5_thresholds:
            replace_burst_with_learnable(l5_burst_index, "learnable_l5_thresholds")
        # 第一阶段：提取浅层局部运动模式，通道数由 n_in（正式配置为 12）提升到 64w。
        # 输入形状：[B, n_in, T]；输出形状：[B, 64w, ceil(T/2)]。
        self.convNet = nn.Sequential(
            # 第一层一维卷积：较大的 7 点时间窗口，padding=3 保持卷积前后的时间长度不变。
            nn.Conv1d(
                n_in, int(64 * w), kernel_size=7, padding=3, stride=1, dilation=1
            ),
            # 第一次时间下采样：长度约减半；ceil_mode=True 使奇数长度采用向上取整。
            nn.MaxPool1d(2, 2, ceil_mode=True),
            # 第 1 个脉冲神经元层：正式 FedBurst 配置为 64w 通道、K=2、阈值 [1, 2]。
            self.spike_neurons[0],
            # 第二层一维卷积：保持 64w 通道和当前时间长度，进一步提取浅层时序特征。
            nn.Conv1d(
                int(64 * w), int(64 * w), kernel_size=3, padding=1, stride=1, dilation=1
            ),
            # 第 2 个脉冲神经元层：正式配置为 64w 通道、K=2、阈值 [1, 2]。
            self.spike_neurons[1],
        )

        # 第二阶段：将浅层特征扩展到 128w 通道，并形成供后续多尺度融合使用的中层特征。
        # 输入形状：[B, 64w, ceil(T/2)]；输出形状：[B, 128w, ceil(T/4)]。
        self.convNet2 = nn.Sequential(
            # 第三层一维卷积：通道数从 64w 增加到 128w，时间长度保持不变。
            nn.Conv1d(
                int(64 * w),
                int(128 * w),
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
            # 第二次时间下采样：将时间分辨率从约 T/2 降至约 T/4。
            nn.MaxPool1d(2, 2, ceil_mode=True),
            # 第 3 个脉冲神经元层：正式配置为 128w 通道、K=2、阈值 [1, 2]。
            self.spike_neurons[2],
            # 第四层一维卷积：保持 128w 通道和约 T/4 的时间分辨率。
            nn.Conv1d(
                int(128 * w),
                int(128 * w),
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
            # 第 4 个脉冲神经元层：正式配置为 128w 通道、K=5、阈值 [1, 2, 3, 5, 7]。
            self.spike_neurons[3],
        )

        # 第三阶段：提取高层签名动态，将通道数扩展到 256w，并输出最深层时序特征。
        # 输入形状：[B, 128w, ceil(T/4)]；输出形状：[B, 256w, ceil(T/8)]。
        self.convNet3 = nn.Sequential(
            # 第五层一维卷积：通道数从 128w 增加到 256w，时间长度保持不变。
            nn.Conv1d(
                int(128 * w),
                int(256 * w),
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
            # 第三次时间下采样：将时间分辨率从约 T/4 降至约 T/8。
            nn.MaxPool1d(2, 2, ceil_mode=True),
            # 第 5 个脉冲神经元层：正式配置为 256w 通道、K=8，使用固定非线性阈值。
            # 阈值布局：[1, 2, 3, 4, 6, 8, 10, 12]。
            self.spike_neurons[4],
            # 第六层一维卷积：保持 256w 通道和约 T/8 的时间分辨率。
            nn.Conv1d(
                int(256 * w),
                int(256 * w),
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
            # 第 6 个脉冲神经元层：正式配置为 256w 通道、K=10。
            # 初始阈值为 [1, 2, 3, 4, 5, 7, 9, 11, 13, 15]；仅此层学习有序阈值间隔。
            self.spike_neurons[5],
        )
        self.Upsample = nn.Upsample(scale_factor=2)
        self.convNet2_1x1 = nn.Conv1d(
            int(128 * w), int(256 * w), kernel_size=1, padding=0, stride=1
        )
        pool_dim = int(256 * w)
        self.sp2 = selectivePooling(pool_dim, head_dim=32, num_heads=16)
        self.sp3 = selectivePooling(pool_dim, head_dim=32, num_heads=16)
        self.salient_readout_proj = None
        self.last_branch_diagnostics = None
        if self.dual_branch_readout:
            self.salient_readout_proj = nn.Linear(
                pool_dim * 2, self.embedding_dim, bias=False
            )
        self.bn = nn.BatchNorm1d(1024, affine=False, momentum=0.001)
        self.cls = nn.Linear(1024, n_classes, bias=False)
        self.classifier = self.cls

        nn.init.kaiming_normal_(self.convNet[0].weight, a=1)
        nn.init.kaiming_normal_(self.convNet[3].weight, a=1)
        nn.init.kaiming_normal_(self.convNet2[0].weight, a=1)
        nn.init.kaiming_normal_(self.convNet2[3].weight, a=1)
        nn.init.kaiming_normal_(self.convNet3[0].weight, a=1)
        nn.init.kaiming_normal_(self.convNet3[3].weight, a=1)
        nn.init.kaiming_normal_(self.convNet2_1x1.weight, a=0)
        if self.salient_readout_proj is not None:
            nn.init.kaiming_normal_(self.salient_readout_proj.weight, a=1)
        nn.init.kaiming_normal_(self.cls.weight, a=1)
        nn.init.zeros_(self.convNet[0].bias)
        nn.init.zeros_(self.convNet[3].bias)
        nn.init.zeros_(self.convNet2[0].bias)
        nn.init.zeros_(self.convNet2[3].bias)
        nn.init.zeros_(self.convNet3[0].bias)
        nn.init.zeros_(self.convNet3[3].bias)
        nn.init.zeros_(self.convNet2_1x1.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for key, param in state_dict.items():
            if not key.endswith("time_weight"):
                continue
            target_size = param.size(0)
            module_path = key[: -len(".time_weight")]
            try:
                target_module = self.get_submodule(module_path)
            except AttributeError:
                continue
            if isinstance(target_module, PLIFTauPSNNode):
                current = target_module.time_weight.size(0)
                if current < target_size:
                    target_module._expand_time_weight(target_size)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def tau_diversity_regularization(self):
        tau_vars = []
        for spike_neuron in self.spike_neurons:
            taus = spike_neuron.effective_taus()
            tau_vars.append(taus.var(unbiased=False))
        return torch.stack(tau_vars).mean()

    def _get_neuron(self, spike_neuron):
        if hasattr(spike_neuron, "neuron"):
            return spike_neuron.neuron
        return spike_neuron

    def collect_mp_stats(self):
        stats = {}
        for i, sn in enumerate(self.spike_neurons):
            neuron = self._get_neuron(sn)
            v = neuron.v if hasattr(neuron, "v") else None
            if v is None:
                continue
            stats["L%d_mean" % i] = v.mean().detach()
            stats["L%d_std" % i] = v.std(unbiased=False).detach()
        return stats

    def mp_alignment_loss(self, targets):
        loss = torch.tensor(0.0, device=self.cls.weight.device)
        stats = self.collect_mp_stats()
        for i in range(len(self.spike_neurons)):
            mean_key = "L%d_mean" % i
            std_key = "L%d_std" % i
            if mean_key in stats and mean_key in targets and std_key in targets:
                local_mean = stats[mean_key]
                target_mean = torch.as_tensor(
                    targets[mean_key], device=local_mean.device, dtype=local_mean.dtype
                )
                target_std = torch.as_tensor(
                    targets[std_key], device=local_mean.device, dtype=local_mean.dtype
                )
                delta = (local_mean - target_mean) ** 2
                denom = target_std**2 + 1e-8
                loss = loss + delta / denom
        return loss

    def collect_tau_values(self):
        tau_parts = []
        for sn in self.spike_neurons:
            neuron = self._get_neuron(sn)
            if hasattr(neuron, "tau_raw"):
                tau_parts.append(neuron.tau_raw.view(-1))
        if not tau_parts:
            return torch.empty(0)
        return torch.cat(tau_parts)

    def tau_alignment_loss(self, tau_consensus):
        tau_local = self.collect_tau_values()
        if tau_local.numel() == 0:
            return tau_local.new_tensor(0.0)
        consensus = tau_consensus.to(device=tau_local.device, dtype=tau_local.dtype)
        return ((tau_local - consensus) ** 2).mean()

    def collect_l5_thresholds(self):
        if not (
            self.learnable_l5_thresholds
            or self.learnable_l45_thresholds
            or self.learnable_all_thresholds
        ):
            return torch.empty(0)
        l5_sn = self.spike_neurons[-1]
        neuron = self._get_neuron(l5_sn)
        if hasattr(neuron, "log_spacings"):
            return neuron.log_spacings.view(-1)
        return torch.empty(0)

    def threshold_alignment_loss(self, log_spacings_consensus):
        local = self.collect_l5_thresholds()
        if local.numel() == 0:
            return local.new_tensor(0.0)
        consensus = log_spacings_consensus.to(device=local.device, dtype=local.dtype)
        return ((local - consensus) ** 2).mean()

    def getOutputMask(self, lens):
        if torch.is_tensor(lens):
            lens = lens.detach().cpu().tolist()

        output_lens = numpy.array(lens, dtype=numpy.int32)
        for _ in range(2):
            output_lens = (output_lens + 1) // 2
        mask = numpy.zeros(
            (len(output_lens), numpy.max(output_lens)), dtype=numpy.float32
        )
        for idx, out_len in enumerate(output_lens):
            mask[idx, :out_len] = 1.0
        return mask

    def _empty_forward_outputs(self, signal):
        batch_size = signal.size(0)
        self.last_branch_diagnostics = None
        embedding = signal.new_zeros(batch_size, 1024)
        if not (self.training and batch_size == 1):
            embedding = self.bn(embedding)
        logits = self.cls(embedding)
        if self.midlevel_forgery_margin:
            return self._format_forward_outputs(
                embedding,
                logits,
                late_feature=signal.new_zeros(batch_size, self.embedding_dim // 2),
            )
        return self._format_forward_outputs(embedding, logits)

    def _format_forward_outputs(self, embedding, logits, late_feature=None):
        if self.midlevel_forgery_margin:
            return embedding, logits, late_feature
        return embedding, logits

    def _resample_mask(self, mask, target_steps):
        if mask.dim() != 2:
            raise ValueError("mask must have shape (batch, time)")

        adjusted_mask = mask
        while adjusted_mask.size(1) > target_steps:
            pooled_mask = adjusted_mask
            if pooled_mask.size(1) % 2 == 1:
                pad = pooled_mask.new_zeros(pooled_mask.size(0), 1)
                pooled_mask = torch.cat([pooled_mask, pad], dim=1)
            adjusted_mask = pooled_mask.view(pooled_mask.size(0), -1, 2).max(dim=2)[0]

        while adjusted_mask.size(1) < target_steps:
            adjusted_mask = adjusted_mask.repeat_interleave(2, dim=1)
            if adjusted_mask.size(1) > target_steps:
                adjusted_mask = adjusted_mask[:, :target_steps]

        if adjusted_mask.size(1) != target_steps:
            raise ValueError(
                "mask shape is incompatible with the pooled temporal resolution"
            )
        return adjusted_mask

    def _mask_from_lens(self, signal, lens):
        if torch.is_tensor(lens):
            lens = lens.detach().cpu().tolist()

        raw_mask = signal.new_zeros(signal.size(0), signal.size(2))
        for idx, length in enumerate(lens):
            valid_steps = max(0, min(int(length), signal.size(2)))
            raw_mask[idx, :valid_steps] = 1.0
        return raw_mask

    def _resolve_masks(self, signal, mask=None, lens=None):
        raw_mask = self._mask_from_lens(signal, lens) if lens is not None else None
        resolved_mask = mask

        if resolved_mask is not None:
            resolved_mask = torch.as_tensor(
                resolved_mask, dtype=signal.dtype, device=signal.device
            )
            if resolved_mask.dim() == 3 and resolved_mask.size(1) == 1:
                resolved_mask = resolved_mask.squeeze(1)
            if resolved_mask.dim() != 2:
                raise ValueError("mask must have shape (batch, time)")
            if resolved_mask.size(0) != signal.size(0):
                raise ValueError(
                    "mask batch dimension does not match sig batch dimension"
                )
            if resolved_mask.size(1) == signal.size(2):
                raw_mask = resolved_mask
            elif raw_mask is None:
                raw_mask = self._resample_mask(resolved_mask, signal.size(2))

        if resolved_mask is None:
            resolved_mask = raw_mask

        if resolved_mask is None:
            return None, None

        if not torch.is_tensor(resolved_mask):
            resolved_mask = torch.as_tensor(
                resolved_mask, dtype=signal.dtype, device=signal.device
            )
        if resolved_mask.dim() == 3 and resolved_mask.size(1) == 1:
            resolved_mask = resolved_mask.squeeze(1)
        if resolved_mask.dim() != 2:
            raise ValueError("mask must have shape (batch, time)")
        if resolved_mask.size(0) != signal.size(0):
            raise ValueError("mask batch dimension does not match sig batch dimension")

        return raw_mask, resolved_mask

    def _resolve_pool_mask(self, signal, target_steps, mask=None, lens=None):
        _, resolved_mask = self._resolve_masks(signal, mask=mask, lens=lens)

        if mask is None and lens is not None:
            return torch.as_tensor(
                self.getOutputMask(lens), dtype=signal.dtype, device=signal.device
            )

        if resolved_mask is None:
            return signal.new_ones(signal.size(0), target_steps)
        if resolved_mask.size(1) != target_steps:
            resolved_mask = self._resample_mask(resolved_mask, target_steps)
        return resolved_mask.to(dtype=signal.dtype, device=signal.device)

    def _time_readout(self, seq, mode=None):
        if seq.dim() != 4:
            raise ValueError("seq must have shape (time, batch, channels, length)")

        readout_mode = self.readout_mode if mode is None else mode
        if readout_mode == "rate":
            return seq.mean(dim=0)
        if readout_mode == "last":
            return seq[-1]
        raise ValueError("readout mode must be 'rate' or 'last'")

    def _masked_salient_readout(self, seq, mask):
        if seq.dim() != 3:
            raise ValueError("seq must have shape (batch, channels, time)")
        if mask.dim() != 2:
            raise ValueError("mask must have shape (batch, time)")
        if seq.size(0) != mask.size(0) or seq.size(2) != mask.size(1):
            raise ValueError("mask shape must match sequence batch and time")

        valid = mask.to(dtype=torch.bool, device=seq.device).unsqueeze(1)
        masked_seq = seq.masked_fill(~valid, torch.finfo(seq.dtype).min)
        salient = masked_seq.max(dim=2).values
        has_valid = valid.any(dim=2)
        return torch.where(has_valid, salient, torch.zeros_like(salient))

    def _compute_readout_branches(self, output2, output3, mask2, mask3):
        global_branch = torch.cat(
            [self.sp2(output2, mask2), self.sp3(output3, mask3)], dim=1
        )
        if not self.dual_branch_readout:
            return global_branch, None
        salient_branch = torch.cat(
            [
                self._masked_salient_readout(output2, mask2),
                self._masked_salient_readout(output3, mask3),
            ],
            dim=1,
        )
        return global_branch, salient_branch

    def _fuse_readout_branches(self, global_branch, salient_branch):
        if not self.dual_branch_readout:
            return global_branch
        if salient_branch is None:
            raise ValueError("dual_branch_readout requires a salient_branch")
        projected_salient_branch = self.salient_readout_proj(salient_branch)
        return global_branch + projected_salient_branch

    def _compute_branch_diagnostics(self, global_branch, salient_branch):
        if not self.dual_branch_readout or salient_branch is None:
            return None

        projected_salient_branch = self.salient_readout_proj(salient_branch)
        global_branch_norm = torch.norm(global_branch.detach(), dim=1)
        fused_embedding_delta_norm = torch.norm(
            projected_salient_branch.detach(), dim=1
        )
        global_salient_cosine = F.cosine_similarity(
            global_branch.detach(), projected_salient_branch.detach(), dim=1, eps=1e-8
        )
        return {
            "global_branch_norm": global_branch_norm,
            "fused_embedding_delta_norm": fused_embedding_delta_norm,
            "global_salient_cosine": global_salient_cosine,
        }

    def smoothCEloss(self, logit, target, eps=0.1):
        n_class = logit.size(1)
        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)
        return -(one_hot * log_prb).sum(dim=1).mean(dim=0)

    def APLoss_DLM(self, x):
        score_std = x.new_tensor(0.0)
        score_aug = x.new_tensor(0.0)
        mAP = x.new_tensor(0.0)

        for anchor, pos, neg in self._iter_episode_triplets(x):
            la = torch.norm(anchor) + 1e-8
            lp = torch.norm(pos, dim=1) + 1e-8
            ln = torch.norm(neg, dim=1) + 1e-8
            scoreAP = torch.sum(anchor.unsqueeze(0) * pos, dim=1) / (la * lp)
            scoreAN = torch.sum(anchor.unsqueeze(0) * neg, dim=1) / (la * ln)
            scoreAP, _ = torch.sort(scoreAP, descending=True)
            scoreAN, _ = torch.sort(scoreAN, descending=True)
            simDiff = scoreAP.unsqueeze(1) - scoreAN.unsqueeze(0)
            Y_std = torch.sign(simDiff)
            score_std = score_std + torch.mean(Y_std * simDiff)

            direction = self.LAI(scoreAP.detach().cpu(), scoreAN.detach().cpu())
            mAP = mAP + self.LAI.AP(direction).to(device=x.device, dtype=x.dtype)
            Y_aug = (-direction[1:, 1:]).to(device=x.device, dtype=x.dtype)
            score_aug = score_aug + torch.mean(Y_aug * simDiff)

        score_std = score_std / self.n_task
        score_aug = score_aug / self.n_task
        mAP = mAP / self.n_task
        loss = 1 / self.epsilon * (self.APAlpha * score_aug - score_std)
        if not self.positive_update:
            loss = -loss
        return loss, mAP

    def _iter_episode_triplets(self, feature):
        if feature.dim() != 2:
            raise ValueError("feature must have shape (batch, feature_dim)")

        step = 1 + self.n_shot_g + self.n_shot_f
        expected = self.n_task * step
        if step <= 1 or self.n_task <= 0:
            return
        if feature.size(0) != expected:
            raise ValueError(
                "feature batch does not match episode layout: expected %d, got %d"
                % (expected, feature.size(0))
            )

        for i in range(self.n_task):
            start = i * step
            anchor = feature[start]
            pos = feature[start + 1 : start + 1 + self.n_shot_g]
            neg = feature[start + 1 + self.n_shot_g : start + step]
            yield anchor, pos, neg

    def midlevel_forgery_margin_loss(self, late_feature, margin=0.5):
        if self.n_task <= 0:
            return late_feature.new_tensor(0.0)

        losses = []
        for anchor, pos, neg in self._iter_episode_triplets(late_feature):
            pos_dist = torch.norm(pos - anchor.unsqueeze(0), dim=1)
            neg_dist = torch.norm(neg - anchor.unsqueeze(0), dim=1)

            tighten_loss = (
                pos_dist.mean()
                if pos_dist.numel() > 0
                else late_feature.new_tensor(0.0)
            )
            separate_loss = (
                F.relu(tighten_loss + float(margin) - neg_dist.mean())
                if neg_dist.numel() > 0
                else late_feature.new_tensor(0.0)
            )
            losses.append(tighten_loss + separate_loss)

        return torch.stack(losses).mean()

    def _forward_backbone_once(self, signal):
        output = self.convNet(signal)
        output2 = self.convNet2(output)
        output3 = self.convNet3(output2)
        return output, output2, output3

    def _forward_backbone_repeated_input(self, signal):
        outputs = []
        outputs2 = []
        outputs3 = []

        for _ in range(self.sim_steps):
            output, output2, output3 = self._forward_backbone_once(signal)
            outputs.append(output)
            outputs2.append(output2)
            outputs3.append(output3)

        return (
            self._time_readout(torch.stack(outputs, dim=0), mode="rate"),
            self._time_readout(torch.stack(outputs2, dim=0), mode="rate"),
            self._time_readout(torch.stack(outputs3, dim=0), mode="rate"),
        )

    def forward(self, sig, mask=None, lens=None):
        # 每个签名批次开始前清空所有有状态脉冲神经元的膜电位，避免前一批次泄漏到当前批次。
        functional.reset_net(self)

        # 输入必须是三维张量，允许 [批量, 时间, 特征] 或 [批量, 特征, 时间] 两种布局。
        if sig.dim() != 3:
            raise ValueError(
                "sig must have shape (batch, time, features) or (batch, features, time)"
            )

        # 通过哪一维等于输入特征数 n_in（正式方法中为 12）来判断输入布局。
        feature_on_dim1 = sig.size(1) == self.n_in
        feature_on_dim2 = sig.size(2) == self.n_in
        # 若时间长度也恰好等于 n_in，则无法可靠判断哪一维是特征维，直接拒绝歧义输入。
        if feature_on_dim1 and feature_on_dim2:
            raise ValueError(
                "ambiguous sig shape: cannot disambiguate feature and time dimensions"
            )
        # 输入已是 Conv1d 所需的 [B, C, T] 格式时，无需转置。
        if feature_on_dim1:
            signal = sig
        # 常见数据加载格式为 [B, T, C]，交换时间维和特征维得到 [B, C, T]。
        elif feature_on_dim2:
            signal = sig.transpose(1, 2)
        # 两个候选维度均不等于 n_in，说明输入特征数与模型配置不一致。
        else:
            raise ValueError("sig feature dimension does not match n_in")

        # 对长度为 0 的极端序列返回形状正确的全零嵌入和分类结果。
        if signal.size(2) == 0:
            return self._empty_forward_outputs(signal)

        # 可选模式：将同一输入重复送入有状态主干若干次，再对多次响应做速率平均。
        if self.rollout_repeated_input:
            output, output2, output3 = self._forward_backbone_repeated_input(signal)
        # 正式 FedBurst-Sig 使用单次六层前向，得到三个不同时间分辨率的阶段特征。
        else:
            output, output2, output3 = self._forward_backbone_once(signal)

        # 将深层特征 output3 上采样到中层时间分辨率，并裁剪到与 output2 完全等长。
        # 同时用 1×1 卷积把 output2 从 128 通道投影到 256 通道，再执行残差相加和 SELU。
        output2 = F.selu(
            self.Upsample(output3)[:, :, : output2.shape[2]]
            + self.convNet2_1x1(output2),
            inplace=True,
        )
        # 根据原始长度或传入 mask，生成与融合特征 output2 时间长度一致的有效位置掩码。
        mask2 = self._resolve_pool_mask(signal, output2.size(2), mask=mask, lens=lens)
        # 保证 mask2 与特征使用相同的数据类型和计算设备。
        mask2 = mask2.to(dtype=output2.dtype, device=output2.device)
        # 再下采样一次 mask2，得到与更深层 output3 对应的有效位置掩码 mask3。
        mask3 = F.max_pool1d(mask2.unsqueeze(1), 2, ceil_mode=True).squeeze(1)
        # 奇数长度和上取整池化可能造成一位偏差，必要时重新采样以严格匹配 output3。
        if mask3.size(1) != output3.size(2):
            mask3 = self._resample_mask(mask3, output3.size(2))

        # 分别对融合特征和深层特征进行 mask-aware selective pooling，并拼接成全局读出。
        # 正式配置下 global_branch 为 512 + 512 = 1024 维，salient_branch 默认为 None。
        global_branch, salient_branch = self._compute_readout_branches(
            output2,
            output3,
            mask2,
            mask3,
        )
        # 若启用了可选双分支读出，记录两分支范数及余弦关系；正式配置下结果为 None。
        self.last_branch_diagnostics = self._compute_branch_diagnostics(
            global_branch, salient_branch
        )
        # 正式配置直接使用 global_branch；可选双分支模式会叠加投影后的显著性分支。
        embedding = self._fuse_readout_branches(global_branch, salient_branch)
        # 对 1024 维签名表示做无仿射 BatchNorm，得到最终签名嵌入。
        embedding = self.bn(embedding)
        # 将嵌入送入无偏置的本地 writer 分类头；该分类头主要服务于训练监督。
        logits = self.cls(embedding)
        # 可选的中层伪造间隔损失需要额外返回深层选择性池化特征。
        if self.midlevel_forgery_margin:
            feat3 = self.sp3(output3, mask3)
            return self._format_forward_outputs(embedding, logits, late_feature=feat3)
        # 正式路径返回 (1024 维签名嵌入, writer 分类 logits)。
        return self._format_forward_outputs(embedding, logits)

    def collect_spike_rate_histograms(self, n_bins=16, layers=None):
        hists = {}
        for i, sn in enumerate(self.spike_neurons):
            if layers is not None and i not in layers:
                continue
            neuron = sn.neuron if hasattr(sn, "neuron") else sn
            v = getattr(neuron, "v", None)
            if v is None:
                continue
            K = getattr(neuron, "burst_k", 1)
            max_val = float(max(K, 1))
            vals = v.detach().cpu().numpy().flatten()
            hist, _ = numpy.histogram(vals, bins=n_bins, range=(0, max_val))
            hists["L%d" % i] = hist.astype(numpy.float32) / max(hist.sum(), 1)
        return hists

    def spike_rate_distillation_loss(self, consensus_hists):
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        matched = 0
        for index, spike_neuron in enumerate(self.spike_neurons):
            key = "L%d" % index
            if key not in consensus_hists:
                continue
            neuron = spike_neuron.neuron if hasattr(spike_neuron, "neuron") else spike_neuron
            membrane = getattr(neuron, "v", None)
            if membrane is None or not torch.is_tensor(membrane):
                continue
            target_histogram = torch.tensor(
                numpy.array(consensus_hists[key]),
                dtype=torch.float32,
                device=loss.device,
            )
            target_histogram = target_histogram / target_histogram.sum().clamp_min(1e-8)
            bin_rates = torch.linspace(
                0.0, 1.0, target_histogram.numel(), device=loss.device
            )
            target_rate = (target_histogram * bin_rates).sum()
            local_rate = torch.sigmoid(membrane).mean()
            loss = loss + (local_rate - target_rate) ** 2
            matched += 1
        return loss / max(matched, 1)

    def collect_burst_count_histograms(self, n_bins=None):
        hists = {}
        for i, sn in enumerate(self.spike_neurons):
            neuron = sn.neuron if hasattr(sn, "neuron") else sn
            v = getattr(neuron, "v", None)
            if v is None:
                continue
            K = getattr(neuron, "burst_k", 1)
            if n_bins is None:
                n_bins = K + 1
            vals = v.detach().cpu().numpy().flatten()
            hist, _ = numpy.histogram(vals, bins=n_bins, range=(0, float(K) + 0.5))
            hists["L%d" % i] = hist.astype(numpy.float32) / max(hist.sum(), 1)
        return hists

    def burst_count_distillation_loss(self, consensus_hists):
        local_hists = self.collect_burst_count_histograms()
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for key in local_hists:
            if key not in consensus_hists:
                continue
            p = torch.tensor(numpy.array(local_hists[key]) + 1e-8)
            q = torch.tensor(numpy.array(consensus_hists[key]) + 1e-8)
            p = p / p.sum()
            q = q / q.sum()
            loss = loss + (p * (p.log() - q.log())).sum()
        return loss

    def collect_v_stats(self):
        stats = {}
        for i, sn in enumerate(self.spike_neurons):
            neuron = sn.neuron if hasattr(sn, "neuron") else sn
            v = getattr(neuron, "v", None)
            if v is None:
                continue
            vnp = v.detach().cpu().numpy().flatten()
            stats["L%d_mean" % i] = float(numpy.mean(vnp))
            stats["L%d_std" % i] = float(numpy.std(vnp) + 1e-6)
        return stats

    def reset_v_to_consensus(self, consensus_stats):
        for i, sn in enumerate(self.spike_neurons):
            neuron = sn.neuron if hasattr(sn, "neuron") else sn
            v = getattr(neuron, "v", None)
            if v is None:
                continue
            loc_mean = float(v.detach().mean())
            loc_std = float(v.detach().std() + 1e-6)
            tgt_mean = consensus_stats.get("L%d_mean" % i, loc_mean)
            tgt_std = consensus_stats.get("L%d_std" % i, loc_std)
            v_new = (v - loc_mean) / loc_std * tgt_std + tgt_mean
            neuron.v = v_new.detach()

    def prototype_alignment_loss(self, global_proto):
        if not hasattr(self, "_last_embedding"):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        local_proto = self._last_embedding.mean(dim=0)
        g = torch.tensor(
            global_proto, dtype=local_proto.dtype, device=local_proto.device
        )
        return ((local_proto - g) ** 2).mean()

    def collect_logit_distribution(self, sig, mask, lens):
        with torch.no_grad():
            _, logits = self(sig, mask, lens)
            probs = torch.softmax(logits, dim=1)
            return probs.mean(dim=0).cpu().numpy().tolist()

    def collect_feature_stats(self, sig, mask, lens):
        with torch.no_grad():
            _ = self(sig, mask, lens)
            return None

    def apply_feat_consensus(self, sig):
        if not hasattr(self, "_fed_feat_stats") or self._fed_feat_stats is None:
            return sig
        s = self._fed_feat_stats
        mean = torch.tensor(s["mean"], dtype=sig.dtype, device=sig.device).view(
            1, -1, 1
        )
        std = torch.tensor(s["std"], dtype=sig.dtype, device=sig.device).view(1, -1, 1)
        return (sig - mean) / (std + 1e-6)

    def collect_embedding_stats(self, sig, mask, lens):
        with torch.no_grad():
            embedding, _ = self(sig, mask, lens)
            mean = embedding.mean(dim=0).cpu().numpy().tolist()
            std = embedding.std(dim=0).cpu().numpy().tolist()
            return {"mean": mean, "std": std}

    def embedding_distillation_loss(self, consensus_emb):
        device = next(self.parameters()).device
        local = self._last_embedding if hasattr(self, "_last_embedding") else None
        if local is None:
            return torch.tensor(0.0, device=device)
        mean_t = torch.tensor(consensus_emb["mean"], dtype=torch.float32, device=device)
        std_t = torch.tensor(consensus_emb["std"], dtype=torch.float32, device=device)
        local_mean = local.mean(dim=0)
        loss = ((local_mean - mean_t) ** 2 / (std_t**2 + 1e-8)).mean()
        return loss

    def logit_distillation_loss(self, consensus_logits, batch_logits):
        device = next(self.parameters()).device
        q = torch.tensor(
            numpy.array(consensus_logits) + 1e-8, dtype=torch.float32, device=device
        )
        q = q / q.sum()
        p = torch.softmax(batch_logits, dim=1).mean(dim=0) + 1e-8
        p = p / p.sum()
        return (p * (p.log() - q.log())).sum()
