"""Ordered learnable threshold intervals with signed integer-valued output.

Borrow the project's Burst idea (positive spacings -> cumulative thresholds ->
threshold counts), not its single-pass PLIF scaling or unused reset dynamics.
This node is stateless. It emits counts in a floating tensor for autograd and
is NOT an integer-only arithmetic implementation.
"""

import math

import torch
from torch import nn


def parse_thresholds(values, levels):
    if values is None:
        return tuple(index + .5 for index in range(levels))
    if isinstance(values, str):
        values = values.split(",")
    values = tuple(float(value) for value in values)
    if len(values) != levels or not all(math.isfinite(value) for value in values):
        raise ValueError("initial thresholds must contain one finite value per level")
    return values


def _level_threshold(thresholds, index):
    threshold = thresholds[..., index]
    return threshold.view(1, -1, 1) if thresholds.ndim == 2 else threshold


def _windows(value, threshold, width, signed):
    positive = (1 - (value - threshold).abs() / width).clamp_min(0)
    negative = ((1 - (-value - threshold).abs() / width).clamp_min(0)
                if signed else torch.zeros_like(positive))
    return positive, negative


class _IntervalCount(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, thresholds, width, signed, input_gradient="normalized-triangle"):
        ctx.save_for_backward(value, thresholds)
        ctx.width, ctx.signed, ctx.input_gradient = width, signed, input_gradient
        count = torch.zeros_like(value)
        # Loop only over the small number of thresholds K, never over time L.
        # Avoid allocating [B,C,L,K] tensors for the longest signatures.
        for index in range(thresholds.shape[-1]):
            threshold = _level_threshold(thresholds, index)
            count = count + (value > threshold).to(value.dtype)
            if signed:
                count = count - (-value > threshold).to(value.dtype)
        return count

    @staticmethod
    def backward(ctx, grad_output):
        value, thresholds = ctx.saved_tensors
        total = torch.zeros_like(value)
        for index in range(thresholds.shape[-1]):
            positive, negative = _windows(value, _level_threshold(thresholds, index), ctx.width, ctx.signed)
            total = total + positive + negative
        denominator = total.clamp_min(1)
        # Either use the smooth normalized triangle or a clipped STE for the
        # input. Threshold gradients remain triangle-based in both modes.
        if ctx.needs_input_grad[0]:
            if ctx.input_gradient == "clipped-ste":
                limit = thresholds[..., -1]
                limit = limit.view(1, -1, 1) if thresholds.ndim == 2 else limit
                grad_value = grad_output * (value.abs() <= limit).to(value.dtype)
            else:
                grad_value = grad_output * (total / denominator)
        else:
            grad_value = None
        grad_thresholds = None
        if ctx.needs_input_grad[1]:
            normalized_gradient = grad_output / denominator
            parts = []
            for index in range(thresholds.shape[-1]):
                positive, negative = _windows(value, _level_threshold(thresholds, index), ctx.width, ctx.signed)
                element_gradient = normalized_gradient * (negative - positive)
                parts.append(element_gradient.sum(dim=(0, 2)) if thresholds.ndim == 2
                             else element_gradient.sum())
            grad_thresholds = torch.stack(parts, dim=-1)
        return grad_value, grad_thresholds, None, None, None


class LearnableIntervalSpike(nn.Module):
    """[B,C,L] -> signed counts, with shared or per-channel ordered thresholds.

    gap = gap_min + (gap_max-gap_min)*sigmoid(raw_gap)
    theta_k = cumulative_sum(gap)_k, starting from ZERO in input-current units.
    The negative side shares the positive thresholds, preserving odd symmetry.
    """

    def __init__(self, channels, levels=4, initial_thresholds=None, gap_min=.05,
                 gap_max=2., surrogate_width=1., channelwise=False, signed=True, fixed=False,
                 input_gradient="normalized-triangle"):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (channels, levels)):
            raise ValueError("channels/levels must be positive integers")
        if not (math.isfinite(gap_min) and math.isfinite(gap_max) and 0 < gap_min < gap_max):
            raise ValueError("gap bounds must be finite and 0 < min < max")
        if not math.isfinite(surrogate_width) or surrogate_width <= 0:
            raise ValueError("surrogate_width must be finite and positive")
        thresholds = parse_thresholds(initial_thresholds, levels)
        # Do not subtract a voltage threshold from dimensionless multipliers,
        # as the historical Burst initializer did. These are all input units.
        gaps = torch.diff(torch.tensor((0.,) + thresholds, dtype=torch.float64))
        if ((gaps <= gap_min) | (gaps >= gap_max)).any():
            raise ValueError("initial gaps must lie strictly between gap_min and gap_max")
        probability = (gaps - gap_min) / (gap_max - gap_min)
        raw = torch.log(probability / (1 - probability)).float()
        if channelwise:
            raw = raw[None, :].repeat(channels, 1)
        self.raw_gaps = nn.Parameter(raw, requires_grad=not fixed)
        self.channels, self.maximum = channels, levels
        self.gap_min, self.gap_max = float(gap_min), float(gap_max)
        self.surrogate_width = float(surrogate_width)
        self.channelwise, self.signed, self.fixed = bool(channelwise), bool(signed), bool(fixed)
        if input_gradient not in ("normalized-triangle", "clipped-ste"):
            raise ValueError("input_gradient must be normalized-triangle or clipped-ste")
        self.input_gradient = input_gradient

    def intervals(self):
        return self.gap_min + (self.gap_max - self.gap_min) * self.raw_gaps.sigmoid()

    def thresholds(self):
        return self.intervals().cumsum(dim=-1)

    def forward(self, value, mask=None):
        if value.ndim != 3 or value.shape[1] != self.channels or not value.is_floating_point():
            raise ValueError("interval spike expects floating [B,C,L] with configured channels")
        if mask is not None:
            if mask.shape != (value.shape[0], value.shape[2]) or mask.dtype != torch.bool:
                raise ValueError("mask must be boolean [B,L]")
            clean = value.masked_fill(~mask[:, None, :], 0)
        else:
            clean = value
        # Include the mask after the spike too: masked positions must not supply
        # a threshold gradient even though surrogate windows can cover zero.
        output = _IntervalCount.apply(clean, self.thresholds().to(value),
                                       self.surrogate_width, self.signed, self.input_gradient)
        return output if mask is None else output.masked_fill(~mask[:, None, :], 0)
