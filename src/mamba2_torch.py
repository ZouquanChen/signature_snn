# Adapted from the Mamba-2 architecture / SSD factorization reference.
# Original source: Copyright (c) 2024, Tri Dao, Albert Gu.
# SPDX-License-Identifier: Apache-2.0
# Modifications (2026-09-10): standalone PyTorch, shared B/C group,
# masked variable lengths, logarithmic inter-chunk affine prefix scan.
# See ../third_party/mamba/NOTICE.md and LICENSE.
"""Portable Mamba-2-style selective SSM, NOT the official fused backend.

No loop over sequence positions and no full L x L matrix. Within chunks we
use SSD's causal matrix factorization; across chunks we compose affine maps
in O(log(number_of_chunks)) parallel stages. This reference path trades speed
and memory for compatibility with Windows / PyTorch 2.1, without new packages.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")


def _causal_segment_sums(log_decay):
    """Entry [i,j] = sum(log_decay[j+1:i+1]), i>=j; no prefix subtraction."""
    length = log_decay.shape[-1]
    indices = torch.arange(length, device=log_decay.device)
    below = indices[:, None] > indices[None, :]
    increments = log_decay.unsqueeze(-1).expand(*log_decay.shape, length)
    sums = increments.masked_fill(~below, 0).cumsum(dim=-2)
    return sums.masked_fill(indices[:, None] < indices[None, :], -torch.inf)


def _affine_prefix(decay, contribution):
    """Inclusive prefix of h -> decay*h + contribution, in chunk order."""
    distance = 1
    # This is logarithmic in chunk count, NOT sequential in token/time count.
    while distance < contribution.shape[1]:
        combined_decay = decay[:, distance:] * decay[:, :-distance]
        combined_state = contribution[:, distance:] + decay[:, distance:] * contribution[:, :-distance]
        decay = torch.cat((decay[:, :distance], combined_decay), dim=1)
        contribution = torch.cat((contribution[:, :distance], combined_state), dim=1)
        distance *= 2
    return decay, contribution


def ssd_chunk_scan(x, log_decay, input_state, output_state, chunk_size=64, initial_state=None):
    """Evaluate the Mamba-2 discrete SSM with one shared B/C group.

    x: [B,L,H,P], already multiplied by dt.
    log_decay: [B,L,H] = dt * negative A.
    input_state/output_state (B/C): [B,L,N], shared across heads.
    h_t = exp(log_decay_t)*h_(t-1) + x_t outer B_t
    y_t = h_t @ C_t
    Returns y [B,L,H,P] and final h [B,H,P,N].
    """
    _positive_integer("chunk_size", chunk_size)
    if x.ndim != 4 or log_decay.shape != x.shape[:3]:
        raise ValueError("expected x=[B,L,H,P], log_decay=[B,L,H]")
    batch, length, heads, head_dim = x.shape
    if (input_state.ndim != 3 or input_state.shape[:2] != (batch, length)
            or output_state.shape != input_state.shape):
        raise ValueError("B/C must have equal shape [B,L,N]")
    if not (x.dtype == log_decay.dtype == input_state.dtype == output_state.dtype):
        raise ValueError("SSD tensors must have the same dtype")
    state_dim = input_state.shape[-1]
    state_shape = (batch, heads, head_dim, state_dim)
    if initial_state is None:
        initial_state = x.new_zeros(state_shape)
    elif initial_state.shape != state_shape:
        raise ValueError("incorrect initial SSM state shape")
    if length == 0:
        return x, initial_state
    padding = (-length) % chunk_size
    if padding:
        x = F.pad(x, (0, 0, 0, 0, 0, padding))
        log_decay = F.pad(log_decay, (0, 0, 0, padding))
        input_state = F.pad(input_state, (0, 0, 0, padding))
        output_state = F.pad(output_state, (0, 0, 0, padding))
    chunks = x.shape[1] // chunk_size
    values = x.reshape(batch, chunks, chunk_size, heads, head_dim)
    keys = input_state.reshape(batch, chunks, chunk_size, state_dim)
    queries = output_state.reshape(batch, chunks, chunk_size, state_dim)
    decay_logs = log_decay.reshape(batch, chunks, chunk_size, heads).permute(0, 3, 1, 2)
    prefix_logs = decay_logs.cumsum(dim=-1)

    # Chunk-local causal interactions. The only square matrix is chunk_size^2.
    local_decay = _causal_segment_sums(decay_logs).exp()
    key_query = torch.matmul(queries, keys.transpose(-1, -2))
    local_weights = local_decay * key_query[:, None]
    local_output = torch.einsum("bhqij,bqjhp->bqihp", local_weights, values)

    # Every chunk's contribution to its ending state, with a reverse scan that
    # avoids subtracting two large negative prefix sums.
    suffix_logs = decay_logs[..., 1:].flip(-1).cumsum(-1).flip(-1)
    ending_decay = F.pad(suffix_logs, (0, 1)).exp()
    contributions = torch.einsum("bqtn,bhqt,bqthp->bqhpn", keys, ending_decay, values)
    chunk_decay = decay_logs.sum(-1).exp().permute(0, 2, 1)[..., None, None]
    inclusive_decay, inclusive_state = _affine_prefix(chunk_decay, contributions)
    inclusive_state = inclusive_state + inclusive_decay * initial_state[:, None]
    preceding = torch.cat((initial_state[:, None], inclusive_state[:, :-1]), dim=1)
    carried_output = torch.einsum("bqtn,bqhpn,bhqt->bqthp", queries, preceding, prefix_logs.exp())
    output = (local_output + carried_output).reshape(batch, -1, heads, head_dim)
    return output[:, :length], inclusive_state[:, -1]


class GatedRMSNorm(nn.Module):
    """Gate BEFORE RMS normalization, matching Mamba2Simple's chosen ordering."""

    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value, gate):
        value = value * F.silu(gate)
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class TorchMamba2(nn.Module):
    """Mamba2Simple-style mixer with ngroups=1 and no persistent inference cache."""

    def __init__(self, width, state_dim=16, expand=1, head_dim=32, conv_size=4, chunk_size=64):
        super().__init__()
        for name, value in dict(width=width, state_dim=state_dim, expand=expand,
                                head_dim=head_dim, conv_size=conv_size, chunk_size=chunk_size).items():
            _positive_integer(name, value)
        self.inner = width * expand
        if self.inner % head_dim:
            raise ValueError("width * expand must be divisible by head_dim")
        self.heads, self.head_dim = self.inner // head_dim, head_dim
        self.state_dim, self.conv_size, self.chunk_size = state_dim, conv_size, chunk_size
        # Projection splits into gate z, value x, shared B/C and selective dt.
        self.in_projection = nn.Linear(width, 2 * self.inner + 2 * state_dim + self.heads, bias=False)
        conv_channels = self.inner + 2 * state_dim
        self.local_conv = nn.Conv1d(conv_channels, conv_channels, conv_size,
                                    groups=conv_channels, bias=True)
        dt = torch.exp(torch.rand(self.heads) * (math.log(.1) - math.log(.001)) + math.log(.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.log_A = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.skip_D = nn.Parameter(torch.ones(self.heads))
        self.gated_norm = GatedRMSNorm(self.inner)
        self.out_projection = nn.Linear(self.inner, width, bias=False)

    def _encode_input(self, sequence, mask):
        return sequence

    def _encode_output(self, sequence, mask):
        return sequence

    def forward(self, sequence, mask):
        if sequence.ndim != 3 or mask.shape != sequence.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Mamba expects [B,L,C] and a boolean [B,L] mask")
        if sequence.shape[1] == 0:
            return sequence
        sequence = sequence.masked_fill(~mask[..., None], 0)
        projected = self.in_projection(self._encode_input(sequence, mask))
        gate, conv_input, dt_raw = torch.split(projected,
            (self.inner, self.inner + 2 * self.state_dim, self.heads), dim=-1)
        local = self.local_conv(F.pad(conv_input.transpose(1, 2), (self.conv_size - 1, 0)))
        local = F.silu(local.transpose(1, 2)).masked_fill(~mask[..., None], 0)
        value, input_state, output_state = torch.split(local,
            (self.inner, self.state_dim, self.state_dim), dim=-1)
        dt = F.softplus(dt_raw + self.dt_bias).masked_fill(~mask[..., None], 0)
        value = value.reshape(*value.shape[:2], self.heads, self.head_dim)
        log_decay = -self.log_A.exp() * dt
        output, _ = ssd_chunk_scan(value * dt[..., None], log_decay, input_state,
                                   output_state, self.chunk_size)
        output = output + value * self.skip_D[None, None, :, None]
        output = output.reshape(*sequence.shape[:2], self.inner)
        output = self.gated_norm(output, gate)
        output = self.out_projection(self._encode_output(output, mask))
        return output.masked_fill(~mask[..., None], 0)


class Mamba2ResidualBlock(nn.Module):
    def __init__(self, width, state_dim=16, expand=1, head_dim=32, conv_size=4,
                 chunk_size=64, residual_scale=.1, mixer_factory=None):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        factory = TorchMamba2 if mixer_factory is None else mixer_factory
        self.mixer = factory(width, state_dim, expand, head_dim, conv_size, chunk_size)
        self.scale = nn.Parameter(torch.full((width,), residual_scale))

    def forward(self, hidden, mask):
        sequence = hidden.transpose(1, 2)
        mixed = self.mixer(self.norm(sequence), mask)
        return (sequence + mixed * self.scale).masked_fill(~mask[..., None], 0).transpose(1, 2)
