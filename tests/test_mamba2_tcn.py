import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluate_stylus
import main
import train_integer_tcn
from mamba2_torch import ssd_chunk_scan, TorchMamba2
from model_factory import build_model
from network_integer_tcn import IntegerTCN
from network_integer_tcn_mamba import IntegerTCNMamba2
from run_integer_tcn_experiment import backward_batch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def sequential_oracle(x, a, b, c, initial=None):
    """Slow recurrence is a TEST ORACLE only; never imported by production."""
    state = initial if initial is not None else x.new_zeros(x.shape[0], x.shape[2], x.shape[3], b.shape[-1])
    values = []
    for t in range(x.shape[1]):
        state = a[:, t, :, None, None].exp() * state + x[:, t, :, :, None] * b[:, t, None, None, :]
        values.append((state * c[:, t, None, None, :]).sum(-1))
    return torch.stack(values, dim=1), state


@pytest.mark.parametrize("length,chunk", [(1, 1), (1, 8), (7, 4), (13, 3), (16, 4), (37, 8)])
def test_ssd_outputs_and_gradients_match_recurrence(length, chunk):
    x = torch.randn(2, length, 2, 3, dtype=torch.float64, requires_grad=True)
    a = (-torch.rand(2, length, 2, dtype=torch.float64) * .2).requires_grad_()
    b = torch.randn(2, length, 4, dtype=torch.float64, requires_grad=True)
    c = torch.randn(2, length, 4, dtype=torch.float64, requires_grad=True)
    initial = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    actual = ssd_chunk_scan(x, a, b, c, chunk, initial)
    expected = sequential_oracle(x, a, b, c, initial)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    inputs = (x, a, b, c, initial)
    grad_actual = torch.autograd.grad(actual[0].square().mean() + actual[1].sum(), inputs, retain_graph=True)
    grad_expected = torch.autograd.grad(expected[0].square().mean() + expected[1].sum(), inputs)
    torch.testing.assert_close(grad_actual, grad_expected, atol=1e-8, rtol=1e-8)


def test_empty_scan_and_chunk_independence():
    x = torch.randn(1, 29, 2, 3)
    a = -torch.rand(1, 29, 2) * .05
    b, c = torch.randn(1, 29, 4), torch.randn(1, 29, 4)
    baseline = ssd_chunk_scan(x, a, b, c, 1)
    for chunk in (4, 8, 16, 64):
        torch.testing.assert_close(ssd_chunk_scan(x, a, b, c, chunk), baseline, atol=2e-5, rtol=2e-5)
    output, state = ssd_chunk_scan(x[:, :0], a[:, :0], b[:, :0], c[:, :0], 4)
    assert output.shape == (1, 0, 2, 3) and torch.count_nonzero(state) == 0


def test_mixer_masked_recurrence_and_gate_order():
    mixer = TorchMamba2(8, state_dim=3, head_dim=4, chunk_size=4)
    x = torch.randn(2, 13, 8)
    mask = torch.arange(13)[None, :] < torch.tensor([13, 9])[:, None]
    actual = mixer(x, mask)
    projected = mixer.in_projection(x.masked_fill(~mask[..., None], 0))
    z, xbc, delta = projected.split((8, 14, 2), dim=-1)
    xbc = F.silu(mixer.local_conv(F.pad(xbc.transpose(1, 2), (3, 0))).transpose(1, 2))
    xbc = xbc.masked_fill(~mask[..., None], 0)
    values, b, c = xbc.split((8, 3, 3), dim=-1)
    values = values.reshape(2, 13, 2, 4)
    dt = F.softplus(delta + mixer.dt_bias).masked_fill(~mask[..., None], 0)
    scanned, _ = sequential_oracle(values * dt[..., None], -mixer.log_A.exp() * dt, b, c)
    scanned = (scanned + values * mixer.skip_D[None, None, :, None]).reshape(2, 13, 8)
    expected = mixer.out_projection(mixer.gated_norm(scanned, z)).masked_fill(~mask[..., None], 0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def small_model(**overrides):
    kwargs = dict(n_in=12, n_classes=3, width=8, embedding_dim=16, dilations=(1, 2),
                  n_task=2, n_shot_g=1, n_shot_f=1, mamba_layers=2,
                  mamba_state_dim=3, mamba_head_dim=4, mamba_chunk_size=4)
    kwargs.update(overrides)
    return IntegerTCNMamba2(**kwargs)


def test_shapes_causality_padding_and_stateless_calls():
    model = small_model().eval()
    x = torch.randn(1, 47, 12)
    sequence = model.forward_sequence(x)[0]
    changed = x.clone()
    changed[:, 23:] += 5
    torch.testing.assert_close(model.forward_sequence(changed)[0][:, :, :23], sequence[:, :, :23])
    embedding = model(x)[0]
    padded = F.pad(x, (0, 0, 0, 15), value=float("nan"))
    torch.testing.assert_close(model(padded, lens=torch.tensor([47]))[0], embedding, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model(x)[0], embedding)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(1))
    assert model.receptive_field is None and model.local_receptive_field == 19
    assert len(model.spike_neurons) == 4  # two TCN blocks, no fake Mamba spikes
    for length in (0, 1, 3):
        e, logits = model(torch.zeros(2, length, 12), lens=torch.tensor([length, 0]))
        assert torch.isfinite(e).all() and torch.count_nonzero(e[1]) == 0
        assert logits.shape == (2, 3)


def test_context_extends_beyond_tcn_window():
    model = small_model(dilations=(1,)).eval()
    for block in model.mamba_blocks:
        with torch.no_grad():
            block.mixer.log_A.zero_()
            block.mixer.dt_bias.fill_(-5)
    x = torch.randn(1, 80, 12)
    changed = x.clone()
    changed[:, 0] += torch.arange(12) * .5
    # Before the SSM, the final local feature is identical.
    local_a = IntegerTCN.forward_sequence(model, x)[0]
    local_b = IntegerTCN.forward_sequence(model, changed)[0]
    torch.testing.assert_close(local_a[:, :, -1], local_b[:, :, -1], atol=0, rtol=0)
    long_a = model.forward_sequence(x)[0][:, :, -1]
    long_b = model.forward_sequence(changed)[0][:, :, -1]
    assert (long_a - long_b).abs().max() > 1e-7


def test_microbatch_and_activation_checkpoint_gradients():
    model = small_model()
    micro = copy.deepcopy(model)
    micro.checkpoint_blocks = True
    rng = np.random.default_rng(4)
    batch = (rng.normal(size=(6, 31, 12)).astype(np.float32),
             np.array([31, 24, 23, 21, 20, 19]), np.array([0, 0, 0, 1, 1, 1]))
    reference = backward_batch(model, batch, torch.device("cpu"), 2)
    actual = backward_batch(micro, batch, torch.device("cpu"), 1)
    assert actual["total_loss"] == pytest.approx(reference["total_loss"], abs=2e-5)
    for original, recomputed in zip(model.parameters(), micro.parameters()):
        torch.testing.assert_close(original.grad, recomputed.grad, atol=3e-5, rtol=1e-3)
    assert model.mamba_blocks[0].mixer.log_A.grad.abs().sum() > 0


def test_factory_checkpoint_and_configuration(tmp_path):
    model = small_model()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    state = evaluate_stylus.load_checkpoint_state(torch, str(path))
    args = evaluate_stylus.build_parser().parse_args(["--model-name", "integer-tcn-mamba2"])
    restored = evaluate_stylus.build_evaluation_model(args, SimpleNamespace(featDim=12), state)
    restored.load_state_dict(state)
    x = torch.randn(2, 25, 12)
    torch.testing.assert_close(restored(x)[0], model(x)[0])
    assert state["_extra_state"]["mamba_backend"] == "torch-ssd-prefix-v1"
    incompatible = small_model(mamba_chunk_size=8)
    with pytest.raises(ValueError, match="mismatch"):
        incompatible.load_state_dict(state)
    config = SRC.parent / "configs/integer_tcn_mamba2.json"
    argv = train_integer_tcn.resolve_training_argv(["--config", str(config), "--run-tag", "new-mamba"])
    training_args = main.build_parser().parse_args(argv)
    assert training_args.model_name == "integer-tcn-mamba2"
    assert training_args.mamba_layers == 2 and main.resolve_training_source(training_args) == "real"
    built = build_model("integer-tcn-mamba2", n_in=12, n_classes=3)
    assert isinstance(built, IntegerTCNMamba2)


@pytest.mark.parametrize("kwargs", [dict(head_dim=3), dict(state_dim=0), dict(chunk_size=0)])
def test_invalid_mamba_dimensions(kwargs):
    with pytest.raises(ValueError):
        TorchMamba2(8, **kwargs)
