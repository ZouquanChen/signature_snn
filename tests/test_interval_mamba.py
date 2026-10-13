import copy
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluate_stylus
import main
import train_integer_tcn
from learnable_interval_spike import LearnableIntervalSpike, _IntervalCount
from network_interval_mamba import IntervalTCNMamba2
from network_integer_tcn_mamba import IntegerTCNMamba2
from run_integer_tcn_experiment import backward_batch
import run_integer_tcn_experiment as experiment


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(old)


def test_ordered_initialization_and_signed_counts():
    node = LearnableIntervalSpike(1)
    torch.testing.assert_close(node.thresholds(), torch.tensor([.5, 1.5, 2.5, 3.5]))
    positive = torch.tensor([.1, .7, 1.7, 2.7, 4.1]).view(1, 1, -1)
    torch.testing.assert_close(node(positive), torch.tensor([0., 1., 2., 3., 4.]).view(1, 1, -1))
    torch.testing.assert_close(node(-positive), -node(positive))
    assert node(torch.zeros(1, 1, 1)).item() == 0
    thresholds = node.thresholds().detach().view(1, 1, -1)
    torch.testing.assert_close(node(thresholds), torch.arange(4.).view(1, 1, -1))
    custom = LearnableIntervalSpike(1, initial_thresholds=[.4, 1.2, 2.7, 4.2])
    torch.testing.assert_close(custom.thresholds(), torch.tensor([.4, 1.2, 2.7, 4.2]))


@pytest.mark.parametrize("channelwise", [False, True])
def test_gap_bounds_and_gradients_are_finite(channelwise):
    node = LearnableIntervalSpike(2, channelwise=channelwise)
    for raw_value in (-50., 0., 50.):
        with torch.no_grad():
            node.raw_gaps.fill_(raw_value)
        intervals = node.intervals()
        thresholds = node.thresholds()
        assert torch.isfinite(thresholds).all()
        assert intervals.min() >= .05 - 1e-6 and intervals.max() <= 2. + 1e-6
        assert (thresholds.diff(dim=-1) > 0).all()
    node = LearnableIntervalSpike(2, channelwise=channelwise)
    x = torch.full((2, 2, 5), .7, requires_grad=True)
    node(x).sum().backward()
    assert node.raw_gaps.grad is not None and node.raw_gaps.grad.abs().sum() > 0
    assert torch.isfinite(node.raw_gaps.grad).all()


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("channelwise", [False, True])
def test_custom_backward_matches_declared_normalized_surrogate(signed, channelwise):
    values = torch.tensor([-.7, -.2, .0, .2, .7, 8.]).repeat(2, 2, 1).requires_grad_()
    thresholds = torch.tensor([.15, .3, .45, .6])
    if channelwise:
        thresholds = thresholds[None, :].repeat(2, 1)
    thresholds.requires_grad_()
    weights = torch.arange(1, values.numel() + 1).reshape_as(values).float()
    out = _IntervalCount.apply(values, thresholds, 1., signed)
    (out * weights).sum().backward()
    th = thresholds[None, :, None, :] if channelwise else thresholds
    pos = (1 - (values.detach()[..., None] - th).abs()).clamp_min(0)
    neg = ((1 - (-values.detach()[..., None] - th).abs()).clamp_min(0)
           if signed else torch.zeros_like(pos))
    total = (pos + neg).sum(-1)
    denominator = total.clamp_min(1)
    expected_value = weights * total / denominator
    element_threshold = weights[..., None] * (neg - pos) / denominator[..., None]
    expected_threshold = element_threshold.sum((0, 2)) if channelwise else element_threshold.sum((0, 1, 2))
    torch.testing.assert_close(values.grad, expected_value)
    torch.testing.assert_close(thresholds.grad, expected_threshold)
    assert (values.grad.abs() <= weights.abs() + 1e-6).all()


@pytest.mark.parametrize("levels", [2, 4, 8])
def test_overlapping_thresholds_do_not_multiply_input_gradient(levels):
    node = LearnableIntervalSpike(1, levels=levels,
                                   initial_thresholds=[.1 * (i + 1) for i in range(levels)])
    x = torch.tensor([[[.2]]], requires_grad=True)
    node(x).sum().backward()
    assert x.grad.item() == pytest.approx(1.0)


def test_thresholds_learn_and_fixed_control_does_not():
    node = LearnableIntervalSpike(1)
    frozen = LearnableIntervalSpike(1, fixed=True)
    x = torch.tensor([[[.2, .7, 1.3, 2.2, 3.2]]])
    torch.testing.assert_close(node(x), frozen(x))
    original = node.thresholds().detach().clone()
    optimizer = torch.optim.SGD(node.parameters(), lr=.05)
    node(x).mean().backward()
    optimizer.step()
    assert not torch.equal(node.thresholds(), original)
    assert not frozen.raw_gaps.requires_grad
    assert torch.equal(node(x), node(x).round())


def test_padding_does_not_update_thresholds():
    reference = LearnableIntervalSpike(1)
    padded_node = copy.deepcopy(reference)
    x = torch.tensor([[[.2, .7, 1.2]]], requires_grad=True)
    padded = torch.tensor([[[.2, .7, 1.2, float("nan"), float("nan")]]], requires_grad=True)
    reference(x).sum().backward()
    out = padded_node(padded, torch.tensor([[True, True, True, False, False]]))
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.count_nonzero(out[..., 3:]) == 0
    torch.testing.assert_close(reference.raw_gaps.grad, padded_node.raw_gaps.grad)
    torch.testing.assert_close(x.grad, padded.grad[..., :3])
    assert torch.count_nonzero(padded.grad[..., 3:]) == 0


@pytest.mark.parametrize("thresholds", [[.5, .4, 2., 3.], [.5, 1.5], [float("nan"), 1., 2., 3.], [.01, 1., 2., 3.]])
def test_invalid_thresholds(thresholds):
    with pytest.raises(ValueError):
        LearnableIntervalSpike(1, initial_thresholds=thresholds)


def options():
    return dict(n_in=12, n_classes=3, width=8, embedding_dim=16, dilations=(1, 2),
                n_task=2, n_shot_g=1, n_shot_f=1, mamba_state_dim=3, mamba_head_dim=4,
                mamba_chunk_size=4)


@pytest.mark.parametrize("sites,count", [("input", 6), ("output", 6), ("both", 8)])
def test_projection_integer_inputs_and_hooks(sites, count):
    model = IntervalTCNMamba2(**options(), interval_sites=sites)
    captured = []
    hooks = []
    for block in model.mamba_blocks:
        if sites in ("input", "both"):
            hooks.append(block.mixer.in_projection.register_forward_pre_hook(
                lambda m, args: captured.append(args[0])))
        if sites in ("output", "both"):
            hooks.append(block.mixer.out_projection.register_forward_pre_hook(
                lambda m, args: captured.append(args[0])))
    e, _ = model(torch.randn(2, 31, 12), lens=torch.tensor([31, 23]))
    assert e.shape == (2, 16) and len(model.spike_neurons) == count
    assert any((q < 0).any() for q in captured)
    for q in captured:
        assert torch.equal(q, q.round()) and q.abs().max() <= 4
        assert torch.count_nonzero(q[1, 23:]) == 0
    for hook in hooks:
        hook.remove()


def test_same_initial_backbone_and_fixed_interval_control():
    torch.manual_seed(9)
    baseline = IntegerTCNMamba2(**options())
    torch.manual_seed(9)
    learned = IntervalTCNMamba2(**options())
    torch.manual_seed(9)
    fixed = IntervalTCNMamba2(**options(), interval_fixed=True)
    for name, value in baseline.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(learned.state_dict()[name], value, atol=0, rtol=0)
    x = torch.randn(2, 29, 12)
    torch.testing.assert_close(learned(x), fixed(x), atol=0, rtol=0)


def test_causality_mask_and_checkpoint_microbatch_gradients():
    model = IntervalTCNMamba2(**options())
    x = torch.randn(1, 31, 12)
    before = model.forward_sequence(x)[0]
    changed = x.clone()
    changed[:, 17:] += 5
    torch.testing.assert_close(model.forward_sequence(changed)[0][:, :, :17], before[:, :, :17])
    padded = torch.cat((x, torch.full((1, 9, 12), float("nan"))), dim=1)
    torch.testing.assert_close(model(padded, lens=torch.tensor([31]))[0], model(x)[0], atol=3e-6, rtol=3e-6)
    micro = copy.deepcopy(model)
    micro.checkpoint_blocks = True
    batch = (np.random.default_rng(4).normal(size=(6, 31, 12)).astype(np.float32),
             np.array([31, 25, 23, 21, 20, 19]), np.array([0, 0, 0, 1, 1, 1]))
    full_result = backward_batch(model, batch, torch.device("cpu"), 2)
    micro_result = backward_batch(micro, batch, torch.device("cpu"), 1)
    assert full_result["total_loss"] == pytest.approx(micro_result["total_loss"], abs=3e-5)
    for left, right in zip(model.parameters(), micro.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=4e-5, rtol=2e-3)
    assert any(p.grad.abs().sum() > 0 for name, p in model.named_parameters() if name.endswith("raw_gaps"))


def test_configuration_checkpoint_and_no_gap_weight_decay(tmp_path):
    config = SRC.parent / "configs/integer_tcn_mamba2_interval.json"
    argv = train_integer_tcn.resolve_training_argv(["--config", str(config), "--run-tag", "interval-test",
                                                   "--interval-fixed", "--interval-channelwise"])
    args = main.build_parser().parse_args(argv)
    assert args.model_name == "integer-tcn-mamba2-interval" and args.interval_fixed
    model = IntervalTCNMamba2(**options(), interval_channelwise=True)
    with torch.no_grad():
        model.mamba_blocks[0].mixer.input_spike.raw_gaps.add_(.1)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    state = evaluate_stylus.load_checkpoint_state(torch, str(path))
    eval_args = evaluate_stylus.build_parser().parse_args(["--model-name", "integer-tcn-mamba2-interval"])
    restored = evaluate_stylus.build_evaluation_model(eval_args, SimpleNamespace(featDim=12), state)
    restored.load_state_dict(state)
    x = torch.randn(2, 27, 12)
    torch.testing.assert_close(restored(x), model(x))
    assert restored.interval_diagnostics() == model.interval_diagnostics()
    optimizer = main.build_training_optimizer(model, args)
    gap_ids = {id(p) for name, p in model.named_parameters() if name.endswith("raw_gaps")}
    for group in optimizer.param_groups:
        if any(id(p) in gap_ids for p in group["params"]):
            assert group["weight_decay"] == 0
    args.optimizer = "sgd"
    for group in main.build_training_optimizer(model, args).param_groups:
        if any(id(p) in gap_ids for p in group["params"]):
            assert group["weight_decay"] == 0
    with pytest.raises((ValueError, RuntimeError)):
        IntervalTCNMamba2(**options(), interval_channelwise=True, interval_gap_max=3).load_state_dict(state)


def test_interval_logs_and_best_snapshot_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "PROJECT", tmp_path)
    monkeypatch.setattr(main, "build_training_model", lambda args, data: IntervalTCNMamba2(**options()))
    losses = iter([3., 1., 2.])
    def fake_epoch(model, loader, optimizer, args, epoch, logger, progress):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("raw_gaps"):
                    parameter.fill_(.2 * (epoch + 1))
        return dict(epoch=epoch + 1, total_loss=next(losses), ap_loss=0., ce_loss=0.)
    monkeypatch.setattr(experiment, "train_epoch", fake_epoch)
    class Dataset:
        data_provenance = {}
        def __len__(self):
            return 3
    args = SimpleNamespace(seed=1111, epochs=3, lr=.001, warmup_epochs=0,
                           save_interval=2, grad_clip_norm=1., optimizer="adamw",
                           model_name="integer-tcn-mamba2-interval")
    job = SimpleNamespace(run_tag="interval", resume=False, device="cpu",
                          checkpoint_blocks=True, microbatch_tasks=1)
    experiment.train_seed(Dataset(), [None], args, job, logging.getLogger("test"), lambda s: None, "hash")
    directory = tmp_path / "models/interval/1111"
    best = json.loads((directory / "best_train_intervals.json").read_text())
    assert best["epoch"] == 2
    assert (directory / "initial_intervals.json").is_file()
    assert json.loads((directory / "interval_diagnostics.json").read_text())["epoch"] == 3
    (directory / "best_train_intervals.json").write_text("{}")
    job.resume = True
    experiment.train_seed(Dataset(), [None], args, job, logging.getLogger("test"), lambda s: None, "hash")
    assert json.loads((directory / "best_train_intervals.json").read_text()) == best


def test_config_rejects_a_mislabelled_interval_gradient(tmp_path):
    source = SRC.parent / "configs/integer_tcn_mamba2_interval.json"
    config = json.loads(source.read_text())
    config["interval_gradient"] = "identity-ste"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="normalized-triangle-v1"):
        train_integer_tcn.build_parser(path)
