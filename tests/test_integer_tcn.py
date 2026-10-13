import copy
import json
import pickle
import sys
from pathlib import Path
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
from model_factory import build_model
from network_integer_tcn import IntegerSpike, IntegerTCN


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(old)


def small_model(**kwargs):
    defaults = dict(n_in=12, n_classes=3, width=8, embedding_dim=16,
                    dilations=(1, 2), n_task=1, n_shot_g=1, n_shot_f=1)
    defaults.update(kwargs)
    return IntegerTCN(**defaults)


@pytest.mark.parametrize("maximum", [1, 2, 4, 8])
def test_integer_forward_and_clipped_ste(maximum):
    x = torch.tensor([-1., 0., .49, .5, .51, .9, float(maximum), maximum + .1],
                     requires_grad=True)
    weights = torch.arange(1, x.numel() + 1, dtype=x.dtype)
    q = IntegerSpike(maximum)(x)
    assert torch.equal(q, x.detach().clamp(0, maximum).round())
    (q * weights).sum().backward()
    assert torch.equal(x.grad, weights * ((x >= 0) & (x <= maximum)))


@pytest.mark.parametrize("maximum", [0, -1, 2.5, True])
def test_invalid_integer_max(maximum):
    with pytest.raises(ValueError, match="positive integer"):
        IntegerSpike(maximum)


def test_shapes_unit_embeddings_and_integer_conv_inputs():
    model = small_model()
    captured = []
    hooks = [conv.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
             for block in model.blocks for conv in (block.depthwise, block.pointwise)]
    x = torch.randn(3, 23, 12)
    embedding, logits = model(x, lens=torch.tensor([23, 19, 1]))
    assert embedding.shape == (3, 16) and logits.shape == (3, 3)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(3))
    assert len(captured) == 4
    for q in captured:
        assert torch.equal(q, q.round()) and q.min() >= 0 and q.max() <= 4
        assert torch.count_nonzero(q[1, :, 19:]) == 0
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("training", [False, True])
def test_padding_batch_and_layout_invariance(training):
    model = small_model().train(training)
    x = torch.randn(1, 19, 12)
    reference = model(x, lens=torch.tensor([19]))[0]
    padded = torch.cat((x, torch.full((1, 14, 12), float("nan"))), dim=1)
    mask = torch.from_numpy(model.getOutputMask([19]))
    actual = model(padded, mask=mask, lens=torch.tensor([19]))[0]
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
    batch = torch.cat((padded, torch.randn_like(padded)), dim=0)
    result = model(batch, lens=torch.tensor([19, 33]))[0]
    torch.testing.assert_close(result[:1], reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model(x.transpose(1, 2), lens=torch.tensor([19]))[0], reference)


def test_causal_sequence_and_no_cross_call_state():
    model = small_model().eval()
    x = torch.randn(1, 43, 12)
    original = model.forward_sequence(x)[0]
    changed = x.clone()
    changed[:, 21:] += 100
    actual = model.forward_sequence(changed)[0]
    torch.testing.assert_close(original[:, :, :21], actual[:, :, :21])
    torch.testing.assert_close(model.forward_sequence(x)[0], original)
    assert model.sim_steps == 1
    assert model.receptive_field == 7 + 4 * (1 + 2)


@pytest.mark.parametrize("steps", [0, 1, 5, 37])
def test_empty_and_short_sequences(steps):
    model = small_model()
    x = torch.randn(2, steps, 12)
    e, logits = model(x, lens=torch.tensor([steps, 0]))
    assert torch.isfinite(e).all() and torch.isfinite(logits).all()
    assert torch.count_nonzero(e[1]) == 0
    assert torch.count_nonzero(logits[1]) == 0
    if steps:
        (e[0].sum() + logits[0].sum()).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_length_and_mask_validation():
    model = small_model()
    x = torch.randn(2, 23, 12)
    for lengths in ([24, 20], [-1, 20], [2.5, 20], [20]):
        with pytest.raises(ValueError):
            model(x, lens=torch.tensor(lengths))
    with pytest.raises(ValueError, match="disagree"):
        model(x, mask=torch.ones(2, 23), lens=torch.tensor([23, 20]))
    with pytest.raises(ValueError, match="ambiguous"):
        model(torch.randn(2, 12, 12))


def test_checkpoint_metadata_and_auto_reconstruction(tmp_path):
    model = small_model(integer_max=8, dilations=(1, 3)).eval()
    path = tmp_path / "epochEnd"
    torch.save(model.state_dict(), path)
    state = evaluate_stylus.load_checkpoint_state(torch, str(path))
    args = evaluate_stylus.build_parser().parse_args(["--model-name", "integer-tcn"])
    restored = evaluate_stylus.build_evaluation_model(args, SimpleNamespace(featDim=12), state)
    restored.load_state_dict(state)
    restored.eval()
    x = torch.randn(2, 29, 12)
    torch.testing.assert_close(restored(x)[0], model(x)[0])
    assert restored.integer_max == 8 and restored.dilations == (1, 3)
    assert not any(key.startswith(("classifier.", "spike_neurons.")) for key in state)
    with pytest.raises(ValueError, match="mismatch"):
        small_model(integer_max=4, dilations=(1, 3)).load_state_dict(state)
    bad = copy.deepcopy(state)
    bad.pop("_extra_state")
    with pytest.raises(ValueError, match="metadata"):
        IntegerTCN.from_state_dict_config(bad, 12)


def training_args():
    return main.build_parser().parse_args([
        "--model-name", "integer-tcn", "--tcn-width", "8", "--tcn-embed-dim", "16",
        "--tcn-dilations", "1,2", "--train-tasks", "1", "--train-shot-g", "1",
        "--train-shot-f", "1", "--grad-clip-norm", "1", "--run-tag", "test-integer-tcn",
    ])


class TinyDataset:
    featDim = 12
    testKeys = [0, 1]
    accumNumNeg = 3
    numGen = 2

    def __len__(self):
        return 3


def test_shared_training_step_changes_weights():
    args = training_args()
    model = main.build_training_model(args, TinyDataset())
    assert isinstance(model, IntegerTCN)
    optimizer = main.build_training_optimizer(model, args)
    assert isinstance(optimizer, torch.optim.AdamW)
    before = model.stem.weight.detach().clone()
    batch = (np.random.default_rng(9).normal(size=(3, 25, 12)).astype(np.float32),
             np.array([25, 21, 17]), np.array([0, 0, 0]))
    main.run_training_epoch(model, [batch], optimizer, 0, args)
    assert not torch.equal(before, model.stem.weight)
    for block in model.blocks:
        assert block.depthwise.weight.grad is not None
        assert block.depthwise.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_ap_objective_matches_legacy():
    from network_snn import Sig2VecSNN
    model = small_model()
    model._iter_episode_triplets = lambda e: iter([(e[0], e[1:2], e[2:3])])
    x = torch.randn(3, 16, requires_grad=True)
    actual = model.APLoss_DLM(x)
    expected = Sig2VecSNN.APLoss_DLM(model, x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual[0], x, retain_graph=True)[0],
                               torch.autograd.grad(expected[0], x)[0])


def test_generic_export_sequence_and_all_spikes(tmp_path):
    model = small_model().eval()
    loader = []
    for steps in (21, 29):
        loader.append((np.random.default_rng(steps).normal(size=(3, steps, 12)).astype(np.float32),
                       np.array([steps, steps - 2, steps - 4]), np.zeros(3, dtype=np.int64)))
    result = evaluate_stylus.export_dataset_features(
        model, loader, TinyDataset(), str(tmp_path), torch, torch.autograd.Variable,
        export_spikes=True, export_seq_feats=True, export_all_layers=True,
    )
    assert result["feats"].shape == (6, 16)
    assert result["seq_feats"].shape == (6, 8, 29)
    assert result["spike_rates"].shape == (6, 32)
    assert result["seq_mask"].sum(axis=1).tolist() == [21, 19, 17, 29, 27, 25]
    assert len(result["all_layer_spikes"]) == 4
    np.testing.assert_array_equal(evaluate_stylus._spike_valid_lengths(model, [21, 17], 3, 21),
                                  [21, 17])
    np.testing.assert_array_equal(evaluate_stylus._spike_valid_lengths(object(), [21, 17], 3, 21),
                                  [6, 5])
    assert not model.sp2._forward_hooks
    assert all(not node._forward_hooks for node in model.spike_neurons)


def test_launcher_config_overrides_and_factory():
    forwarded = train_integer_tcn.resolve_training_argv(["--run-tag", "test", "--integer-max", "2"])
    args = main.build_parser().parse_args(forwarded)
    assert args.model_name == "integer-tcn" and args.integer_max == 2
    assert args.epochs == 100 and args.optimizer == "adamw" and args.grad_clip_norm == 1
    assert args.lambda_div == 0 and args.tcn_embed_dim == 256
    assert isinstance(build_model("integer-tcn", n_in=12, n_classes=3), IntegerTCN)
    with pytest.raises(SystemExit):
        train_integer_tcn.resolve_training_argv([])
    args.burst_spike = True
    with pytest.raises(ValueError, match="burst-spike"):
        main.build_training_model(args, TinyDataset())


def test_legacy_optimizer_default_unchanged():
    args = main.build_parser().parse_args([])
    assert args.model_name == "cnn" and args.embed_dim == 1024
    assert isinstance(main.build_training_optimizer(torch.nn.Linear(2, 2), args), torch.optim.SGD)


@pytest.mark.parametrize("model_name", ["integer-tcn", "integer-tcn-mamba2", "integer-tcn-mamba2-interval"])
def test_actual_train_and_evaluation_entrypoints(tmp_path, monkeypatch, model_name):
    """Exercise metadata saving AND the evaluator's legacy-alias bypass."""
    monkeypatch.chdir(tmp_path)
    batch = (np.random.default_rng(1).normal(size=(3, 25, 12)).astype(np.float32),
             np.array([25, 21, 19]), np.array([0, 0, 0]))
    monkeypatch.setattr(main, "load_training_data", lambda args: (TinyDataset(), [batch]))
    monkeypatch.setattr(main, "get_runtime_device", lambda torch: torch.device("cpu"))
    argv = ["--model-name", model_name, "--run-tag", "e2e", "--seed", "1111",
            "--epochs", "1", "--warmup-epochs", "0", "--tcn-width", "8",
            "--tcn-embed-dim", "16", "--tcn-dilations", "1,3", "--integer-max", "2",
            "--train-tasks", "1", "--train-shot-g", "1", "--train-shot-f", "1",
            "--mamba-head-dim", "4", "--mamba-state-dim", "3", "--mamba-chunk-size", "4"]
    main.main(argv)
    checkpoint = tmp_path / "models/e2e/1111/epochEnd"
    assert checkpoint.is_file()
    config = json.loads((checkpoint.parent / "training_config.json").read_text())
    assert config["spike_gradient"] == "clipped-ste" and config["surrogate_sigma"] is None
    assert config["model_config"]["integer_max"] == 2
    with pytest.raises(FileExistsError):
        main.main(argv)
    with open(tmp_path / "MSDS_eva.pkl", "wb") as handle:
        pickle.dump({0: []}, handle)
    monkeypatch.setattr(evaluate_stylus, "_import_dataset_module", lambda: SimpleNamespace(
        dataset=lambda sigDict: TinyDataset()))
    monkeypatch.setattr(evaluate_stylus, "get_runtime_device", lambda torch: torch.device("cpu"))
    seen = []

    def capture_export(model, *args, **kwargs):
        seen.append(model)
        with torch.no_grad():
            e, _ = model(torch.from_numpy(batch[0]), lens=torch.from_numpy(batch[1]))
        assert torch.isfinite(e).all() and e.shape == (3, 16)

    monkeypatch.setattr(evaluate_stylus, "export_split", capture_export)
    evaluate_stylus.main(["--model-name", model_name, "--checkpoint", str(checkpoint),
                         "--data-root", str(tmp_path), "--eval-datasets", "msds"])
    assert len(seen) == 1 and seen[0].integer_max == 2 and seen[0].dilations == (1, 3)
    assert seen[0].get_extra_state()["model_name"] == model_name
