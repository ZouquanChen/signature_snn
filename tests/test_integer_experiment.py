import copy
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import main
import run_integer_tcn_experiment as experiment
from network_integer_tcn import IntegerTCN


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_model():
    return IntegerTCN(12, 3, width=8, embedding_dim=16, dilations=(1, 2),
                      n_task=2, n_shot_g=1, n_shot_f=1)


def batch():
    rng = np.random.default_rng(4)
    return (rng.normal(size=(6, 31, 12)).astype(np.float32),
            np.asarray([31, 24, 23, 21, 20, 19]), np.asarray([0, 0, 0, 1, 1, 1]))


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_microbatch_gradients_match_full_batch(checkpoint_blocks):
    torch.manual_seed(4)
    original = small_model()
    micro = copy.deepcopy(original)
    micro.checkpoint_blocks = checkpoint_blocks
    full_metrics = experiment.backward_batch(original, batch(), torch.device("cpu"), 2)
    micro_metrics = experiment.backward_batch(micro, batch(), torch.device("cpu"), 1)
    for key in full_metrics:
        assert micro_metrics[key] == pytest.approx(full_metrics[key], abs=3e-6)
    for left, right in zip(original.parameters(), micro.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=3e-4)


def test_epoch_returns_complete_not_last_window_statistics():
    model = small_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    expected = experiment.backward_batch(model, batch(), torch.device("cpu"), 1)
    progress = []
    result = experiment.train_epoch(model, [batch(), batch()], optimizer,
                                    experiment.SimpleEpochArgs(1, 1.0), 0,
                                    logging.getLogger("test"), progress.append)
    assert result["total_loss"] == pytest.approx(expected["total_loss"])
    assert result["samples"] == 12 and result["batches"] == 2
    assert progress[-1]["batch"] == 2


def test_best_train_selection_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "PROJECT", tmp_path)
    monkeypatch.setattr(main, "build_training_model", lambda args, data: small_model())
    scores = iter([3., 1., 2.])
    def fake_epoch(model, loader, optimizer, args, epoch, logger, progress):
        with torch.no_grad():
            model.cls.weight.fill_(epoch + 1)
        return dict(epoch=epoch + 1, total_loss=next(scores), ap_loss=0.0, ce_loss=0.0)
    monkeypatch.setattr(experiment, "train_epoch", fake_epoch)
    dataset = SimpleNamespace(data_provenance={"writers": 528})
    class Data:
        data_provenance = dataset.data_provenance
        def __len__(self):
            return 528
    args = SimpleNamespace(seed=1111, epochs=3, lr=.001, warmup_epochs=1,
                           save_interval=2, grad_clip_norm=1., optimizer="adamw",
                           model_name="integer-tcn")
    job = SimpleNamespace(run_tag="test", resume=False, device="cpu",
                          checkpoint_blocks=True, microbatch_tasks=1)
    best = experiment.train_seed(Data(), [None], args, job, logging.getLogger("test"), lambda s: None, "hash")
    root = tmp_path / "models/test/1111"
    weights = experiment.load_owned_checkpoint(root / "best_train_loss.pt")
    assert best["epoch"] == 2 and torch.all(weights["cls.weight"] == 2)
    assert torch.all(experiment.load_owned_checkpoint(root / "epochEnd")["cls.weight"] == 3)
    assert len(json.loads((root / "history.json").read_text())) == 3
    # Simulate a stale sidecar: the complete last checkpoint repairs it.
    experiment.atomic_checkpoint(root / "best_train_loss.pt", {})
    job.resume = True
    resumed = experiment.train_seed(Data(), [None], args, job, logging.getLogger("test"), lambda s: None, "hash")
    assert resumed["epoch"] == 2
    assert torch.all(experiment.load_owned_checkpoint(root / "best_train_loss.pt")["cls.weight"] == 2)
    with pytest.raises(ValueError, match="configuration"):
        experiment.train_seed(Data(), [None], args, job, logging.getLogger("test"), lambda s: None, "different")


def test_seed_statistics_do_not_claim_single_seed_std():
    one = dict(seed=1111, evaluation={"1": {"mean_eer_percent": 4.}, "4": {"mean_eer_percent": 2.}})
    assert experiment.summarize([one])["1vs1"]["sample_std_percent"] is None
    two = dict(seed=2222, evaluation={"1": {"mean_eer_percent": 6.}, "4": {"mean_eer_percent": 4.}})
    result = experiment.summarize([one, two])
    assert result["1vs1"]["mean_eer_percent"] == 5.
    assert result["1vs1"]["sample_std_percent"] == pytest.approx(2 ** .5)
