import builtins
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
import real_forgery_data as real
import train_integer_tcn
from dataset.datasetRealForgery import (
    RealTrainingDataset, RealEvaluationDataset, TrainingBatchSampler,
    extract_features, collate_fn, evaluation_module,
)


def trajectory(offset=0.0, length=31):
    t = np.linspace(0, 1, length)
    return np.stack((np.sin(t * 3) + offset, np.cos(t * 2), t + .2), axis=1)


def writer(offset=0.0):
    return {True: [trajectory(offset + i) for i in range(3)],
            False: [trajectory(offset + 10 + i) for i in range(2)]}


@pytest.fixture
def sources(tmp_path):
    source, output = tmp_path / "old_data", tmp_path / "new_real_data"
    source.mkdir()
    groups = {"train": {1: writer(), 2: writer(2)},
              "validation": {3: writer(3)}, "evaluation": {4: writer(4)}}
    for role, filename in real.SPLIT_FILES.items():
        with (source / filename).open("wb") as handle:
            pickle.dump(groups[role], handle)
    split = tmp_path / "original_split.json"
    split.write_text(json.dumps(dict(dataset="MSDS-ChS", split_seed=11,
                                    **{real.SPLIT_KEYS[role]: list(payload)
                                       for role, payload in groups.items()})))
    protocol = tmp_path / "original_protocol.json"
    protocol.write_text(json.dumps(dict(dataset="MSDS-ChS", split_seed=11)))
    return source, output, split, protocol


def test_prepare_preserves_originals_and_copies_independently(sources):
    source, output, split, protocol = sources
    before = {path: (real.sha256_file(path), path.stat().st_mtime_ns) for path in source.iterdir()}
    manifest = real.prepare(*sources)
    assert manifest["training_source"] == "real"
    assert manifest["splits"]["train"]["genuine"] == 6
    for path, info in before.items():
        assert (real.sha256_file(path), path.stat().st_mtime_ns) == info
        assert real.sha256_file(output / path.name) == info[0]
        assert not path.samefile(output / path.name)
    payload, provenance = real.load_split(output, "msds", "train")
    assert set(payload) == {1, 2} and provenance["forgery"] == 4
    assert set(real.load_split(output, "msds", "validation")[0]) == {3}
    assert set(real.load_split(output, "msds", "evaluation")[0]) == {4}
    with pytest.raises(FileExistsError):
        real.prepare(*sources)
    with pytest.raises(ValueError, match="separate"):
        real.prepare(source, source / "inside", split, protocol)


def test_dry_run_and_refuse_old_unmanifested_data(sources):
    source, output, *_ = sources
    real.prepare(*sources, dry_run=True)
    assert not output.exists()
    with pytest.raises(FileNotFoundError, match="manifest"):
        real.load_split(source, "msds")
    with pytest.raises(ValueError, match="GPDS"):
        real.load_split(output, "gpds")


def test_writer_leakage_rejected_before_writing(sources):
    source, output, split, _ = sources
    with (source / real.SPLIT_FILES["evaluation"]).open("wb") as handle:
        pickle.dump({1: writer()}, handle)
    declaration = json.loads(split.read_text())
    declaration["evaluation"] = [1]
    split.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="leakage"):
        real.prepare(*sources)
    assert not output.exists()


def test_tampered_copy_rejected(sources):
    real.prepare(*sources)
    path = sources[1] / real.SPLIT_FILES["train"]
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        real.load_split(sources[1], "msds")


def test_missing_forgeries_rejected():
    with pytest.raises(ValueError, match="both"):
        real.inspect_signatures({1: {True: [trajectory()], False: []}})


@pytest.mark.parametrize("length", [2, 3, 12, 31])
def test_features_are_finite_non_mutating_and_short_safe(length):
    path = trajectory(length=length)
    original = path.copy()
    features = extract_features(path)
    assert features.shape == (length, 12) and np.isfinite(features).all()
    np.testing.assert_array_equal(path, original)
    constant = extract_features(np.ones((length, 3)))
    assert np.isfinite(constant).all()


def test_feature_definitions_match_legacy_on_normal_signatures():
    from dataset.utils import featExt
    path = trajectory(length=51)
    legacy = []
    featExt([path.copy()], legacy)
    np.testing.assert_allclose(extract_features(path), legacy[0], atol=2e-5, rtol=2e-5)


def test_real_episode_layout_and_no_replacement():
    payload = {1: writer(), 2: writer(2)}
    data = RealTrainingDataset(payload, taskSize=2, taskNumGen=2, taskNumNeg=2)
    np.random.seed(2)
    batch = next(iter(TrainingBatchSampler(data)))
    assert len(batch) == 10
    for start in (0, 5):
        selected = batch[start:start + 5]
        assert len(set(selected)) == 5
        assert [data.is_genuine[i] for i in selected] == [True, True, True, False, False]
        assert len({data.labels[i] for i in selected}) == 1
    assert data.labels[batch[0]] != data.labels[batch[5]]
    evaluation = RealEvaluationDataset(payload)
    np.testing.assert_array_equal(data.feats[0], evaluation.feats[0])
    x, lengths, labels = collate_fn([(extract_features(trajectory(length=12)), 12, 0)])
    assert x.shape == (1, 13, 12) and lengths.tolist() == [12]
    with pytest.raises(ValueError, match="not enough"):
        RealTrainingDataset(payload, taskSize=2, taskNumGen=3, taskNumNeg=2)


def test_real_loader_never_imports_sln(sources, monkeypatch):
    real.prepare(*sources)
    actual_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if "sln" in name.lower() or "slbox" in name.lower() or "synthesis" in name.lower():
            raise AssertionError("synthetic dependency was imported: " + name)
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(main, "_import_dataset_module", lambda: pytest.fail("legacy synthesis loader called"))
    args = main.build_parser().parse_args([
        "--model-name", "integer-tcn", "--training-source", "real",
        "--data-root", str(sources[1]), "--train-datasets", "msds",
        "--sln-root", "does-not-exist", "--train-tasks", "2",
        "--train-shot-g", "2", "--train-shot-f", "2",
    ])
    dataset, loader = main.load_training_data(args)
    assert dataset.data_provenance["split"] == "train"
    assert dataset.data_provenance["writers"] == 2
    assert next(iter(loader))[0].shape == (10, 31, 12)
    assert not (sources[1] / "cache").exists()


def test_default_routing_preserves_old_pipeline():
    legacy = main.build_parser().parse_args(["--model-name", "snn"])
    assert main.resolve_training_source(legacy) == "synthetic"
    assert legacy.data_root == "data"
    new = train_integer_tcn.build_parser().parse_args([])
    assert new.training_source == "real" and new.train_datasets == "deepsigndb"
    assert new.data_root == "data_real_forgery/deepsigndb_stylus_v1"
    with pytest.raises(SystemExit):
        train_integer_tcn.resolve_training_argv(["--run-tag", "test", "--training-source", "synthetic"])


def test_real_evaluation_routing_and_validation_isolation(sources):
    real.prepare(*sources)
    args = evaluate_stylus.build_parser().parse_args([
        "--data-root", str(sources[1]), "--eval-datasets", "msds"])
    assert evaluate_stylus._uses_real_data(args)
    assert set(evaluate_stylus._load_evaluation_signatures(args, "MSDS")) == {4}
    args.eval_split = "validation"
    assert set(evaluate_stylus._load_evaluation_signatures(args, "MSDS")) == {3}
    with pytest.raises(SystemExit):
        evaluate_stylus.main(["--data-root", str(sources[1]), "--eval-split", "validation"])
    args.data_source = "legacy"
    with pytest.raises(ValueError, match="matching real"):
        evaluate_stylus._uses_real_data(args)
    training_args = main.build_parser().parse_args([
        "--model-name", "snn", "--data-root", str(sources[1])])
    with pytest.raises(ValueError, match="legacy synthesis"):
        main.load_training_data(training_args)


def test_real_training_and_export_end_to_end(sources, tmp_path, monkeypatch):
    real.prepare(*sources)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "get_runtime_device", lambda torch: torch.device("cpu"))
    monkeypatch.setattr(evaluate_stylus, "get_runtime_device", lambda torch: torch.device("cpu"))
    monkeypatch.setattr(main, "_import_dataset_module", lambda: pytest.fail("SLN loader called"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        main.main(["--model-name", "integer-tcn", "--data-root", str(sources[1]),
                   "--train-datasets", "msds", "--train-tasks", "2", "--train-shot-g", "2",
                   "--train-shot-f", "2", "--epochs", "1", "--run-tag", "real-e2e",
                   "--seed", "1111", "--tcn-width", "8", "--tcn-embed-dim", "16"])
        checkpoint = tmp_path / "models/real-e2e/1111/epochEnd"
        info = json.loads((checkpoint.parent / "training_config.json").read_text())
        assert info["training_source"] == "real" and info["data_provenance"]["split"] == "train"
        evaluate_stylus.main(["--model-name", "integer-tcn", "--checkpoint", str(checkpoint),
                             "--data-root", str(sources[1]), "--eval-datasets", "msds",
                             "--run-tag", "real-e2e", "--seed", "1111", "--export-seq-feats"])
        exported = tmp_path / "exps/real-e2e/seed1111/msds"
        features = np.load(exported / "feats_epochEnd.npy")
        assert features.shape == (5, 16) and np.isfinite(features).all()
        assert np.load(exported / "inds_epochEnd.npy").tolist() == [1, 1, 1, 0, 0]
        assert json.loads((exported / "real_data_provenance.json").read_text())["split"] == "evaluation"
    finally:
        torch.set_num_threads(old_threads)
