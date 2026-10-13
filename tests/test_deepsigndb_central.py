import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import deepsigndb_data as deep
import real_forgery_data as real
import main
import train_integer_tcn
import verifier_deepsign_central as central
import verifier_stylus_embedding as embedding_verifier
import verifier_stylus_snn_tvs as tvs


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source, target = tmp_path / "old", tmp_path / "deep_new"
    source.mkdir()
    specs = []
    for index, (role, subset, filename, *_) in enumerate(deep.FILE_SPECS):
        # Intentionally reuse ID 1 in train/evaluation: official split namespaces.
        specs.append((role, subset, filename, 1, 1, 3, 2))
        t = np.linspace(0, 1, 31)
        def path(i):
            return np.stack((t + index * 100 + i, t ** 2, .1 + t), axis=1)
        payload = {1: {True: [path(i) for i in range(3)], False: [path(i) for i in range(3, 5)]}}
        with (source / filename).open("wb") as handle:
            pickle.dump(payload, handle)
    monkeypatch.setattr(deep, "FILE_SPECS", tuple(specs))
    manifest = deep.prepare(source, target)
    return source, target, manifest


def test_split_scoped_ids_copies_and_central_training(prepared):
    source, target, manifest = prepared
    assert manifest["training_writers"] == 3 and manifest["evaluation_writers"] == 5
    assert manifest["exact_train_evaluation_trajectory_overlap"] == 0
    for record in manifest["files"].values():
        assert real.sha256_file(source / record["filename"]) == record["sha256"]
        assert not (source / record["filename"]).samefile(target / record["filename"])
    loaded = real.load_manifest(target)
    payload, provenance = deep.training_payload(target, "deepsigndb", loaded)
    assert len(payload) == 3 and len(set(payload)) == 3
    assert provenance["genuine"] == 9 and provenance["forgery"] == 6
    args = main.build_parser().parse_args([
        "--model-name", "integer-tcn", "--train-datasets", "deepsigndb",
        "--data-root", str(target), "--train-tasks", "2", "--train-shot-g", "1", "--train-shot-f", "1"])
    dataset, loader = main.load_training_data(args)
    assert dataset.data_provenance["environment"] == "centralized"
    assert next(iter(loader))[0].shape == (6, 31, 12)
    with pytest.raises(ValueError, match="no independent validation"):
        real.load_split(target, "mcyt", "validation")
    with pytest.raises(ValueError, match="deferred"):
        deep.training_payload(target, "msds", loaded)
    args.fed_partition = "old-fl-partition.json"
    with pytest.raises(ValueError, match="centralized"):
        main.load_training_data(args)


def test_duplicate_trajectory_leakage_and_overwrite_rejected(prepared, tmp_path):
    source, target, _ = prepared
    with pytest.raises(FileExistsError):
        deep.prepare(source, target)
    with (source / "MCYT_dev.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    with (source / "MCYT_eva.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match="leakage"):
        deep.prepare(source, tmp_path / "should_not_exist")
    assert not (tmp_path / "should_not_exist").exists()


def test_default_deepsign_and_retained_msds_preset():
    args = train_integer_tcn.build_parser().parse_args([])
    assert args.train_datasets == "deepsigndb"
    assert args.data_root.endswith("deepsigndb_stylus_v1")
    old = train_integer_tcn.build_parser(SRC.parent / "configs/integer_tcn_msds.json").parse_args([])
    assert old.train_datasets == "msds" and old.data_root.endswith("msds_v1")
    assert tvs.build_parser().parse_args([]).references == 1
    assert embedding_verifier.build_parser().parse_args([
        "--run-tag", "test", "--seeds", "1", "--references", "4"]).references == 4


def test_four_reference_path_and_single_seed_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "exps/test/seed1/mcyt"
    directory.mkdir(parents=True)
    np.save(directory / "feats_epochEnd.npy", np.eye(10, dtype=np.float32))
    np.save(directory / "labels_epochEnd.npy", np.repeat([0, 1], 5))
    np.save(directory / "inds_epochEnd.npy", np.tile([1, 1, 1, 0, 0], 2))
    calls = []
    def verify(*args):
        calls.append((args[4], args[5]))
        return [5.], [4.], np.array([0., 2.]), np.array([2., 0.]), 2, 2
    monkeypatch.setattr(tvs, "verify_dataset_sf", verify)
    eer = embedding_verifier.main(["--run-tag", "test", "--seeds", "1", "--datasets", "mcyt", "--references", "4",
                                   "--embedding-normalization", "l2"])
    assert calls == [(4, 1)] and np.isfinite(eer)


def test_export_provenance_and_seed_statistics(prepared, tmp_path, monkeypatch):
    _, root, manifest = prepared
    monkeypatch.chdir(tmp_path)
    digest = real.sha256_file(root / "manifest.json")
    for seed in [1, 2]:
        for subset in deep.EVALUATION_SUBSETS:
            directory = tmp_path / "exps/run" / ("seed%d" % seed) / subset
            directory.mkdir(parents=True)
            np.save(directory / "feats_epochEnd.npy", np.ones((5, 8), dtype=np.float32))
            np.save(directory / "labels_epochEnd.npy", np.zeros(5))
            np.save(directory / "inds_epochEnd.npy", [1, 1, 1, 0, 0])
            (directory / "real_data_provenance.json").write_text(json.dumps(dict(
                data_source="real", split="evaluation", manifest_sha256=digest,
                feature_version=real.FEATURE_VERSION)))
    central.validate_exports(root, "run", 1, "End", manifest)
    values = iter([3., 5.])
    monkeypatch.setattr(central.verifier, "main", lambda argv: next(values))
    result = central.run(SimpleNamespace(data_root=root, run_tag="run", seeds=[1, 2], epoch="End", references=1))
    assert result["mean_eer_percent"] == 4
    assert result["sample_std_percent"] == pytest.approx(2 ** .5)
    path = tmp_path / "exps/run/seed1/mcyt/real_data_provenance.json"
    invalid = json.loads(path.read_text())
    invalid["split"] = "validation"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="provenance"):
        central.validate_exports(root, "run", 1, "End", manifest)
