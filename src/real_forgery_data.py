"""Isolated real-forgery data (MSDS or DeepSignDB); no SLN/synthetic caches."""

import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data_real_forgery" / "msds_v1"
SPLIT_FILES = {
    "train": "MSDS_dev_train.pkl",
    "validation": "MSDS_dev_val.pkl",
    "evaluation": "MSDS_eva.pkl",
}
SPLIT_KEYS = {
    "train": "development_train", "validation": "development_validation",
    "evaluation": "evaluation",
}
FEATURE_VERSION = "real-dynamic12-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pickle(path):
    # Only use trusted, locally supplied dataset pickles (pickle is executable).
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="iso-8859-1")


def inspect_signatures(payload):
    if not isinstance(payload, dict) or not payload:
        raise ValueError("signature split must be a non-empty writer dictionary")
    genuine, forgery = [], []
    for writer, samples in payload.items():
        if not isinstance(writer, (int, np.integer)) or isinstance(writer, bool):
            raise ValueError("MSDS writer IDs must be integers")
        if not isinstance(samples, dict) or set(samples) != {True, False}:
            raise ValueError("each writer must have genuine=True and forgery=False groups")
        for label in (True, False):
            if not len(samples[label]):
                raise ValueError("each writer needs both real genuine and real forgery samples")
            for trajectory in samples[label]:
                value = np.asarray(trajectory)
                if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < 2:
                    raise ValueError("expected a trajectory with shape [L>=2,3] (x,y,pressure)")
                if not np.isfinite(value).all():
                    raise ValueError("trajectory contains non-finite values")
        genuine.append(len(samples[True]))
        forgery.append(len(samples[False]))
    return dict(
        writer_ids=sorted(int(writer) for writer in payload), writers=len(payload),
        genuine=sum(genuine), forgery=sum(forgery),
        min_genuine=min(genuine), min_forgery=min(forgery),
    )


def validate_disjoint(records):
    names = list(SPLIT_FILES)
    for position, name in enumerate(names):
        ids = records[name]["writer_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate writer IDs in " + name)
        for other in names[position + 1:]:
            if set(ids).intersection(records[other]["writer_ids"]):
                raise ValueError("writer leakage between %s and %s" % (name, other))


def _exclusive_copy(source, destination, expected_hash):
    # No move, symlink, hard link or overwrite of the original data.
    with Path(source).open("rb") as src, Path(destination).open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if sha256_file(destination) != expected_hash or sha256_file(source) != expected_hash:
        raise RuntimeError("source changed during copy or copy checksum mismatch")


def prepare(source_root, output_root, split_config, protocol_config, dry_run=False):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if (source_root == output_root or source_root in output_root.parents
            or output_root in source_root.parents):
        raise ValueError("new data folder must be separate from the old data folder")
    if output_root.exists():
        raise FileExistsError("new data folder already exists; choose a new version: %s" % output_root)
    split_config, protocol_config = Path(split_config).resolve(), Path(protocol_config).resolve()
    declaration = json.loads(split_config.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_config.read_text(encoding="utf-8"))
    if declaration.get("dataset") != "MSDS-ChS" or protocol.get("dataset") != "MSDS-ChS":
        raise ValueError("only the documented MSDS-ChS human-recorded data is supported in v1")
    if declaration.get("split_seed") != protocol.get("split_seed"):
        raise ValueError("split and protocol seeds disagree")
    records = {}
    for role, filename in SPLIT_FILES.items():
        source = source_root / filename
        before = sha256_file(source)
        records[role] = inspect_signatures(read_pickle(source))
        if records[role]["writer_ids"] != sorted(declaration[SPLIT_KEYS[role]]):
            raise ValueError("writer IDs differ from the declared split: " + role)
        if sha256_file(source) != before:
            raise RuntimeError("source changed during audit")
        records[role].update(filename=filename, source=str(source), sha256=before,
                             size_bytes=source.stat().st_size)
    validate_disjoint(records)
    manifest = dict(
        version=1, dataset="MSDS-ChS", training_source="real",
        protocol_name="real-skilled-forgery-supervised", feature_version=FEATURE_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(), split_seed=declaration["split_seed"],
        provenance="Existing MSDS-ChS labelled raw trajectory files; no SLN cache or synthesis",
        label_semantics={"True": "human genuine", "False": "human skilled forgery"},
        split_config_sha256=sha256_file(split_config),
        protocol_config_sha256=sha256_file(protocol_config),
        splits=records,
    )
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=False)
        for role, record in records.items():
            _exclusive_copy(record["source"], output_root / record["filename"], record["sha256"])
        _exclusive_copy(split_config, output_root / "writer_split.json", manifest["split_config_sha256"])
        _exclusive_copy(protocol_config, output_root / "cross_session_protocol.json", manifest["protocol_config_sha256"])
        # Completion marker is written last. A partial copy cannot be loaded.
        with (output_root / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def load_manifest(root):
    root = Path(root).resolve()
    if root == (PROJECT_ROOT / "data").resolve():
        raise ValueError("real-forgery training must use the new isolated folder, not data/")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("isolated real-forgery manifest missing; run prepare_real_forgery.py: %s" % root)
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("dataset") == "DeepSignDB-stylus":
        if __package__:
            from .deepsigndb_data import validate_manifest
        else:
            from deepsigndb_data import validate_manifest
        return validate_manifest(root, result)
    if (result.get("version") != 1 or result.get("dataset") != "MSDS-ChS"
            or result.get("training_source") != "real" or result.get("feature_version") != FEATURE_VERSION):
        raise ValueError("unsupported real-forgery manifest")
    validate_disjoint(result["splits"])
    for role, expected in SPLIT_FILES.items():
        if result["splits"][role]["filename"] != expected:
            raise ValueError("unexpected split filename")
    for filename, key in (("writer_split.json", "split_config_sha256"),
                          ("cross_session_protocol.json", "protocol_config_sha256")):
        if sha256_file(root / filename) != result[key]:
            raise ValueError("copied protocol metadata checksum mismatch: " + filename)
    return result


def load_split(root, dataset_name, role="train"):
    if dataset_name.lower() not in ("msds", "mcyt", "bsid", "bsds2", "ebio1", "ebio2"):
        raise ValueError("unknown real data subset; do not mix GPDS synthetic data")
    if role not in SPLIT_FILES:
        raise ValueError("unknown data split")
    manifest = load_manifest(root)
    if manifest["dataset"] == "DeepSignDB-stylus":
        if __package__:
            from .deepsigndb_data import load_split as load_deepsign_split
        else:
            from deepsigndb_data import load_split as load_deepsign_split
        return load_deepsign_split(root, dataset_name, role, manifest)
    if dataset_name.lower() != "msds":
        raise ValueError("this directory contains MSDS, not DeepSignDB")
    record = manifest["splits"][role]
    path = Path(root).resolve() / record["filename"]
    if path.is_symlink() or path.resolve().parent != Path(root).resolve():
        raise ValueError("split must be an independent file inside the isolated folder")
    if sha256_file(path) != record["sha256"]:
        raise ValueError("real-forgery split checksum mismatch: " + role)
    payload = read_pickle(path)
    actual = inspect_signatures(payload)
    for key in actual:
        if actual[key] != record[key]:
            raise ValueError("split contents differ from manifest: " + key)
    return payload, dict(
        training_source="real", protocol="real-skilled-forgery-supervised",
        dataset="MSDS-ChS", split=role, feature_version=FEATURE_VERSION,
        manifest_sha256=sha256_file(Path(root) / "manifest.json"),
        data_root=str(Path(root).resolve()), split_sha256=record["sha256"],
        writers=record["writers"], genuine=record["genuine"], forgery=record["forgery"],
    )


def load_training_data(args):
    from torch.utils.data import DataLoader
    try:
        from .dataset.datasetRealForgery import RealTrainingDataset, TrainingBatchSampler, collate_fn
    except ImportError:
        from dataset.datasetRealForgery import RealTrainingDataset, TrainingBatchSampler, collate_fn
    if getattr(args, "fed_partition", None) or getattr(args, "client_id", -1) >= 0:
        raise ValueError("real-forgery v1 is centralized; legacy synthetic FL partitions are not reused")
    manifest = load_manifest(args.data_root)
    if manifest["dataset"] == "DeepSignDB-stylus":
        if __package__:
            from .deepsigndb_data import training_payload
        else:
            from deepsigndb_data import training_payload
        payload, provenance = training_payload(args.data_root, args.train_datasets.strip(), manifest)
    else:
        if args.train_datasets.strip().lower() != "msds":
            raise ValueError("use --train-datasets msds for this MSDS data directory")
        payload, provenance = load_split(args.data_root, "msds", "train")
    dataset = RealTrainingDataset(payload, args.train_tasks, args.train_shot_g, args.train_shot_f)
    dataset.data_provenance = provenance
    print("REAL skilled-forgery training: %d writers, %d genuine, %d forgery; no synthesis"
          % (provenance["writers"], provenance["genuine"], provenance["forgery"]))
    loader = DataLoader(dataset, num_workers=args.workers,
                        batch_sampler=TrainingBatchSampler(dataset), collate_fn=collate_fn)
    return dataset, loader
