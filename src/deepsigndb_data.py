"""DeepSignDB stylus: the published DsDTW 528-writer training subset.

Development/Evaluation IDs have different namespaces, as documented by the
upstream preprocessors. Numeric overlap is NOT treated as writer leakage.
We audit expected ranges and reject exact cross-split trajectory duplicates;
this is not a substitute for raw-subject provenance or official comparison files.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from . import real_forgery_data as common
except ImportError:
    import real_forgery_data as common


DEFAULT_ROOT = common.PROJECT_ROOT / "data_real_forgery/deepsigndb_stylus_v1"
PROTOCOL_PATH = common.PROJECT_ROOT / "configs/protocols/deepsigndb_stylus_central.json"
# role, dataset, filename, inclusive writer range, genuine/writer, forgery/writer
FILE_SPECS = (
    ("train", "mcyt", "MCYT_dev.pkl", 1, 230, 25, 25),
    ("train", "bsid", "BSID_dev.pkl", 231, 498, 16, 12),
    ("train", "ebio1", "EBio1_dev.pkl", 1009, 1038, 40, 30),
    ("evaluation", "mcyt", "MCYT_eva.pkl", 1, 100, 25, 25),
    ("evaluation", "bsid", "BSID_eva.pkl", 101, 232, 16, 12),
    ("evaluation", "bsds2", "BSDS2_eva.pkl", 233, 372, 30, 20),
    ("evaluation", "ebio1", "EBio1_eva.pkl", 373, 407, 40, 30),
    ("evaluation", "ebio2", "EBio2_eva.pkl", 408, 442, 8, 6),
)
TRAIN_SUBSETS = ("mcyt", "bsid", "ebio1")
EVALUATION_SUBSETS = ("mcyt", "bsid", "bsds2", "ebio1", "ebio2")


def _fingerprints(payload):
    result = set()
    for samples in payload.values():
        for paths in samples.values():
            for path in paths:
                array = np.ascontiguousarray(path, dtype=np.float64)
                digest = hashlib.sha256(str(array.shape).encode("ascii") + array.tobytes()).hexdigest()
                result.add(digest)
    return result


def prepare(source_root, output_root=DEFAULT_ROOT, protocol_path=PROTOCOL_PATH, dry_run=False):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if (source_root == output_root or source_root in output_root.parents
            or output_root in source_root.parents):
        raise ValueError("new folder must be separate from old data")
    if output_root.exists():
        raise FileExistsError("choose a NEW DeepSignDB version directory: %s" % output_root)
    records, fingerprints = {}, {"train": set(), "evaluation": set()}
    for role, dataset, filename, first, last, genuine, forgery in FILE_SPECS:
        source = source_root / filename
        checksum = common.sha256_file(source)
        payload = common.read_pickle(source)
        statistics = common.inspect_signatures(payload)
        if statistics["writer_ids"] != list(range(first, last + 1)):
            raise ValueError("unexpected split-scoped IDs in " + filename)
        if any(len(s[True]) != genuine or len(s[False]) != forgery for s in payload.values()):
            raise ValueError("unexpected per-writer sample counts in " + filename)
        fingerprints[role].update(_fingerprints(payload))
        statistics.update(role=role, subset=dataset, filename=filename, source=str(source),
                          sha256=checksum, size_bytes=source.stat().st_size)
        records[role + "/" + dataset] = statistics
        if common.sha256_file(source) != checksum:
            raise RuntimeError("source changed during audit")
    overlap = len(fingerprints["train"] & fingerprints["evaluation"])
    if overlap:
        raise ValueError("exact train/evaluation trajectory leakage: %d" % overlap)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if protocol.get("dataset") != "DeepSignDB" or protocol.get("training_environment") != "centralized":
        raise ValueError("expected centralized DeepSignDB protocol")
    manifest = dict(
        version=2, dataset="DeepSignDB-stylus", training_source="real",
        protocol_name="deepsigndb-stylus-central-real-v1", feature_version=common.FEATURE_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        id_semantics="split-scoped Development/Evaluation IDs, not globally shared integers",
        training_subsets=list(TRAIN_SUBSETS), evaluation_subsets=list(EVALUATION_SUBSETS),
        training_writers=sum(r["writers"] for r in records.values() if r["role"] == "train"),
        evaluation_writers=sum(r["writers"] for r in records.values() if r["role"] == "evaluation"),
        exact_train_evaluation_trajectory_overlap=overlap,
        independent_validation_split=False,
        protocol_sha256=common.sha256_file(protocol_path), files=records,
        provenance_sources=protocol["sources"],
        provenance_limit="Existing preprocessed local files; original Comparison_Files and subject mapping not independently reconstructed",
    )
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=False)
        for record in records.values():
            common._exclusive_copy(record["source"], output_root / record["filename"], record["sha256"])
        common._exclusive_copy(protocol_path, output_root / "protocol.json", manifest["protocol_sha256"])
        with (output_root / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def validate_manifest(root, manifest):
    if (manifest.get("version") != 2 or manifest.get("dataset") != "DeepSignDB-stylus"
            or manifest.get("training_source") != "real"
            or manifest.get("feature_version") != common.FEATURE_VERSION):
        raise ValueError("unsupported DeepSignDB manifest")
    expected = {role + "/" + dataset: filename for role, dataset, filename, *_ in FILE_SPECS}
    if set(manifest["files"]) != set(expected):
        raise ValueError("DeepSignDB file set differs from recorded protocol")
    for key, filename in expected.items():
        if manifest["files"][key]["filename"] != filename:
            raise ValueError("unexpected DeepSignDB filename")
    if common.sha256_file(Path(root) / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("DeepSignDB protocol checksum mismatch")
    if manifest.get("exact_train_evaluation_trajectory_overlap") != 0:
        raise ValueError("DeepSignDB audit indicates cross-split duplicate trajectories")
    return manifest


def load_split(root, subset, role, manifest):
    if role == "validation":
        raise ValueError("published 528-writer DeepSignDB setup has no independent validation split; do not use evaluation for tuning")
    key = role + "/" + subset.lower()
    if key not in manifest["files"]:
        raise ValueError("subset is not in this DeepSignDB protocol: " + key)
    record = manifest["files"][key]
    path = Path(root).resolve() / record["filename"]
    if path.is_symlink() or path.resolve().parent != Path(root).resolve():
        raise ValueError("expected an independent file inside the new folder")
    if common.sha256_file(path) != record["sha256"]:
        raise ValueError("DeepSignDB data checksum mismatch: " + key)
    payload = common.read_pickle(path)
    actual = common.inspect_signatures(payload)
    if any(actual[k] != record[k] for k in actual):
        raise ValueError("DeepSignDB contents differ from manifest")
    provenance = dict(dataset="DeepSignDB-stylus", subset=subset, split=role,
                      training_source="real", environment="centralized",
                      protocol=manifest["protocol_name"], data_root=str(Path(root).resolve()),
                      manifest_sha256=common.sha256_file(Path(root) / "manifest.json"),
                      feature_version=common.FEATURE_VERSION, split_sha256=record["sha256"],
                      writers=record["writers"], genuine=record["genuine"], forgery=record["forgery"])
    return payload, provenance


def training_payload(root, selected, manifest):
    names = list(TRAIN_SUBSETS) if selected.lower() == "deepsigndb" else selected.lower().split(",")
    names = [name.strip() for name in names]
    if len(names) != len(set(names)) or set(names) != set(TRAIN_SUBSETS):
        raise ValueError("central DeepSignDB requires deepsigndb or mcyt,bsid,ebio1; other datasets are deferred")
    merged, sources = {}, []
    for name in TRAIN_SUBSETS:
        payload, provenance = load_split(root, name, "train", manifest)
        # Namespace corpus IDs for future-proof class labels; preserve file IDs.
        merged.update({(name, int(writer)): samples for writer, samples in payload.items()})
        sources.append(provenance)
    provenance = dict(sources[0], subset="mcyt,bsid,ebio1", source_files=sources,
                      writers=sum(p["writers"] for p in sources),
                      genuine=sum(p["genuine"] for p in sources),
                      forgery=sum(p["forgery"] for p in sources))
    provenance.pop("split_sha256")
    return merged, provenance
