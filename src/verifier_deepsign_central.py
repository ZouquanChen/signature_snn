"""Central DeepSignDB stylus skilled-forgery results, one EER per seed.

Uses the existing embedding scorer with fixed reference selection. Reports
seed mean/sample std rather than mislabelling a pooled multi-seed ROC as a mean.
Official Comparison_Files were not available for exact pair-list validation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import real_forgery_data as real
    from . import deepsigndb_data as deep
    from . import verifier_stylus_embedding as verifier
except ImportError:
    import real_forgery_data as real
    import deepsigndb_data as deep
    import verifier_stylus_embedding as verifier


def validate_exports(root, run_tag, seed, epoch, manifest):
    manifest_hash = real.sha256_file(Path(root) / "manifest.json")
    for subset in deep.EVALUATION_SUBSETS:
        directory = Path("exps") / run_tag / ("seed%d" % seed) / subset
        provenance = json.loads((directory / "real_data_provenance.json").read_text(encoding="utf-8"))
        if (provenance.get("data_source") != "real" or provenance.get("split") != "evaluation"
                or provenance.get("manifest_sha256") != manifest_hash
                or provenance.get("feature_version") != real.FEATURE_VERSION):
            raise ValueError("export provenance differs from this DeepSignDB experiment")
        record = manifest["files"]["evaluation/" + subset]
        labels = np.load(directory / ("labels_epoch%s.npy" % epoch))
        inds = np.load(directory / ("inds_epoch%s.npy" % epoch))
        features = np.load(directory / ("feats_epoch%s.npy" % epoch))
        if features.ndim != 2 or len(features) != len(labels) or len(inds) != len(labels):
            raise ValueError("invalid embedding export shapes")
        if not np.isfinite(features).all():
            raise ValueError("non-finite evaluation embeddings")
        if (len(np.unique(labels)) != record["writers"] or len(labels) != record["genuine"] + record["forgery"]
                or np.count_nonzero(inds == 1) != record["genuine"]
                or np.count_nonzero(inds == 0) != record["forgery"]):
            raise ValueError("export writer/sample counts differ from audited data")


def run(args):
    manifest = real.load_manifest(args.data_root)
    if manifest["dataset"] != "DeepSignDB-stylus":
        raise ValueError("this verifier only evaluates DeepSignDB stylus")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seeds")
    # Validate every export before starting partial metric output.
    for seed in args.seeds:
        validate_exports(args.data_root, args.run_tag, seed, args.epoch, manifest)
    results = []
    for seed in args.seeds:
        eer = verifier.main([
            "--run-tag", args.run_tag, "--seeds", str(seed), "--epoch", args.epoch,
            "--datasets", ",".join(deep.EVALUATION_SUBSETS), "--mode", "sf",
            "--references", str(args.references),
            "--embedding-normalization", "l2",
        ])
        results.append(dict(seed=seed, eer_percent=float(eer)))
    values = np.asarray([row["eer_percent"] for row in results])
    return dict(dataset="DeepSignDB-stylus", environment="centralized", mode="skilled-forgery",
                references=args.references, training_writers=manifest["training_writers"],
                evaluation_writers=manifest["evaluation_writers"],
                protocol=manifest["protocol_name"],
                score="whole-vector L2 + existing min/mean reference-distance scorer",
                comparison_files_verified=False,
                runs=results, mean_eer_percent=float(values.mean()),
                sample_std_percent=float(values.std(ddof=1)) if len(values) > 1 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--data-root", default=str(deep.DEFAULT_ROOT))
    parser.add_argument("--epoch", default="End")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444, 5555])
    parser.add_argument("--references", type=int, choices=[1, 4], default=1)
    parser.add_argument("--output", help="optional new JSON result path; never overwrite")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error("result file already exists")
    result = run(args)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
