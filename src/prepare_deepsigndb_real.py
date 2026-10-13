"""Create an independent DeepSignDB stylus, centralized real-forgery data copy."""

import argparse
import json

try:
    from . import deepsigndb_data as data
except ImportError:
    import deepsigndb_data as data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(data.common.PROJECT_ROOT / "data"))
    parser.add_argument("--output-root", default=str(data.DEFAULT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = data.prepare(args.source_root, args.output_root, dry_run=args.dry_run)
    print(json.dumps(dict(output_root=args.output_root, dry_run=args.dry_run,
                          training_writers=manifest["training_writers"],
                          evaluation_writers=manifest["evaluation_writers"],
                          exact_overlap=manifest["exact_train_evaluation_trajectory_overlap"],
                          files={k: {field: value for field, value in record.items()
                                     if field != "writer_ids"} for k, record in manifest["files"].items()}), indent=2))


if __name__ == "__main__":
    main()
