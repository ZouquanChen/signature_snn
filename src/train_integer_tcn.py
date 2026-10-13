"""Train the recorded Integer-TCN config with explicit CLI overrides.

Run from the standalone project root. This launcher does not start training
when imported; --help also never loads data. Existing output paths are refused
by the shared trainer. Federated baseline runners remain unchanged.
"""

import argparse
import json
from pathlib import Path

try:
    from . import main as training
except ImportError:
    import main as training


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "integer_tcn.json"


def build_parser(config_path=DEFAULT_CONFIG):
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if (config["model_name"] not in training.integer_tcn_config.INTEGER_MODEL_NAMES or config["simulation_steps"] != 1
            or config["spike_gradient"] != "clipped-ste"):
        raise ValueError("config must describe S=1 integer-tcn with clipped-ste")
    if (config["model_name"] == "integer-tcn-mamba2-interval"
            and config.get("interval_gradient") not in (
                "normalized-triangle-v1",
                "clipped-input-ste-plus-normalized-threshold-triangle-v1",
            )):
        raise ValueError("interval-Mamba config declares an unsupported interval gradient; expected normalized-triangle-v1 or clipped-input-ste-plus-normalized-threshold-triangle-v1")
    parser = training.build_parser()
    defaults = dict(config["architecture"], **config["training"])
    available = {action.dest for action in parser._actions}
    if set(defaults).difference(available):
        raise ValueError("unknown configuration keys: %s" % sorted(set(defaults) - available))
    parser.set_defaults(model_name=config["model_name"], **defaults)
    return parser


def resolve_training_argv(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, remainder = config_parser.parse_known_args(argv)
    parser = build_parser(config_args.config)
    args = parser.parse_args(remainder)
    if args.model_name not in training.integer_tcn_config.INTEGER_MODEL_NAMES:
        parser.error("this launcher only trains the integer-tcn model family")
    if args.training_source != "real":
        parser.error("this launcher requires real forgeries, not the SLN synthesis pipeline")
    if not args.run_tag:
        parser.error("--run-tag is required (existing checkpoints are never overwritten)")
    # Resolve defaults through the shared parser so this entry point and main.py
    # execute exactly the same training path.
    forwarded = ["--model-name", args.model_name]
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.dest in ("help", "model_name") or value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                forwarded.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                forwarded.append(option)
        else:
            forwarded.extend((option, str(value)))
    return forwarded


if __name__ == "__main__":
    training.main(resolve_training_argv())
