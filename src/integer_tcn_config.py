"""Shared CLI settings for the independent single-step integer TCN."""

MAMBA_MODEL_NAMES = ("integer-tcn-mamba2", "integer-tcn-mamba2-interval")
INTEGER_MODEL_NAMES = ("integer-tcn", *MAMBA_MODEL_NAMES)


def parse_dilations(value):
    if isinstance(value, str):
        value = value.split(",")
    result = tuple(int(item) for item in value)
    if not result or any(item < 1 for item in result):
        raise ValueError("TCN dilations must be a non-empty list of positive integers")
    return result


def add_integer_tcn_arguments(parser):
    group = parser.add_argument_group("single-step integer TCN (not inhibitory ILIF)")
    group.add_argument("--integer-max", type=int, default=4, help="maximum count D")
    group.add_argument("--tcn-width", type=int, default=128)
    group.add_argument("--tcn-kernel", type=int, default=5)
    group.add_argument("--tcn-dilations", type=str, default="1,2,4,8")
    group.add_argument("--tcn-embed-dim", type=int, default=256)
    group.add_argument("--tcn-residual-scale", type=float, default=0.1)
    group.add_argument("--mamba-layers", type=int, default=2)
    group.add_argument("--mamba-state-dim", type=int, default=16)
    group.add_argument("--mamba-expand", type=int, default=1)
    group.add_argument("--mamba-head-dim", type=int, default=32)
    group.add_argument("--mamba-conv-size", type=int, default=4)
    group.add_argument("--mamba-chunk-size", type=int, default=64)
    group.add_argument("--mamba-residual-scale", type=float, default=0.1)
    group.add_argument("--interval-sites", choices=["input", "output", "both"], default="both")
    group.add_argument("--interval-init-thresholds", default=None,
                        help="positive thresholds in input units, e.g. 0.5,1.5,2.5,3.5")
    group.add_argument("--interval-gap-min", type=float, default=0.05)
    group.add_argument("--interval-gap-max", type=float, default=2.0)
    group.add_argument("--interval-surrogate-width", type=float, default=1.0)
    group.add_argument("--interval-channelwise", action="store_true")
    group.add_argument("--interval-fixed", action="store_true", help="freeze gaps for a matched fixed-interval control")
    group.add_argument("--interval-tcn", action="store_true", help="use signed learnable intervals at all eight TCN quantizers")
    group.add_argument("--interval-input-gradient", choices=["normalized-triangle", "clipped-ste"], default="normalized-triangle")


def model_kwargs(args):
    options = dict(
        width=args.tcn_width,
        kernel_size=args.tcn_kernel,
        dilations=parse_dilations(args.tcn_dilations),
        embedding_dim=args.tcn_embed_dim,
        integer_max=args.integer_max,
        residual_scale=args.tcn_residual_scale,
    )
    if args.model_name in MAMBA_MODEL_NAMES:
        for name in ("mamba_layers", "mamba_state_dim", "mamba_expand", "mamba_head_dim",
                     "mamba_conv_size", "mamba_chunk_size", "mamba_residual_scale"):
            options[name] = getattr(args, name)
    if args.model_name == "integer-tcn-mamba2-interval":
        for name in ("interval_sites", "interval_init_thresholds", "interval_gap_min", "interval_gap_max",
                     "interval_surrogate_width", "interval_channelwise", "interval_fixed"):
            options[name] = getattr(args, name)
        options["interval_tcn"] = getattr(args, "interval_tcn", False)
        options["interval_input_gradient"] = getattr(args, "interval_input_gradient", "normalized-triangle")
    return options


def validate_integer_tcn_options(args):
    """Reject enabled legacy neuron options rather than silently ignoring them."""
    incompatible = []
    named = {
        "comparison_neuron", "clif_tau", "clif_temporal", "temporal_tau",
        "temporal_tau_channelwise", "tau_psn", "tau_channelwise",
        "rollout_repeated_input", "dual_branch_readout", "midlevel_forgery_margin",
        "learnable_l5_thresholds", "learnable_l45_thresholds",
        "learnable_all_thresholds", "fed_consensus_path",
    }
    for name, value in vars(args).items():
        # burst_k has a legacy parser default even when Burst is disabled.
        is_burst = name.startswith("burst_") and name != "burst_k"
        if (name in named or name.endswith("_plif") or is_burst) and value:
            incompatible.append("--" + name.replace("_", "-"))
    if incompatible:
        raise ValueError("integer-tcn does not support " + ", ".join(incompatible))
