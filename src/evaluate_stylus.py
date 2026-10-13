# -*- coding:utf-8 -*-
"""加载集中式/联邦模型权重，并为各签名评估集导出验证特征。

本脚本本身不计算 EER。它负责按训练配置重建模型、严格加载 state_dict，
然后导出普通验证所需的嵌入，以及可选的 TVS 发放率/发放时间特征；
后续由 verifier_stylus_*.py 根据这些导出文件计算 SF、RF 或 TVS EER。
"""

import argparse
import json
import os
import pickle

import numpy

try:
    from . import integer_tcn_config
except ImportError:
    import integer_tcn_config


def _has_spike_neurons(model):
    """判断当前模型是否暴露了可供 TVS 注册钩子的脉冲神经元列表。"""
    return hasattr(model, "spike_neurons") and len(model.spike_neurons) > 0


def _normalize_valid_lengths(spike_np, valid_lengths=None):
    """把有效长度整理为 [B]，并限制在当前脉冲张量的时间范围内。"""
    batch_size, _, total_steps = spike_np.shape
    if valid_lengths is None:
        return numpy.full(batch_size, total_steps, dtype=numpy.int64)
    lengths = numpy.asarray(valid_lengths, dtype=numpy.int64).reshape(-1)
    if lengths.size != batch_size:
        raise ValueError(
            "valid_lengths size must match spike batch size: %d != %d"
            % (lengths.size, batch_size)
        )
    return numpy.clip(lengths, 0, total_steps)


def _build_valid_time_mask(spike_np, valid_lengths=None):
    """根据每条签名自身的有效长度生成 [B, T] 布尔掩码。"""
    lengths = _normalize_valid_lengths(spike_np, valid_lengths)
    steps = numpy.arange(spike_np.shape[2], dtype=numpy.int64)[None, :]
    return steps < lengths[:, None]


def _downsample_valid_lengths(raw_lengths, num_pools, max_steps):
    """按照 ceil-mode 二倍池化次数，将原始长度换算到某个脉冲层。"""
    lengths = numpy.asarray(raw_lengths, dtype=numpy.int64).reshape(-1)
    for _ in range(int(num_pools)):
        lengths = (lengths + 1) // 2
    return numpy.clip(lengths, 0, int(max_steps))


def _spike_valid_lengths(model, raw_lengths, layer_idx, max_steps):
    if hasattr(model, "spike_valid_lengths"):
        return model.spike_valid_lengths(raw_lengths, layer_idx, max_steps)
    return _downsample_valid_lengths(raw_lengths, layer_idx // 2 + 1, max_steps)


def _masked_spike_rate(spike_np, valid_lengths=None):
    """只在每条签名的有效时间步内计算逐通道平均发放率。"""
    valid_mask = _build_valid_time_mask(spike_np, valid_lengths)
    valid = valid_mask[:, None, :].astype(spike_np.dtype, copy=False)
    denominator = numpy.maximum(valid.sum(axis=2), 1.0)
    return (spike_np * valid).sum(axis=2) / denominator


def _compress_interval_profile(spike_np, starts, ends, num_bins):
    """将每个样本各自的 [start, end) 有效区间压缩为固定数量时间箱。"""
    batch_size = spike_np.shape[0]
    bins = numpy.zeros((batch_size, num_bins), dtype=numpy.float32)
    for sample_idx in range(batch_size):
        interval_start = int(starts[sample_idx])
        interval_end = int(ends[sample_idx])
        interval_length = interval_end - interval_start
        if interval_length <= 0:
            continue
        bin_size = interval_length / float(num_bins)
        for bin_idx in range(num_bins):
            start = interval_start + int(round(bin_idx * bin_size))
            end = interval_start + int(round((bin_idx + 1) * bin_size))
            start = min(start, interval_end - 1)
            end = min(max(end, start + 1), interval_end)
            # 每个时间箱对通道和有效箱内时间求均值；补零区不会进入统计。
            bins[sample_idx, bin_idx] = spike_np[
                sample_idx, :, start:end
            ].mean()
    return bins


def _compress_time_profile(spike_np, num_bins=16, valid_lengths=None):
    """按每条签名自身的有效区间生成固定维度脉冲时间描述子。"""
    lengths = _normalize_valid_lengths(spike_np, valid_lengths)
    starts = numpy.zeros_like(lengths)
    return _compress_interval_profile(spike_np, starts, lengths, num_bins)


def _pad_variable_length_sequences(seq_feats_list, seq_mask_list):
    """将不同批次的变长预池化序列补零到统一的最大时间长度。"""
    max_T = max(f.shape[2] for f in seq_feats_list)
    B = sum(f.shape[0] for f in seq_feats_list)
    C = seq_feats_list[0].shape[1]

    padded_feats = numpy.zeros((B, C, max_T), dtype=seq_feats_list[0].dtype)
    padded_mask = numpy.zeros((B, max_T), dtype=seq_mask_list[0].dtype)
    offset = 0
    for f, m in zip(seq_feats_list, seq_mask_list):
        T = f.shape[2]
        padded_feats[offset : offset + f.shape[0], :, :T] = f
        padded_mask[offset : offset + f.shape[0], :T] = m[:, :T]
        offset += f.shape[0]

    return padded_feats, padded_mask


def _pad_spike_batches(spike_batches):
    """将不同批次的 [B, C, T] 脉冲补零后沿样本维拼接。"""
    if not spike_batches:
        raise ValueError("spike_batches must not be empty")
    max_steps = max(batch.shape[2] for batch in spike_batches)
    total_samples = sum(batch.shape[0] for batch in spike_batches)
    channels = spike_batches[0].shape[1]
    padded = numpy.zeros(
        (total_samples, channels, max_steps), dtype=spike_batches[0].dtype
    )
    offset = 0
    for batch in spike_batches:
        if batch.shape[1] != channels:
            raise ValueError("spike channel count changed between batches")
        batch_size, _, steps = batch.shape
        padded[offset : offset + batch_size, :, :steps] = batch
        offset += batch_size
    return padded


def _compress_time_profile_earlylate(
    spike_np, num_bins_per_half=8, valid_lengths=None
):
    """按各自有效长度分为前/后半段，并分别生成固定维度时间描述子。"""
    lengths = _normalize_valid_lengths(spike_np, valid_lengths)
    midpoints = lengths // 2
    starts = numpy.zeros_like(lengths)
    bins_early = _compress_interval_profile(
        spike_np, starts, midpoints, num_bins_per_half
    )
    bins_late = _compress_interval_profile(
        spike_np, midpoints, lengths, num_bins_per_half
    )
    return bins_early, bins_late


def _accumulate_signed_count_diagnostics(state, spike_np, valid_lengths=None):
    """累计 signed-burst 的正、负脉冲幅值，仅用于诊断而不参与 EER。"""
    B = spike_np.shape[0]
    valid_mask = _build_valid_time_mask(spike_np, valid_lengths)[:, None, :]
    pos_mask = (spike_np > 0) & valid_mask
    neg_mask = (spike_np < 0) & valid_mask
    pos_sum = float(spike_np[pos_mask].sum()) if pos_mask.any() else 0.0
    neg_sum = float(numpy.abs(spike_np[neg_mask]).sum())
    state["pos"] += pos_sum
    state["neg"] += neg_sum
    state["count"] += B


def _finalize_signed_count_diagnostics(state):
    count = max(state["count"], 1)
    return {
        "mean_pos": float(state["pos"]) / count,
        "mean_neg": float(state["neg"]) / count,
        "num_samples": state["count"],
    }


def _build_spike_export_result(
    feats,
    labels,
    inds,
    spike_rates=None,
    spike_times=None,
    seq_feats=None,
    seq_mask=None,
    spike_meta=None,
    signed_l5_diagnostics=None,
):
    """统一组织嵌入、标签以及可选脉冲/序列特征的返回字典。"""
    result = {"feats": feats, "labels": labels, "inds": inds}
    if spike_rates is not None:
        result["spike_rates"] = spike_rates
    if spike_times is not None:
        result["spike_times"] = spike_times
    if seq_feats is not None:
        result["seq_feats"] = seq_feats
    if seq_mask is not None:
        result["seq_mask"] = seq_mask
    if spike_meta is not None:
        result["spike_meta"] = spike_meta
    if signed_l5_diagnostics is not None:
        result["signed_l5_diagnostics"] = signed_l5_diagnostics
    return result


def build_parser():
    # 这里的结构类参数必须与训练时一致；评估脚本不会从普通 state_dict
    # 自动推断所有布尔开关，因此调用方需要显式传入对应配置。
    parser = argparse.ArgumentParser(description="Online signature verification")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="directory containing dataset pickle files",
    )
    parser.add_argument("--data-source", choices=["auto", "legacy", "real"], default="auto",
                        help="auto detects the isolated real-data manifest; real never uses SLN")
    parser.add_argument("--eval-split", choices=["evaluation", "validation"], default="evaluation",
                        help="validation is supported only by the isolated real data pipeline")
    parser.add_argument(
        "--train-shot-g",
        type=int,
        default=5,
        help="number of genuine samples per class per training batch(default: 5)",
    )
    parser.add_argument(
        "--train-shot-f",
        type=int,
        default=5,
        help="number of forgery samples per class per training batch(default: 5)",
    )
    parser.add_argument(
        "--train-tasks", type=int, default=1, help="number of tasks per batch"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="number of workers for DataLoader"
    )
    parser.add_argument(
        "--seed", type=int, default=333, help="numpy random seed (default: 333)"
    )
    parser.add_argument(
        "--epoch", type=str, default="End", help="model from the i-th epoch for testing"
    )
    parser.add_argument(
        "--eval-datasets",
        type=str,
        default="mcyt,bsid,bsds2,ebio1,ebio2",
        help=(
            "comma-separated evaluation datasets; supported: "
            "mcyt,bsid,bsds2,ebio1,ebio2,msds,gpds"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "explicit model state_dict path; when provided it overrides the legacy "
            "models/<run-tag>/<seed>/epoch<epoch> lookup"
        ),
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help=(
            "optional full client checkpoint supplying personalized classifier "
            "parameters when --checkpoint is a classifier-free global_shared.pt"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn",
        choices=["cnn", "snn", *integer_tcn_config.INTEGER_MODEL_NAMES],
        help="model backend to evaluate",
    )
    integer_tcn_config.add_integer_tcn_arguments(parser)
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=1024,
        help="embedding dimension for the SNN readout head",
    )
    parser.add_argument(
        "--clif-tau",
        action="store_true",
        help="replace tau-only spike neurons with CLIF-style complementary-memory tau neurons",
    )
    parser.add_argument(
        "--clif-temporal",
        action="store_true",
        help="enable the temporalized CLIF experiment branch for SNN neurons",
    )
    parser.add_argument(
        "--temporal-tau",
        action="store_true",
        help="enable the temporal tau experiment branch for SNN neurons",
    )
    parser.add_argument(
        "--tau-psn",
        action="store_true",
        help="enable the parameter-space noise (PSN) branch for learned SNN tau parameters",
    )
    parser.add_argument(
        "--temporal-tau-channelwise",
        action="store_true",
        help="enable the channelwise temporal tau experiment branch for SNN neurons",
    )
    parser.add_argument(
        "--tau-channelwise",
        action="store_true",
        help="enable the single-pass per-channel tau pointwise PLIF branch for SNN neurons",
    )
    parser.add_argument(
        "--comparison-neuron",
        choices=[
            "mlf-k3",
            "ilif",
            "klif",
            "scalar-plif",
            "sliding-psn-k2",
        ],
        default=None,
        help="optional neuron-only baseline used by the matching checkpoint",
    )
    parser.add_argument(
        "--comparison-psn-order",
        type=int,
        default=2,
        help=(
            "SlidingPSN-k2 的因果滑动时间核长度；正式 baseline 固定为 2，"
            "即使用当前与前一时刻"
        ),
    )
    parser.add_argument(
        "--rctm-channelwise-plif",
        action="store_true",
        help="enable the residual causal temporal mix plus per-channel tau pointwise PLIF branch for SNN neurons",
    )
    parser.add_argument(
        "--sgr-channelwise-plif",
        action="store_true",
        help="enable the spike-gated reset-message plus per-channel tau pointwise PLIF branch for SNN neurons",
    )
    parser.add_argument(
        "--full-history-sgr-channelwise-plif",
        action="store_true",
        help="enable the full-history spike-gated reset-message plus per-channel tau pointwise PLIF branch for SNN neurons",
    )
    parser.add_argument(
        "--full-history-sgr-visible-norm-channelwise-plif",
        action="store_true",
        help="visible-history-normalized full-history spike-gated reset-message plus per-channel tau pointwise PLIF branch",
    )
    parser.add_argument(
        "--full-history-sgr-learnable-decay-channelwise-plif",
        action="store_true",
        help="learnable-decay full-history spike-gated reset-message plus per-channel tau pointwise PLIF branch",
    )
    parser.add_argument(
        "--st-sgr-full-history-channelwise-plif",
        action="store_true",
        help="ST-SGR full-history dynamic-threshold channelwise PLIF branch",
    )
    parser.add_argument(
        "--gated-mode-full-history-sgr-channelwise-plif",
        action="store_true",
        help="Gated-Mode full-history SGR channelwise PLIF branch",
    )
    parser.add_argument(
        "--cc-full-history-sgr-channelwise-plif",
        action="store_true",
        help="Cross-Channel full-history SGR channelwise PLIF branch",
    )
    parser.add_argument(
        "--dt-full-history-sgr-channelwise-plif",
        action="store_true",
        help="Dual-Timescale full-history SGR channelwise PLIF branch",
    )
    parser.add_argument(
        "--per-channel-learnable-decay-full-history-sgr-channelwise-plif",
        action="store_true",
        help="Per-Channel Learnable-Decay full-history SGR channelwise PLIF branch",
    )
    parser.add_argument(
        "--multi-state-full-history-sgr-channelwise-plif",
        action="store_true",
        help="Multi-State full-history SGR channelwise PLIF branch",
    )
    parser.add_argument(
        "--sn-width-scale",
        type=float,
        default=1.0,
        help="width scale for SNN backbone (default: 1.0)",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=4,
        help="number of internal SNN simulation steps for the restored repeated-input rollout",
    )
    parser.add_argument(
        "--rollout-repeated-input",
        action="store_true",
        help="enable the dedicated repeated-input rollout branch for SNN experiments",
    )
    parser.add_argument(
        "--dual-branch-readout",
        action="store_true",
        help="enable the dual-branch readout head for SNN experiments",
    )
    parser.add_argument(
        "--burst-spike", action="store_true", help="enable burst spike neuron"
    )
    parser.add_argument(
        "--burst-k", type=int, default=2, help="burst level count (default: 2)"
    )
    parser.add_argument(
        "--burst-slots", action="store_true", help="use binary burst slots"
    )
    parser.add_argument(
        "--burst-raw-count", action="store_true", help="output raw spike count"
    )
    parser.add_argument(
        "--burst-k-per-layer", type=str, default=None, help="per-layer burst K"
    )
    parser.add_argument(
        "--burst-thresholds", type=str, default=None, help="non-uniform thresholds"
    )
    parser.add_argument(
        "--burst-thresholds-per-layer",
        type=str,
        default=None,
        help="per-layer thresholds",
    )
    parser.add_argument(
        "--burst-signed-l5",
        action="store_true",
        help="enable signed-count burst diagnostics for L5 layer",
    )
    parser.add_argument(
        "--learnable-l5-thresholds",
        action="store_true",
        help="make L5 burst thresholds learnable (must match training config)",
    )
    parser.add_argument(
        "--learnable-l45-thresholds",
        action="store_true",
        help="make L4 and L5 burst thresholds learnable (must match training config)",
    )
    parser.add_argument(
        "--learnable-all-thresholds",
        action="store_true",
        help="make all burst thresholds learnable (must match training config)",
    )
    parser.add_argument(
        "--midlevel-forgery-margin",
        action="store_true",
        help="enable mid-level forgery margin loss term",
    )
    parser.add_argument(
        "--lambda-midlevel-margin",
        type=float,
        default=0.0,
        help="weight for midlevel forgery margin loss (default: 0.0)",
    )
    parser.add_argument(
        "--export-snn-tvs",
        action="store_true",
        help="export SNN spike-rate and spike-time features for SNN-TVS verifier",
    )
    parser.add_argument(
        "--export-tag",
        type=str,
        default=None,
        help="optional export tag for dedicated output directory (falls back to --run-tag)",
    )
    parser.add_argument(
        "--export-time-view",
        type=str,
        default="global",
        choices=["global", "earlylate"],
        help="spike-time profile view: global or earlylate (default: global)",
    )
    parser.add_argument(
        "--export-all-layer-spike",
        action="store_true",
        help="export spike features from all SNN layers, not just the last",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="optional run tag for non-overwriting checkpoint lookup paths",
    )
    parser.add_argument(
        "--export-seq-feats",
        action="store_true",
        help="export pre-pooling feature sequences for DTW verifier",
    )
    return parser


def _import_build_model():
    if __package__:
        from .model_factory import build_model
    else:
        from model_factory import build_model
    return build_model


def _import_dataset_module():
    if __package__:
        from .dataset import datasetTest_SF as dataset
    else:
        import dataset.datasetTest_SF as dataset
    return dataset


def _data_path(args, filename):
    return os.path.join(args.data_root, filename)


def _uses_real_data(args):
    source = getattr(args, "data_source", "auto")
    has_manifest = os.path.isfile(os.path.join(args.data_root, "manifest.json"))
    if source == "legacy" and has_manifest:
        raise ValueError("isolated real data must use its matching real feature extractor")
    return source == "real" or (source == "auto" and has_manifest)


def _load_evaluation_signatures(args, dataset_key):
    if _uses_real_data(args):
        if __package__:
            from .real_forgery_data import load_split
        else:
            from real_forgery_data import load_split
        payload, _ = load_split(args.data_root, dataset_key.lower(),
                                getattr(args, "eval_split", "evaluation"))
        return payload
    if getattr(args, "eval_split", "evaluation") != "evaluation":
        raise ValueError("validation split requires the isolated real-forgery pipeline")
    with open(_data_path(args, "%s_eva.pkl" % dataset_key), "rb") as handle:
        return pickle.load(handle, encoding="iso-8859-1")


def get_num_classes(weight_dict):
    """从分类头权重的第一维恢复训练时类别数，用于重建完全同形的模型。"""
    for key in ("cls.weight", "classifier.weight"):
        if key in weight_dict:
            weights = weight_dict[key]
            if hasattr(weights, "shape"):
                return weights.shape[0]
            return len(weights)
    raise KeyError("Could not determine classifier weights from checkpoint")


def build_evaluation_model(args, dset, weight_dict):
    """使用评估数据维度和训练配置重建与 checkpoint 严格匹配的网络。"""
    if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES:
        integer_tcn_config.validate_integer_tcn_options(args)
        if __package__:
            from .network_integer_tcn import IntegerTCN
            from .network_integer_tcn_mamba import IntegerTCNMamba2
            from .network_interval_mamba import IntervalTCNMamba2
        else:
            from network_integer_tcn import IntegerTCN
            from network_integer_tcn_mamba import IntegerTCNMamba2
            from network_interval_mamba import IntervalTCNMamba2
        # Includes non-tensor settings D and dilation: shapes alone cannot detect
        # a wrong quantizer or temporal receptive field. CLI defaults do not win.
        model_type = {"integer-tcn": IntegerTCN, "integer-tcn-mamba2": IntegerTCNMamba2,
                      "integer-tcn-mamba2-interval": IntervalTCNMamba2}[args.model_name]
        return model_type.from_state_dict_config(weight_dict, n_in=dset.featDim)
    build_model = _import_build_model()
    common_kwargs = dict(
        # 输入维度来自评估集的项目特征，类别数来自权重而不是评估集用户数。
        n_in=dset.featDim,
        n_classes=get_num_classes(weight_dict),
    )
    if args.model_name == "cnn":
        common_kwargs.update(
            dict(
                n_task=args.train_tasks,
                n_shot_g=args.train_shot_g,
                n_shot_f=args.train_shot_f,
            )
        )
    else:
        common_kwargs.update(
            dict(
                n_task=args.train_tasks,
                n_shot_g=args.train_shot_g,
                n_shot_f=args.train_shot_f,
                embedding_dim=args.embed_dim,
                sim_steps=getattr(args, "sim_steps", 4),
                rollout_repeated_input=getattr(args, "rollout_repeated_input", False),
                width_scale=float(getattr(args, "sn_width_scale", 1.0)),
                burst_spike=getattr(args, "burst_spike", False),
                burst_k=int(getattr(args, "burst_k", 2)),
                burst_slots=getattr(args, "burst_slots", False),
                burst_raw_count=getattr(args, "burst_raw_count", False),
                burst_k_per_layer=getattr(args, "burst_k_per_layer", None),
                burst_thresholds=getattr(args, "burst_thresholds", None),
                burst_thresholds_per_layer=getattr(
                    args, "burst_thresholds_per_layer", None
                ),
                burst_signed_l5=getattr(args, "burst_signed_l5", False),
                learnable_l5_thresholds=getattr(args, "learnable_l5_thresholds", False),
                learnable_l45_thresholds=getattr(
                    args, "learnable_l45_thresholds", False
                ),
                learnable_all_thresholds=getattr(
                    args, "learnable_all_thresholds", False
                ),
                comparison_neuron=getattr(args, "comparison_neuron", None),
                # SlidingPSN-k2 固定使用长度为 2 的因果时间核。
                # 它与卷积层数、网络层数以及评估种子数量均无关。
                comparison_psn_order=getattr(args, "comparison_psn_order", 2),
                midlevel_forgery_margin=getattr(args, "midlevel_forgery_margin", False),
                lambda_midlevel_margin=float(
                    getattr(args, "lambda_midlevel_margin", 0.0)
                ),
                dual_branch_readout=getattr(args, "dual_branch_readout", False),
            )
        )
        if getattr(args, "clif_tau", False):
            common_kwargs["clif_tau"] = True
        if getattr(args, "clif_temporal", False):
            common_kwargs["clif_temporal"] = True
    if getattr(args, "temporal_tau", False):
        common_kwargs["temporal_tau"] = True
    if getattr(args, "tau_psn", False):
        common_kwargs["tau_psn"] = True
    if getattr(args, "temporal_tau_channelwise", False):
        common_kwargs["temporal_tau_channelwise"] = True
    if getattr(args, "tau_channelwise", False):
        common_kwargs["tau_channelwise"] = True
    if getattr(args, "rctm_channelwise_plif", False):
        common_kwargs["rctm_channelwise_plif"] = True
    if getattr(args, "sgr_channelwise_plif", False):
        common_kwargs["sgr_channelwise_plif"] = True
    if getattr(args, "full_history_sgr_channelwise_plif", False):
        common_kwargs["full_history_sgr_channelwise_plif"] = True
    if getattr(args, "full_history_sgr_visible_norm_channelwise_plif", False):
        common_kwargs["full_history_sgr_visible_norm_channelwise_plif"] = True
    if getattr(args, "full_history_sgr_learnable_decay_channelwise_plif", False):
        common_kwargs["full_history_sgr_learnable_decay_channelwise_plif"] = True
    if getattr(args, "st_sgr_full_history_channelwise_plif", False):
        common_kwargs["st_sgr_full_history_channelwise_plif"] = True
    if getattr(args, "gated_mode_full_history_sgr_channelwise_plif", False):
        common_kwargs["gated_mode_full_history_sgr_channelwise_plif"] = True
    if getattr(args, "cc_full_history_sgr_channelwise_plif", False):
        common_kwargs["cc_full_history_sgr_channelwise_plif"] = True
    if getattr(args, "dt_full_history_sgr_channelwise_plif", False):
        common_kwargs["dt_full_history_sgr_channelwise_plif"] = True
    if getattr(
        args, "per_channel_learnable_decay_full_history_sgr_channelwise_plif", False
    ):
        common_kwargs[
            "per_channel_learnable_decay_full_history_sgr_channelwise_plif"
        ] = True
    if getattr(args, "multi_state_full_history_sgr_channelwise_plif", False):
        common_kwargs["multi_state_full_history_sgr_channelwise_plif"] = True
    return build_model(args.model_name, **common_kwargs)


def extract_embedding(model_output):
    """从模型前向返回值中提取第一个元素，即用于签名匹配的嵌入向量。"""
    if not isinstance(model_output, tuple):
        raise ValueError(
            "Model output must be a tuple of (embedding, logits) or (embedding, logits, aux)"
        )
    if len(model_output) not in (2, 3):
        raise ValueError(
            "Model output must be a tuple of (embedding, logits) or (embedding, logits, aux)"
        )
    return model_output[0]


def get_checkpoint_path(args):
    """显式 --checkpoint 优先，否则兼容旧实验的目录命名方式。"""
    if getattr(args, "checkpoint", None):
        return os.path.abspath(args.checkpoint)
    if args.run_tag:
        return "models/%s/%d/epoch%s" % (args.run_tag, args.seed, args.epoch)
    return "models/%d/epoch%s" % (args.seed, args.epoch)


def load_checkpoint_state(torch, checkpoint_path):
    """Load a plain state_dict from legacy or federated checkpoints."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("checkpoint does not exist: %s" % checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:  # PyTorch < 1.13 has no weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 同时兼容“直接保存 state_dict”和“外层字典包含 state_dict”两种格式。
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(
            "checkpoint must contain a non-empty model state_dict: %s"
            % checkpoint_path
        )
    return checkpoint


def get_export_dir(args, dataset_name):
    """为不同方法和种子建立隔离目录，避免覆盖其他实验的特征。"""
    tag = getattr(args, "export_tag", None) or getattr(args, "run_tag", None)
    if tag:
        return "exps/%s/seed%d/%s" % (tag, args.seed, dataset_name)
    return "exps/seed%d/%s" % (args.seed, dataset_name)


def get_runtime_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_dataset_features(
    model,
    data_loader,
    dset,
    save_dir,
    torch,
    Variable,
    export_spikes=False,
    export_seq_feats=False,
    export_all_layers=False,
    export_time_view="global",
    midlevel_forgery_margin_kwargs=None,
    dual_branch_readout=False,
    collect_signed_diagnostics=False,
):
    """遍历一个评估集并收集嵌入，以及按需收集 TVS/DTW 辅助特征。"""
    feats = []
    spike_rate_feats = [] if export_spikes else None
    spike_time_feats = [] if export_spikes else None
    spike_hooks = []
    seq_feats_list = [] if export_seq_feats else None
    seq_mask_list = [] if export_seq_feats else None
    seq_hook = None
    all_layer_spike_batches = [] if (export_spikes and export_all_layers) else None
    spike_meta = None
    layer_rate_dims = None
    layer_time_dims = None
    signed_diag_state = (
        dict(pos=0.0, neg=0.0, count=0) if collect_signed_diagnostics else None
    )

    if export_seq_feats:
        if not hasattr(model, "sp2"):
            raise ValueError(
                "--export-seq-feats requires a model with sp2 pooling layer"
            )

        seq_captured = []

        def _capture_prepooling(module, inputs, output):
            # sp2 的输入依次是预池化特征和有效时间掩码；转到 CPU 后立即脱离计算图。
            seq_captured.append(
                (
                    inputs[0].detach().cpu().numpy(),
                    inputs[1].detach().cpu().numpy(),
                )
            )

        seq_hook = model.sp2.register_forward_hook(_capture_prepooling)

    if export_spikes:
        spike_outputs = {}

        def _make_spike_hook(idx):
            def hook(_, input, output):
                spike_outputs[idx] = output.detach()

            return hook

        # 前向钩子不改变神经元输出，只把各层脉冲暂存下来供 TVS 特征统计。
        for i, neuron in enumerate(model.spike_neurons):
            spike_hooks.append(neuron.register_forward_hook(_make_spike_hook(i)))

    device = next(model.parameters()).device
    for batch in data_loader:
        sig, lens, label = batch
        if export_spikes:
            # 防止某层在异常分支中未执行时误用上一批次残留的钩子输出。
            spike_outputs.clear()
        raw_lengths = numpy.asarray(lens, dtype=numpy.int64)
        # 输出掩码由真实序列长度推导，用于排除补零时间步对池化的影响。
        mask = model.getOutputMask(lens)

        sig = Variable(torch.from_numpy(sig)).to(device)
        mask = Variable(torch.from_numpy(mask)).to(device)
        lens = Variable(torch.from_numpy(lens)).to(device)

        # 前向输出中的 embedding 用于普通 SF/RF；分类 logits 在评估导出时不用。
        # no_grad 禁止构建反向图，降低六层脉冲钩子同时工作时的显存与时间开销。
        with torch.no_grad():
            output = extract_embedding(model(sig, mask, lens))
        output = output.detach().cpu().numpy()
        feats.append(output)

        if export_spikes:
            num_spike_layers = len(model.spike_neurons)
            missing_layers = [
                layer_idx
                for layer_idx in range(num_spike_layers)
                if layer_idx not in spike_outputs
            ]
            if missing_layers:
                raise RuntimeError(
                    "spike hooks captured no output for layers: %s" % missing_layers
                )

            # 默认 TVS 仅使用末层；显式开启全层导出时，将六层描述子按层拼接。
            selected_layers = (
                list(range(num_spike_layers))
                if export_all_layers
                else [num_spike_layers - 1]
            )
            batch_rates = []
            batch_times = []
            current_rate_dims = []
            current_time_dims = []
            layer_spike_np = [] if export_all_layers else None

            for layer_idx in selected_layers:
                spike_np = spike_outputs[layer_idx].cpu().numpy()
                # 当前六层主干每两个脉冲层共享一次时间下采样：L0/1、L2/3、L4/5
                # 分别对应经过 1、2、3 次 ceil-mode 二倍池化后的有效长度。
                valid_lengths = _spike_valid_lengths(
                    model, raw_lengths, layer_idx, max_steps=spike_np.shape[2]
                )
                rate = _masked_spike_rate(spike_np, valid_lengths=valid_lengths)
                if export_time_view == "earlylate":
                    early_bins, late_bins = _compress_time_profile_earlylate(
                        spike_np,
                        num_bins_per_half=8,
                        valid_lengths=valid_lengths,
                    )
                    time_profile = numpy.concatenate(
                        [early_bins, late_bins], axis=1
                    )
                else:
                    time_profile = _compress_time_profile(
                        spike_np, num_bins=16, valid_lengths=valid_lengths
                    )
                batch_rates.append(rate)
                batch_times.append(time_profile)
                current_rate_dims.append(rate.shape[1])
                current_time_dims.append(time_profile.shape[1])
                if export_all_layers:
                    layer_spike_np.append(spike_np)

            if layer_rate_dims is None:
                layer_rate_dims = tuple(current_rate_dims)
                layer_time_dims = tuple(current_time_dims)
            elif (
                tuple(current_rate_dims) != layer_rate_dims
                or tuple(current_time_dims) != layer_time_dims
            ):
                raise RuntimeError("spike descriptor dimensions changed between batches")

            spike_rate_feats.append(numpy.concatenate(batch_rates, axis=1))
            spike_time_feats.append(numpy.concatenate(batch_times, axis=1))

            if export_all_layers:
                all_layer_spike_batches.append(layer_spike_np)

            if signed_diag_state is not None:
                final_layer_idx = num_spike_layers - 1
                final_spike_np = spike_outputs[final_layer_idx].cpu().numpy()
                final_valid_lengths = _spike_valid_lengths(
                    model, raw_lengths, final_layer_idx,
                    max_steps=final_spike_np.shape[2],
                )
                _accumulate_signed_count_diagnostics(
                    signed_diag_state,
                    final_spike_np,
                    valid_lengths=final_valid_lengths,
                )

        if export_seq_feats:
            f, m = seq_captured[-1]
            seq_feats_list.append(f)
            seq_mask_list.append(m)
            del seq_captured[:]

    if export_spikes:
        for h in spike_hooks:
            h.remove()
    if seq_hook is not None:
        seq_hook.remove()

    feats = numpy.concatenate(feats, axis=0)
    # datasetTest_SF 按“每位用户一批”且按 genuine→forgery 排列样本，
    # 因此可按固定顺序重建用户标签和真伪标记；这里依赖该数据集/采样器约定。
    labels = numpy.kron(
        numpy.arange(0, len(dset.testKeys)), numpy.ones(dset.accumNumNeg)
    )
    inds = numpy.ones(dset.accumNumNeg)
    inds[dset.numGen :] = 0
    inds = numpy.kron(numpy.ones(len(dset.testKeys)), inds)

    os.makedirs(save_dir, exist_ok=True)

    if export_spikes:
        spike_rates = numpy.concatenate(spike_rate_feats, axis=0)
        # TVS 在后续进行余弦式距离计算，因此两类脉冲描述子在保存前分别做 L2 归一化。
        spike_rates = spike_rates / (
            numpy.sum(spike_rates**2, axis=1, keepdims=True) ** 0.5 + 1e-8
        )
        spike_times = numpy.concatenate(spike_time_feats, axis=0)
        spike_times = spike_times / (
            numpy.sum(spike_times**2, axis=1, keepdims=True) ** 0.5 + 1e-8
        )

        if export_all_layers and all_layer_spike_batches:
            num_layers = len(all_layer_spike_batches[0])
            layer_shapes = []
            layer_list_dict = {li: [] for li in range(num_layers)}
            for batch_layers in all_layer_spike_batches:
                for li in range(num_layers):
                    layer_list_dict[li].append(batch_layers[li])
            layer_spikes = {}
            for li in range(num_layers):
                # 不同用户批次的最大签名长度不同，原始层脉冲需先补到统一长度再拼接。
                concat_spikes = _pad_spike_batches(layer_list_dict[li])
                layer_spikes[li] = concat_spikes
                layer_shapes.append(concat_spikes.shape)
            spike_meta = {
                "rate_dims": numpy.asarray(layer_rate_dims, dtype=numpy.int64),
                "time_dims": numpy.asarray(layer_time_dims, dtype=numpy.int64),
                "layer_shapes": numpy.asarray(layer_shapes, dtype=numpy.int64),
            }

        signed_l5_result = None
        if signed_diag_state is not None:
            signed_l5_result = _finalize_signed_count_diagnostics(signed_diag_state)

        result = _build_spike_export_result(
            feats=feats,
            labels=labels,
            inds=inds,
            spike_rates=spike_rates,
            spike_times=spike_times,
            spike_meta=spike_meta,
            signed_l5_diagnostics=signed_l5_result,
        )
        if export_seq_feats:
            seq_feats_np, seq_mask_np = _pad_variable_length_sequences(
                seq_feats_list, seq_mask_list
            )
            result["seq_feats"] = seq_feats_np
            result["seq_mask"] = seq_mask_np
        if export_all_layers:
            result["all_layer_spikes"] = layer_spikes
        return result

    if export_seq_feats:
        seq_feats_np, seq_mask_np = _pad_variable_length_sequences(
            seq_feats_list, seq_mask_list
        )
        return _build_spike_export_result(
            feats=feats,
            labels=labels,
            inds=inds,
            seq_feats=seq_feats_np,
            seq_mask=seq_mask_np,
        )

    return _build_spike_export_result(feats=feats, labels=labels, inds=inds)


def export_split(
    model,
    dataset_module,
    torch,
    Variable,
    DataLoader,
    args,
    dataset_name,
    dataset_key,
    sig_dict=None,
):
    """加载一个评估数据集，运行特征导出并保存为 verifier 约定的文件名。"""
    if sig_dict is None:
        sig_dict = _load_evaluation_signatures(args, dataset_key)
    if not sig_dict:
        raise ValueError("evaluation split %s is empty" % dataset_name)
    # datasetTest_SF 会保持每个用户内部“真实签名在前、伪造签名在后”的顺序。
    dset = dataset_module.dataset(sigDict=sig_dict)
    sampler = dataset_module.batchSampler(dset)
    data_loader = DataLoader(
        dset,
        num_workers=args.workers,
        batch_sampler=sampler,
        collate_fn=dataset_module.collate_fn,
    )
    save_dir = get_export_dir(args, dataset_name)
    export_seq_feats = getattr(args, "export_seq_feats", False)
    export_all_layers = getattr(args, "export_all_layer_spike", False)
    export_time_view = getattr(args, "export_time_view", "global")
    collect_signed = getattr(args, "burst_signed_l5", False)
    midlevel_kwargs = (
        dict(
            midlevel_forgery_margin=getattr(args, "midlevel_forgery_margin", False),
            lambda_midlevel_margin=getattr(args, "lambda_midlevel_margin", 0.0),
        )
        if getattr(args, "midlevel_forgery_margin", False)
        else None
    )
    dual_branch = getattr(args, "dual_branch_readout", False)

    # 只有 SNN 且显式要求 TVS 时才导出发放率/发放时间；CNN 或普通导出只保存嵌入。
    if getattr(args, "export_snn_tvs", False) and _has_spike_neurons(model):
        result = export_dataset_features(
            model,
            data_loader,
            dset,
            save_dir,
            torch,
            Variable,
            export_spikes=True,
            export_seq_feats=export_seq_feats,
            export_all_layers=export_all_layers,
            export_time_view=export_time_view,
            midlevel_forgery_margin_kwargs=midlevel_kwargs,
            dual_branch_readout=dual_branch,
            collect_signed_diagnostics=collect_signed,
        )
        if isinstance(result, dict):
            feats = result["feats"]
            labels = result["labels"]
            inds = result["inds"]
            spike_rates = result.get("spike_rates")
            spike_times = result.get("spike_times")
            seq_feats = result.get("seq_feats")
            seq_mask = result.get("seq_mask")
            spike_meta = result.get("spike_meta")
            signed_l5_diagnostics = result.get("signed_l5_diagnostics")
            all_layer_spikes = result.get("all_layer_spikes")
        else:
            if export_seq_feats:
                feats, labels, inds, spike_rates, spike_times, seq_feats, seq_mask = (
                    result
                )
            else:
                feats, labels, inds, spike_rates, spike_times = result
            spike_meta = None
            signed_l5_diagnostics = None
            seq_feats = None
            seq_mask = None
            all_layer_spikes = None
        numpy.save("%s/spike_rate_epoch%s.npy" % (save_dir, args.epoch), spike_rates)
        numpy.save("%s/spike_time_epoch%s.npy" % (save_dir, args.epoch), spike_times)
    else:
        result = export_dataset_features(
            model,
            data_loader,
            dset,
            save_dir,
            torch,
            Variable,
            export_spikes=False,
            export_seq_feats=export_seq_feats,
            export_all_layers=export_all_layers,
            export_time_view=export_time_view,
            midlevel_forgery_margin_kwargs=midlevel_kwargs,
            dual_branch_readout=dual_branch,
            collect_signed_diagnostics=collect_signed,
        )
        if isinstance(result, dict):
            feats = result["feats"]
            labels = result["labels"]
            inds = result["inds"]
            seq_feats = result.get("seq_feats")
            seq_mask = result.get("seq_mask")
            spike_meta = result.get("spike_meta")
            signed_l5_diagnostics = result.get("signed_l5_diagnostics")
            all_layer_spikes = result.get("all_layer_spikes")
        else:
            if export_seq_feats:
                feats, labels, inds, seq_feats, seq_mask = result
            else:
                feats, labels, inds = result
            spike_meta = None
            signed_l5_diagnostics = None
            seq_feats = None
            seq_mask = None
            all_layer_spikes = None
        spike_rates = None
        spike_times = None

    # 下面三个文件是普通 SF/RF 验证器的必需输入。
    numpy.save("%s/feats_epoch%s.npy" % (save_dir, args.epoch), feats)
    numpy.save("%s/labels_epoch%s.npy" % (save_dir, args.epoch), labels)
    numpy.save("%s/inds_epoch%s.npy" % (save_dir, args.epoch), inds)
    if _uses_real_data(args):
        if __package__:
            from .real_forgery_data import FEATURE_VERSION, sha256_file
        else:
            from real_forgery_data import FEATURE_VERSION, sha256_file
        with open(os.path.join(save_dir, "real_data_provenance.json"), "w", encoding="utf-8") as handle:
            json.dump(dict(data_source="real", dataset=dataset_name,
                           split=getattr(args, "eval_split", "evaluation"),
                           feature_version=FEATURE_VERSION,
                           manifest_sha256=sha256_file(os.path.join(args.data_root, "manifest.json"))),
                      handle, indent=2)
    if export_seq_feats and seq_feats is not None:
        numpy.save("%s/seq_feat_epoch%s.npy" % (save_dir, args.epoch), seq_feats)
        numpy.save("%s/seq_mask_epoch%s.npy" % (save_dir, args.epoch), seq_mask)
    if spike_meta is not None:
        # TVS 验证器按 .npz 读取 rate_dims/time_dims，以确定每层在拼接特征中的切片。
        numpy.savez(
            "%s/spike_meta_epoch%s.npz" % (save_dir, args.epoch),
            **spike_meta,
        )
    if signed_l5_diagnostics is not None:
        numpy.save(
            "%s/signed_count_diag_epoch%s.npy" % (save_dir, args.epoch),
            signed_l5_diagnostics,
        )
    if all_layer_spikes is not None:
        for layer_idx, layer_spike in all_layer_spikes.items():
            numpy.save(
                "%s/all_layer_spike_l%d_epoch%s.npy"
                % (save_dir, layer_idx, args.epoch),
                layer_spike,
            )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    import torch
    from torch.autograd import Variable
    from torch.utils.data import DataLoader

    if args.eval_split == "validation" and (
        not _uses_real_data(args) or not args.export_tag or args.export_tag == args.run_tag
    ):
        parser.error("validation requires real data and a separate --export-tag")
    if _uses_real_data(args):
        if __package__:
            from .dataset.datasetRealForgery import evaluation_module
        else:
            from dataset.datasetRealForgery import evaluation_module
        dataset = evaluation_module()
    else:
        dataset = _import_dataset_module()

    # 固定导出过程中的 NumPy/PyTorch随机性；模型随后会切到 eval 模式。
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    dataset_specs = {
        "mcyt": ("mcyt", "MCYT"),
        "bsid": ("bsid", "BSID"),
        "bsds2": ("bsds2", "BSDS2"),
        "ebio1": ("ebio1", "EBio1"),
        "ebio2": ("ebio2", "EBio2"),
        "msds": ("msds", "MSDS"),
        "gpds": ("gpds", "GPDS"),
    }
    requested = [
        value.strip().lower()
        for value in args.eval_datasets.split(",")
        if value.strip()
    ]
    if "deepsigndb" in requested:
        if requested != ["deepsigndb"]:
            parser.error("use deepsigndb alone; do not mix other experimental datasets")
        requested = ["mcyt", "bsid", "bsds2", "ebio1", "ebio2"]
    unknown = sorted(set(requested).difference(dataset_specs))
    if not requested or unknown:
        raise ValueError(
            "invalid --eval-datasets; unknown=%s supported=%s"
            % (unknown, sorted(dataset_specs))
        )
    # 模型输入维度对所有项目数据集一致，因此用首个评估集创建临时 dset 来重建模型。
    first_key = dataset_specs[requested[0]][1]
    sig_dict = _load_evaluation_signatures(args, first_key)
    if _uses_real_data(args):
        from types import SimpleNamespace
        dset = SimpleNamespace(featDim=12)
    else:
        dset = dataset.dataset(sigDict=sig_dict)
    checkpoint_path = get_checkpoint_path(args)
    weight_dict = load_checkpoint_state(torch, checkpoint_path)
    if args.classifier_checkpoint:
        # 联邦 global_shared.pt 可能不含个性化分类头，此时从指定客户端权重补入分类头。
        classifier_state = load_checkpoint_state(torch, args.classifier_checkpoint)
        personalized = {
            key: value
            for key, value in classifier_state.items()
            if key == "cls.weight"
            or key.startswith("cls.")
            or key.startswith("classifier.")
        }
        if not personalized:
            raise ValueError(
                "classifier checkpoint contains no personalized classifier: %s"
                % args.classifier_checkpoint
            )
        weight_dict.update(personalized)
    model = build_evaluation_model(args, dset, weight_dict)
    try:
        # v2 联邦全局权重只保存去别名后的共享张量；补入个性化分类头后，
        # 在 strict load 前恢复模型完整的 alias schema。旧版完整 state_dict
        # 也会经过同一规范化路径，避免不同 checkpoint 格式产生加载差异。
        if args.model_name not in integer_tcn_config.INTEGER_MODEL_NAMES:
            import fedsnn

            weight_dict = fedsnn.expand_state_dict_aliases(model, weight_dict)
        # Integer-TCN has no registered tensor aliases and carries non-tensor
        # architecture metadata. Do not pass it through the legacy FL expander.
        # 默认 strict=True：结构参数、参数名称和张量形状必须全部匹配，禁止静默漏载权重。
        model.load_state_dict(weight_dict)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise RuntimeError(
            "checkpoint architecture does not match the requested evaluation model: %s\n%s"
            % (checkpoint_path, exc)
        ) from exc
    device = get_runtime_device(torch)
    model.to(device)
    # 关闭训练态随机行为和归一化层更新，保证导出可复现。
    model.train(mode=False)
    model.eval()

    # 同一个已加载模型依次导出所有请求的数据集，避免重复加载权重。
    for name in requested:
        dataset_name, dataset_key = dataset_specs[name]
        export_split(
            model,
            dataset,
            torch,
            Variable,
            DataLoader,
            args,
            dataset_name,
            dataset_key,
        )


if __name__ == "__main__":
    main()
