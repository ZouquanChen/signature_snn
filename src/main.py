# -*- coding:utf-8 -*-
import argparse
import json
import os
import pickle

import numpy

try:
    from . import integer_tcn_config
except ImportError:
    import integer_tcn_config


TRAIN_DATASET_SPECS = {
    "mcyt": {
        "data_file": "MCYT_dev.pkl",
        "sln_dir": "mcyt_dev_full",
        "cache_prefix": "MCYT",
    },
    "bsid": {
        "data_file": "BSID_dev.pkl",
        "sln_dir": "bsid_dev_full",
        "cache_prefix": "BSID",
    },
    "ebio1": {
        "data_file": "EBio1_dev.pkl",
        "sln_dir": "ebio1_dev_full",
        "cache_prefix": "EBio1",
    },
    "msds": {
        "data_file": "MSDS_dev_train.pkl",
        "sln_dir": "msds_dev_train_full",
        "cache_prefix": "MSDS",
    },
    "gpds": {
        "data_file": "GPDS_dev_train.pkl",
        "sln_dir": "gpds_dev_train_full",
        "cache_prefix": "GPDS",
    },
}


def parse_train_datasets(value):
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--train-datasets must contain at least one dataset name")
    unknown = sorted(set(names) - set(TRAIN_DATASET_SPECS))
    if unknown:
        raise ValueError(
            "unknown training datasets %s; available datasets are %s"
            % (unknown, sorted(TRAIN_DATASET_SPECS))
        )
    if len(names) != len(set(names)):
        raise ValueError("--train-datasets contains duplicate dataset names")
    return names


def build_parser():
    parser = argparse.ArgumentParser(description="Online signature verification")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="directory containing dataset pickle files",
    )
    parser.add_argument(
        "--sln-root",
        type=str,
        default="sigma_lognormal/params",
        help="directory containing sigma-lognormal parameter folders",
    )
    parser.add_argument(
        "--training-source", choices=["real", "synthetic"], default=None,
        help="default: real for integer-tcn; legacy synthetic pipeline for cnn/snn",
    )
    parser.add_argument(
        "--train-datasets",
        type=str,
        default="mcyt,bsid,ebio1",
        help="legacy datasets: mcyt, bsid, ebio1, msds, gpds; real centralized pipeline also accepts deepsigndb",
    )
    parser.add_argument(
        "--train-shot-g",
        type=int,
        default=5,
        help="number of genuine samples per class per training batch(default: 5)",
    )
    parser.add_argument(
        "--train-shot-f",
        type=int,
        default=10,
        help="number of forgery samples per class per training batch(default: 5)",
    )
    parser.add_argument(
        "--train-tasks", type=int, default=4, help="number of tasks per batch"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--epochs-per-round",
        type=int,
        default=None,
        help="number of epochs per federated round (overrides --epochs when set)",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="number of workers for DataLoader"
    )
    parser.add_argument(
        "--seed", type=int, default=111, help="numpy random seed (default: 111)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=25,
        help="how many epochs to wait before saving the model.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="clip the global gradient norm when positive (default: disabled)",
    )
    parser.add_argument(
        "--tau-raw-clamp",
        type=float,
        default=0.0,
        help="clamp channel-wise tau_raw to [-value, value] after each update",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn",
        choices=["cnn", "snn", *integer_tcn_config.INTEGER_MODEL_NAMES],
        help="model backend to train",
    )
    integer_tcn_config.add_integer_tcn_arguments(parser)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=None,
                        help="default: SGD for legacy models, AdamW for integer-tcn")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="integer-tcn linear warmup before cosine decay")
    parser.add_argument(
        "--lambda-proto",
        type=float,
        default=1.0,
        help="weight for prototype contrastive loss in SNN training",
    )
    parser.add_argument(
        "--lambda-cls",
        type=float,
        default=1.0,
        help="weight for classification loss in SNN training",
    )
    parser.add_argument(
        "--lambda-div",
        type=float,
        default=1e-5,
        help="weight for tau diversity regularization in SNN training",
    )
    parser.add_argument(
        "--midlevel-forgery-margin",
        action="store_true",
        help="enable mid-level forgery margin supervision for the SNN model",
    )
    parser.add_argument(
        "--lambda-midlevel-margin",
        type=float,
        default=0.0,
        help="weight for mid-level forgery margin supervision in SNN training",
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
        help="optional neuron-only baseline; leaves the formal FedBurst path unchanged",
    )
    parser.add_argument("--comparison-psn-order", type=int, default=2)
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
        help="scale factor for SNN backbone channel widths (default: 1.0, try 2.0)",
    )
    parser.add_argument(
        "--burst-spike",
        action="store_true",
        help="enable burst spike neuron (K output levels per timestep)",
    )
    parser.add_argument(
        "--burst-k",
        type=int,
        default=2,
        help="number of burst levels (default: 2)",
    )
    parser.add_argument(
        "--burst-slots",
        action="store_true",
        help="use binary burst slots (pure binary, time expansion by K)",
    )
    parser.add_argument(
        "--burst-raw-count",
        action="store_true",
        help="output raw spike count {0,1,2} instead of normalized {0,0.5,1}",
    )
    parser.add_argument(
        "--burst-signed-l5",
        action="store_true",
        help="enable signed burst counting only for the L5 spike layer",
    )
    parser.add_argument(
        "--burst-signed-all",
        action="store_true",
        help="enable signed burst counting for all spike layers",
    )
    parser.add_argument(
        "--l4-tau",
        type=float,
        default=None,
        help="override initial tau for L4 burst neuron (default 2.0)",
    )
    parser.add_argument(
        "--surrogate-sigma",
        type=float,
        default=1.0,
        help="spike surrogate gradient width (default 1.0)",
    )
    parser.add_argument(
        "--spike-gradient",
        choices=["surrogate", "ste"],
        default="surrogate",
        help=(
            "backward estimator for every hard spike: historical triangular "
            "surrogate or identity straight-through estimator (STE)"
        ),
    )
    parser.add_argument(
        "--learnable-l5-thresholds",
        action="store_true",
        help="make L5 burst thresholds learnable, initialized from burst-thresholds-per-layer",
    )
    parser.add_argument(
        "--learnable-l45-thresholds",
        action="store_true",
        help="make L4 and L5 burst thresholds learnable",
    )
    parser.add_argument(
        "--learnable-all-thresholds",
        action="store_true",
        help="make all burst thresholds learnable",
    )
    parser.add_argument(
        "--burst-k-per-layer",
        type=str,
        default=None,
        help="comma-separated per-layer burst K (e.g. 2,2,2,4,8,8)",
    )
    parser.add_argument(
        "--burst-thresholds",
        type=str,
        default=None,
        help="non-uniform thresholds for all layers (e.g. 1,2,3,4,6,8,10,12)",
    )
    parser.add_argument(
        "--burst-thresholds-per-layer",
        type=str,
        default=None,
        help="semicolon-sep per-layer thresholds (e.g. 1,2;1,2;1,2;1,2,3,4;1,2,3,4,5,6,7,8;1,2,3,4,6,8,10,12)",
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
        help="enable the dual-branch readout path for the SNN embedding head",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=1024,
        help="embedding dimension for the SNN readout head",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="optional run tag for non-overwriting checkpoint output paths",
    )
    parser.add_argument(
        "--fed-partition",
        type=str,
        default=None,
        help="path to JSON partition config for federated simulation",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=-1,
        help="client index for federated training (requires --fed-partition)",
    )
    parser.add_argument(
        "--fed-consensus-path",
        type=str,
        default=None,
        help="path to JSON consensus file for federated MP alignment",
    )
    parser.add_argument(
        "--fed-lambda-mp",
        type=float,
        default=0.05,
        help="weight for membrane potential alignment loss",
    )
    parser.add_argument(
        "--fed-lambda-tau",
        type=float,
        default=0.01,
        help="weight for tau alignment loss",
    )
    parser.add_argument(
        "--fed-lambda-thresh",
        type=float,
        default=0.005,
        help="weight for threshold alignment loss",
    )
    return parser


def _import_build_model():
    if __package__:
        from .model_factory import build_model
    else:
        from model_factory import build_model
    return build_model


def _import_build_user_prototypes():
    if __package__:
        from .network_snn import build_user_prototypes
    else:
        from network_snn import build_user_prototypes
    return build_user_prototypes


def _import_dataset_module():
    if __package__:
        from .dataset import datasetTrainAll_SLN as dataset
    else:
        import dataset.datasetTrainAll_SLN as dataset
    return dataset


def _data_path(args, filename):
    return os.path.join(args.data_root, filename)


def _sln_path(args, folder_name):
    return os.path.join(args.sln_root, folder_name)


def prototype_contrastive_loss(embedding, labels, margin=0.25):
    import torch
    import torch.nn.functional as F

    build_user_prototypes = _import_build_user_prototypes()
    prototypes, prototype_labels = build_user_prototypes(embedding, labels)
    if prototypes.size(0) <= 1:
        return embedding.new_tensor(0.0)

    embedding = F.normalize(embedding, dim=1, eps=1e-4)
    prototypes = F.normalize(prototypes, dim=1, eps=1e-4)
    similarities = torch.mm(embedding, prototypes.transpose(0, 1))
    positive_mask = labels.unsqueeze(1).eq(prototype_labels.unsqueeze(0))
    positive_scores = similarities[positive_mask]
    negative_scores = similarities.masked_fill(positive_mask, -1e9).max(dim=1)[0]
    return F.relu(margin - positive_scores + negative_scores).mean()


def label_smoothing_cross_entropy(logit, target, eps=0.1):
    import torch
    import torch.nn.functional as F

    n_class = logit.size(1)
    one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(logit, dim=1)
    return -(one_hot * log_prb).sum(dim=1).mean(dim=0)


def forge_contrastive_loss(embedding, n_task, n_shot_g, n_shot_f, margin=0.2):
    import torch
    import torch.nn.functional as F

    step = 1 + n_shot_g + n_shot_f
    normalized = F.normalize(embedding, dim=1, eps=1e-4)
    loss = embedding.new_tensor(0.0)
    map_proxy = embedding.new_tensor(0.0)
    for i in range(n_task):
        anchor = normalized[i * step]
        pos = normalized[i * step + 1 : i * step + 1 + n_shot_g]
        neg = normalized[i * step + 1 + n_shot_g : (i + 1) * step]
        score_ap = torch.sum(anchor.unsqueeze(0) * pos, dim=1)
        score_an = torch.sum(anchor.unsqueeze(0) * neg, dim=1)
        loss = (
            loss + F.relu(margin - score_ap.unsqueeze(1) + score_an.unsqueeze(0)).mean()
        )
        map_proxy = (
            map_proxy
            + (score_ap.unsqueeze(1) > score_an.unsqueeze(0)).to(anchor.dtype).mean()
        )
    return loss / n_task, map_proxy / n_task


def adjust_learning_rate(optimizer, epoch, base_lr):
    lr = base_lr if epoch < 150 else base_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_checkpoint_dir(args):
    if args.run_tag:
        return "./models/%s/%d" % (args.run_tag, args.seed)
    return "./models/%d" % args.seed


def prepare_checkpoint_dir(args):
    save_dir = get_checkpoint_dir(args)
    if args.run_tag and os.path.exists(save_dir):
        raise FileExistsError("Checkpoint directory already exists: %s" % save_dir)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def resolve_training_source(args):
    source = getattr(args, "training_source", None)
    if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES:
        if source == "synthetic":
            raise ValueError("integer-tcn now uses real forgeries; legacy cnn/snn synthesis remains available")
        return "real"
    return source or "synthetic"


def load_training_data(args):
    if resolve_training_source(args) == "real":
        if __package__:
            from .real_forgery_data import load_training_data as load_real_data
        else:
            from real_forgery_data import load_training_data as load_real_data
        return load_real_data(args)
    if os.path.isfile(os.path.join(args.data_root, "manifest.json")):
        raise ValueError("do not run legacy synthesis inside the isolated real-forgery data folder")
    from torch.utils.data import DataLoader
    import json

    dataset = _import_dataset_module()
    n_task = args.train_tasks
    n_shot_g = args.train_shot_g
    n_shot_f = args.train_shot_f

    client_id = getattr(args, "client_id", -1)
    fed_partition_path = getattr(args, "fed_partition", None)

    train_datasets = parse_train_datasets(args.train_datasets)

    if fed_partition_path is not None and client_id >= 0:
        with open(fed_partition_path, "r") as fh:
            partition = json.load(fh)
        client_key = "client_%d" % client_id
        if client_key not in partition:
            raise ValueError(
                "client %d not found in partition %s" % (client_id, fed_partition_path)
            )
        client_writers = {
            name: set(partition[client_key].get(name, []))
            for name in train_datasets
        }
    else:
        client_writers = {name: None for name in train_datasets}

    # 联邦客户端只持有各自的 writer 子集，不能共享按数据集名称生成的合成轨迹缓存。
    # 否则先初始化的客户端会写入一个不完整缓存，后续客户端加载该缓存时会因缺少
    # 自己的 writer/样本而在 synthesisPressure 中发生索引越界。集中式训练仍沿用
    # 原来的 MCYT/BSID/EBio1 缓存名称，联邦训练则为每个 client 建立独立缓存。
    if client_id >= 0:
        # 将划分文件名也纳入作用域，避免 IID、不同 Dirichlet alpha 等划分复用缓存。
        partition_name = os.path.splitext(os.path.basename(fed_partition_path))[0]
        cache_suffix = "_%s_client%d" % (partition_name, client_id)
    else:
        cache_suffix = ""

    dset = None
    for name in train_datasets:
        spec = TRAIN_DATASET_SPECS[name]
        with open(_data_path(args, spec["data_file"]), "rb") as fh:
            sigDict = pickle.load(fh)
        selected_writers = client_writers[name]
        if selected_writers is not None:
            sigDict = {
                key: value
                for key, value in sigDict.items()
                if key in selected_writers
            }
        if not sigDict:
            raise ValueError(
                "dataset %s contains no writers for client %d" % (name, client_id)
            )

        prefix = "%s%s" % (spec["cache_prefix"], cache_suffix)
        if dset is None:
            dset = dataset.dataset(
                sigDict=sigDict,
                slnPath=_sln_path(args, spec["sln_dir"]),
                slnLevel=0.4,
                taskSize=n_task,
                taskNumGen=n_shot_g,
                taskNumNeg=n_shot_f,
                numSynthesis=10,
                prefix=prefix,
            )
        else:
            dset.addDatabase(
                sigDict=sigDict,
                slnPath=_sln_path(args, spec["sln_dir"]),
                prefix=prefix,
            )
        del sigDict

    dset.computeStats()

    sampler = dataset.batchSampler(dset, loop=False)
    dataLoader = DataLoader(
        dset,
        num_workers=args.workers,
        batch_sampler=sampler,
        collate_fn=dataset.collate_fn,
    )
    return dset, dataLoader


def build_training_model(args, dset):
    build_model = _import_build_model()
    common_kwargs = dict(
        n_in=dset.featDim,
        n_classes=getattr(dset, "n_classes_override", len(dset)),
    )
    if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES:
        integer_tcn_config.validate_integer_tcn_options(args)
        common_kwargs.update(integer_tcn_config.model_kwargs(args))
        common_kwargs.update(n_task=args.train_tasks, n_shot_g=args.train_shot_g,
                             n_shot_f=args.train_shot_f)
        return build_model(args.model_name, **common_kwargs)
    if args.model_name == "cnn":
        common_kwargs.update(
            dict(
                n_task=args.train_tasks,
                n_shot_g=args.train_shot_g,
                n_shot_f=args.train_shot_f,
                APAlpha=6.0,
            )
        )
    else:
        common_kwargs.update(
            dict(
                embedding_dim=args.embed_dim,
                n_task=args.train_tasks,
                n_shot_g=args.train_shot_g,
                n_shot_f=args.train_shot_f,
                lambda_div=getattr(args, "lambda_div", 1e-5),
                lambda_midlevel_margin=getattr(args, "lambda_midlevel_margin", 0.0),
                sim_steps=getattr(args, "sim_steps", 4),
                rollout_repeated_input=getattr(args, "rollout_repeated_input", False),
                dual_branch_readout=getattr(args, "dual_branch_readout", False),
                width_scale=float(getattr(args, "sn_width_scale", 1.0)),
                burst_spike=getattr(args, "burst_spike", False),
                burst_k=int(getattr(args, "burst_k", 2)),
                burst_slots=getattr(args, "burst_slots", False),
                burst_raw_count=getattr(args, "burst_raw_count", False),
                burst_signed_l5=getattr(args, "burst_signed_l5", False),
                burst_signed_all=getattr(args, "burst_signed_all", False),
                l4_tau=getattr(args, "l4_tau", None),
                burst_k_per_layer=getattr(args, "burst_k_per_layer", None),
                burst_thresholds=getattr(args, "burst_thresholds", None),
                burst_thresholds_per_layer=getattr(
                    args, "burst_thresholds_per_layer", None
                ),
                learnable_l5_thresholds=getattr(args, "learnable_l5_thresholds", False),
                learnable_l45_thresholds=getattr(
                    args, "learnable_l45_thresholds", False
                ),
                learnable_all_thresholds=getattr(
                    args, "learnable_all_thresholds", False
                ),
                comparison_neuron=getattr(args, "comparison_neuron", None),
                comparison_psn_order=getattr(args, "comparison_psn_order", 2),
            )
        )
        if getattr(args, "midlevel_forgery_margin", False):
            common_kwargs["midlevel_forgery_margin"] = True
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


def get_runtime_device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch_cnn(model, dataLoader, optimizer, epoch):
    import torch
    from torch.autograd import Variable

    device = next(model.parameters()).device
    train_loss = 0
    train_lossCE = 0
    train_mAP = 0
    for idx, batch in enumerate(dataLoader):
        sig, lens, label = batch
        mask = model.getOutputMask(lens)

        sig = Variable(torch.from_numpy(sig)).to(device)
        mask = Variable(torch.from_numpy(mask)).to(device)
        lens = Variable(torch.from_numpy(lens)).to(device)
        label = Variable(torch.from_numpy(label)).to(device)

        model.zero_grad()
        output, output2 = model(sig, mask, lens)
        lossCE = model.smoothCEloss(output2, label, eps=0.10)
        loss, mAP = model.APLoss_DLM(output)

        (loss + lossCE).backward()
        optimizer.step()

        train_mAP += mAP.item()
        train_loss += loss.item()
        train_lossCE += lossCE.item()

        if idx % 50 == 0:
            if idx == 0:
                print(epoch, idx, train_loss, train_lossCE, train_mAP)
            else:
                print(epoch, idx, train_loss / 50, train_lossCE / 50, train_mAP / 50)
            train_loss = 0
            train_lossCE = 0
            train_mAP = 0


def train_epoch_snn(model, dataLoader, optimizer, epoch, args):
    import torch

    device = next(model.parameters()).device
    lambda_div = float(getattr(args, "lambda_div", 1e-5))
    lambda_midlevel_margin = float(getattr(args, "lambda_midlevel_margin", 0.0))
    use_midlevel_margin = bool(
        getattr(args, "midlevel_forgery_margin", False) and lambda_midlevel_margin > 0.0
    )
    train_loss = 0
    train_lossCE = 0
    train_mAP = 0
    for idx, batch in enumerate(dataLoader):
        sig, lens, label = batch
        mask = model.getOutputMask(lens)

        sig = torch.from_numpy(sig).to(device)
        mask = torch.from_numpy(mask).to(device)
        lens = torch.from_numpy(lens).to(device)
        label = torch.from_numpy(label).long().to(device)

        model.zero_grad()
        outputs = model(sig, mask, lens)
        model._last_embedding = outputs[0] if isinstance(outputs, tuple) else outputs
        midlevel_feature = None
        if isinstance(outputs, tuple) and len(outputs) == 3:
            output, output2, midlevel_feature = outputs
        else:
            output, output2 = outputs
        lossCE = model.smoothCEloss(output2, label, eps=0.10)
        loss, mAP = model.APLoss_DLM(output)
        tau_div = model.tau_diversity_regularization()
        total_loss = loss + lossCE - lambda_div * tau_div
        fedlec_teacher = getattr(model, "_fedlec_teacher", None)
        missing_labels = getattr(model, "_fedlec_missing_labels", None)
        if fedlec_teacher is not None and missing_labels is not None:
            fedlec_lambda = float(getattr(args, "fedlec_lambda", 0.1))
            if fedlec_lambda > 0.0 and len(missing_labels) > 1:
                with torch.no_grad():
                    teacher_outputs = fedlec_teacher(sig, mask, lens)
                    teacher_logits = teacher_outputs[1]
                missing = torch.as_tensor(missing_labels, dtype=torch.long, device=device)
                student_missing = torch.log_softmax(output2.index_select(1, missing), dim=1)
                teacher_missing = torch.softmax(
                    teacher_logits.index_select(1, missing), dim=1
                )
                fedlec_loss = torch.nn.functional.kl_div(
                    student_missing, teacher_missing, reduction="batchmean"
                )
                total_loss = total_loss + fedlec_lambda * fedlec_loss
        if hasattr(model, "_fed_prox_reference") and model._fed_prox_reference is not None:
            prox_mu = float(getattr(args, "fed_prox_mu", 0.0))
            if prox_mu > 0.0:
                prox_loss = torch.tensor(0.0, device=device)
                for name, parameter in model.named_parameters():
                    if name in model._fed_prox_reference:
                        reference = model._fed_prox_reference[name].to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                        prox_loss = prox_loss + ((parameter - reference) ** 2).sum()
                total_loss = total_loss + 0.5 * prox_mu * prox_loss
        if use_midlevel_margin:
            if midlevel_feature is None:
                raise ValueError(
                    "midlevel_forgery_margin requires the model to return a midlevel feature"
                )
            aux_loss = model.midlevel_forgery_margin_loss(midlevel_feature)
            total_loss = total_loss + lambda_midlevel_margin * aux_loss

        if hasattr(model, "_fed_consensus") and model._fed_consensus is not None:
            consensus = model._fed_consensus
            lambda_mp = float(getattr(args, "fed_lambda_mp", 0.05))
            lambda_tau = float(getattr(args, "fed_lambda_tau", 0.01))
            lambda_thresh = float(getattr(args, "fed_lambda_thresh", 0.005))
            if (
                "mp_targets" in consensus
                and lambda_mp > 0
                and len(consensus.get("mp_targets", {})) > 0
            ):
                total_loss = total_loss + lambda_mp * model.mp_alignment_loss(
                    consensus["mp_targets"]
                )
            if (
                "tau" in consensus
                and lambda_tau > 0
                and len(consensus.get("tau", [])) > 0
            ):
                tau_consensus = torch.tensor(
                    consensus["tau"], device=device, dtype=torch.float32
                )
                total_loss = total_loss + lambda_tau * model.tau_alignment_loss(
                    tau_consensus
                )
            if (
                "l5_log_spacings" in consensus
                and lambda_thresh > 0
                and len(consensus.get("l5_log_spacings", [])) > 0
            ):
                thresh_consensus = torch.tensor(
                    consensus["l5_log_spacings"], device=device, dtype=torch.float32
                )
                total_loss = (
                    total_loss
                    + lambda_thresh * model.threshold_alignment_loss(thresh_consensus)
                )

        if (
            hasattr(model, "_fed_distill_targets")
            and model._fed_distill_targets is not None
        ):
            lambda_distill = float(getattr(args, "fed_lambda_distill", 0.01))
            if lambda_distill > 0:
                total_loss = (
                    total_loss
                    + lambda_distill
                    * model.spike_rate_distillation_loss(model._fed_distill_targets)
                )

        if (
            hasattr(model, "_fed_logit_targets")
            and model._fed_logit_targets is not None
            and output2 is not None
        ):
            lambda_logit = float(getattr(args, "fed_lambda_logit", 0.005))
            if lambda_logit > 0:
                total_loss = total_loss + lambda_logit * model.logit_distillation_loss(
                    model._fed_logit_targets, output2
                )

        if hasattr(model, "_fed_emb_targets") and model._fed_emb_targets is not None:
            lambda_emb = float(getattr(args, "fed_lambda_emb", 0.01))
            if lambda_emb > 0:
                total_loss = (
                    total_loss
                    + lambda_emb
                    * model.embedding_distillation_loss(model._fed_emb_targets)
                )

        if hasattr(model, "_fed_proto") and model._fed_proto is not None:
            lambda_proto = float(getattr(args, "fed_lambda_proto", 0.1))
            if lambda_proto > 0:
                total_loss = total_loss + lambda_proto * model.prototype_alignment_loss(
                    model._fed_proto
                )

        if getattr(model, "_fed_flea_decorrelation", False):
            flea_weight = float(getattr(args, "fed_flea_corr_weight", 3.0))
            embedding = getattr(model, "_last_embedding", None)
            if flea_weight > 0.0 and embedding is not None and embedding.shape[0] > 1:
                centered = embedding - embedding.mean(dim=0, keepdim=True)
                covariance = centered.transpose(0, 1).matmul(centered)
                covariance = covariance / float(embedding.shape[0] - 1)
                diagonal = torch.diag(torch.diag(covariance))
                total_loss = total_loss + flea_weight * (
                    (covariance - diagonal) ** 2
                ).mean()

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "non-finite SNN loss at epoch=%d batch=%d: ap=%s ce=%s tau_div=%s"
                % (
                    epoch,
                    idx,
                    loss.detach().item(),
                    lossCE.detach().item(),
                    tau_div.detach().item(),
                )
            )

        total_loss.backward()
        if hasattr(model, "_scaffold_global") and model._scaffold_global is not None:
            local_control = model._scaffold_local
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and name in model._scaffold_global:
                    global_value = model._scaffold_global[name].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                    local_value = local_control[name].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                    parameter.grad.add_(global_value - local_value)
        clip_norm = float(getattr(args, "grad_clip_norm", 0.0))
        if clip_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_norm, error_if_nonfinite=True
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    "non-finite gradient norm at epoch=%d batch=%d" % (epoch, idx)
                )
        optimizer.step()

        # 联邦长程训练可选择限制 tau_raw 的数值范围。该操作不改变通道级可学习
        # 时间常数的形式，只防止负方差多样性目标将原始参数推到浮点溢出区域。
        tau_raw_clamp = float(getattr(args, "tau_raw_clamp", 0.0))
        if tau_raw_clamp > 0.0:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if not name.endswith("tau_raw"):
                        continue
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            "non-finite tau_raw after optimizer step at "
                            "epoch=%d batch=%d parameter=%s" % (epoch, idx, name)
                        )
                    parameter.clamp_(-tau_raw_clamp, tau_raw_clamp)

        train_loss += loss.item()
        train_lossCE += lossCE.item()
        train_mAP += mAP.item()

        if idx % 50 == 0:
            if idx == 0:
                print(epoch, idx, train_loss, train_lossCE, train_mAP)
            else:
                print(epoch, idx, train_loss / 50, train_lossCE / 50, train_mAP / 50)
            train_loss = 0
            train_lossCE = 0
            train_mAP = 0


def run_training_epoch(model, dataLoader, optimizer, epoch, args):
    if args.model_name == "cnn":
        train_epoch_cnn(model, dataLoader, optimizer, epoch)
    else:
        train_epoch_snn(model, dataLoader, optimizer, epoch, args)


def build_training_optimizer(model, args):
    import torch

    choice = getattr(args, "optimizer", None)
    choice = choice or ("adamw" if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES else "sgd")
    if choice == "sgd":
        gaps = [parameter for name, parameter in model.named_parameters() if name.endswith("raw_gaps")]
        gap_ids = {id(parameter) for parameter in gaps}
        parameters = ([{"params": [p for p in model.parameters() if id(p) not in gap_ids]},
                       {"params": gaps, "weight_decay": 0.0}] if gaps else model.parameters())
        return torch.optim.SGD(parameters, lr=args.lr, momentum=0.9,
                               weight_decay=1e-5, nesterov=True)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = no_decay if parameter.ndim < 2 or "residual_scale" in name or name.endswith("raw_gaps") else decay
        target.append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_training_source(args)
    if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES:
        integer_tcn_config.validate_integer_tcn_options(args)
        if not args.run_tag:
            parser.error("integer-tcn requires --run-tag to isolate new checkpoints")
        if args.warmup_epochs < 0:
            parser.error("--warmup-epochs must be nonnegative")

    try:
        import torch
        import torch.backends.cudnn as cudnn
    except ImportError as exc:
        raise ImportError("torch is required to train Sig2Vec models") from exc

    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

    dset, dataLoader = load_training_data(args)
    model = build_training_model(args, dset)
    device = get_runtime_device(torch)
    model.to(device)

    if args.model_name == "snn" and args.fed_consensus_path is not None:
        with open(args.fed_consensus_path, "r") as fh:
            model._fed_consensus = json.load(fh)

    optimizer = build_training_optimizer(model, args)

    if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES:
        print("TCN spike gradient: clipped-ste; S=1; D=%d; temporal context=%s"
              % (model.integer_max, model.receptive_field or "full causal prefix (continuous Mamba-2)"))
        if hasattr(model, "interval_config"):
            print("Mamba signed interval gradient: normalized-triangle-v1; options=%s" % model.interval_config)
    elif args.model_name != "cnn":
        from network_snn import configure_spike_gradient

        configure_spike_gradient(args.spike_gradient, args.surrogate_sigma)
        print(
            "Spike gradient: mode=%s sigma=%g"
            % (args.spike_gradient, float(args.surrogate_sigma))
        )

    save_dir = prepare_checkpoint_dir(args)
    # A state_dict does not encode the backward-only estimator. Persist the
    # setting beside centralized checkpoints so STE and legacy surrogate runs
    # remain auditable even though their forward architectures are identical.
    with open(os.path.join(save_dir, "training_config.json"), "w") as handle:
        json.dump(
            {
                "model_name": args.model_name,
                "seed": args.seed,
                "spike_gradient": (
                    "clipped-ste" if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES else args.spike_gradient
                ),
                "surrogate_sigma": (
                    None if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES else float(args.surrogate_sigma)
                ),
                "model_config": model.get_extra_state() if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES else None,
                "trainer_args": vars(args),
                "training_source": resolve_training_source(args),
                "data_provenance": getattr(dset, "data_provenance", None),
            },
            handle,
            indent=2,
        )

    num_epochs = (
        args.epochs_per_round if args.epochs_per_round is not None else args.epochs
    )

    for epoch in range(num_epochs):
        if args.model_name in integer_tcn_config.INTEGER_MODEL_NAMES:
            import math
            warmup = min(args.warmup_epochs, max(num_epochs - 1, 0))
            if epoch < warmup:
                factor = (epoch + 1) / float(warmup)
            else:
                progress = (epoch - warmup) / max(num_epochs - warmup - 1, 1)
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
        run_training_epoch(model, dataLoader, optimizer, epoch, args)

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), "%s/epoch%d" % (save_dir, epoch))

    torch.save(model.state_dict(), "%s/epochEnd" % save_dir)


if __name__ == "__main__":
    main()
