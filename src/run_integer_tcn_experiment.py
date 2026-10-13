"""Train all configured seeds, save minimum TRAIN-loss weights, then evaluate.

No validation split; no checkpoint/hyperparameter selection on the evaluation
set. Full epochs, fixed recipe, atomic best/last checkpoints and resumable RNG.
Use from the standalone project root. Only owned run directories are written.
"""

import argparse
import copy
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

try:
    from . import main as training
    from . import train_integer_tcn
except ImportError:
    import main as training
    import train_integer_tcn


PROJECT = Path(__file__).resolve().parents[1]
SELECTION = "minimum_sample_weighted_epoch_mean_training_AP_plus_CE"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def learning_rate(epoch, epochs, warmup, base_lr):
    warmup = min(warmup, max(epochs - 1, 0))
    if epoch < warmup:
        return base_lr * (epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def backward_batch(model, batch, device, microbatch_tasks=1):
    """Equivalent episode-wise AP+CE accumulation; no batch/time normalization.

    The caller zeros gradients and updates the optimizer once per original
    batch. Only padding is trimmed; no valid signature points are removed.
    """
    signals, lengths, labels = batch
    episode_size = 1 + model.n_shot_g + model.n_shot_f
    if len(signals) != model.n_task * episode_size:
        raise ValueError("batch does not match the recorded episode layout")
    if microbatch_tasks < 1:
        raise ValueError("microbatch_tasks must be positive")
    metrics = dict(total_loss=0.0, ap_loss=0.0, ce_loss=0.0, train_map=0.0)
    for start in range(0, len(signals), microbatch_tasks * episode_size):
        end = min(start + microbatch_tasks * episode_size, len(signals))
        tasks = (end - start) // episode_size
        # Preserve enough padded length to disambiguate the 12-feature axis.
        steps = max(int(np.max(lengths[start:end])), model.n_in + 1)
        x = torch.from_numpy(signals[start:end, :steps]).to(device)
        lens = torch.as_tensor(lengths[start:end], device=device)
        target = torch.as_tensor(labels[start:end], device=device, dtype=torch.long)
        embedding, logits = model(x, lens=lens)
        ap, map_value = model.APLoss_DLM(embedding, n_task=tasks)
        ce = model.smoothCEloss(logits, target, eps=0.10)
        loss = ap + ce
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite AP+CE training loss")
        weight = tasks / float(model.n_task)
        (loss * weight).backward()
        for key, value in (("total_loss", loss), ("ap_loss", ap),
                           ("ce_loss", ce), ("train_map", map_value)):
            metrics[key] += float(value.detach()) * weight
    return metrics


def train_epoch(model, loader, optimizer, args, epoch, logger, progress):
    model.train()
    device = next(model.parameters()).device
    totals = dict(total_loss=0., ap_loss=0., ce_loss=0., train_map=0.)
    samples, batches = 0, 0
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        metrics = backward_batch(model, batch, device, args.microbatch_tasks)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        count = len(batch[0])
        for key in totals:
            totals[key] += metrics[key] * count
        samples += count
        batches += 1
        if batch_index % 25 == 0 or batch_index + 1 == len(loader):
            elapsed = time.perf_counter() - start_time
            state = dict(epoch=epoch + 1, batch=batch_index + 1, batches=len(loader),
                         current_train_loss=metrics["total_loss"],
                         running_train_loss=totals["total_loss"] / samples,
                         epoch_elapsed_seconds=elapsed)
            progress(state)
            logger.info("epoch=%d batch=%d/%d loss=%.6f ap=%.6f ce=%.6f map=%.4f grad=%.4f elapsed=%.1fs",
                        epoch + 1, batch_index + 1, len(loader), metrics["total_loss"],
                        metrics["ap_loss"], metrics["ce_loss"], metrics["train_map"],
                        float(gradient_norm), elapsed)
    if not samples:
        raise ValueError("empty training loader")
    result = {key: value / samples for key, value in totals.items()}
    result.update(epoch=epoch + 1, batches=batches, samples=samples,
                  lr=optimizer.param_groups[0]["lr"],
                  seconds=time.perf_counter() - start_time,
                  peak_allocated_mb=(torch.cuda.max_memory_allocated(device) / 1024**2
                                     if device.type == "cuda" else None))
    return result


def cpu_weights(model):
    return {key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in model.state_dict().items()}


def load_owned_checkpoint(path):
    # Full RNG snapshots contain numpy/Python objects. Only use locally created,
    # trusted checkpoints under this experiment's owned directory.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_logger(path, seed):
    logger = logging.getLogger("integer-experiment-%d" % seed)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def train_seed(dataset, loader, train_args, job, logger, status, config_hash):
    seed_all(train_args.seed)
    directory = PROJECT / "models" / job.run_tag / str(train_args.seed)
    directory.mkdir(parents=True, exist_ok=job.resume)
    model = training.build_training_model(train_args, dataset).to(job.device)
    model.checkpoint_blocks = job.checkpoint_blocks
    optimizer = training.build_training_optimizer(model, train_args)
    last_path = directory / "last.pt"
    history, best, best_weights, best_intervals, start_epoch = [], None, None, None, 0
    if job.resume and last_path.exists():
        snapshot = load_owned_checkpoint(last_path)
        if snapshot["config_hash"] != config_hash:
            raise ValueError("resume configuration differs from saved run")
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        history, best, start_epoch = snapshot["history"], snapshot["best"], snapshot["epoch"]
        best_weights = snapshot.get("best_weights")
        best_intervals = snapshot.get("best_intervals")
        if best is not None:
            if best_weights is None:
                raise ValueError("resume checkpoint has no authoritative best weights")
            atomic_checkpoint(directory / "best_train_loss.pt", best_weights)
            atomic_json(directory / "best_train_loss.json", best)
            if hasattr(model, "interval_diagnostics"):
                if best_intervals is None:
                    raise ValueError("resume checkpoint is missing best interval diagnostics")
                atomic_json(directory / "best_train_intervals.json", best_intervals)
        restore_rng(snapshot["rng"])
        logger.info("Resuming completed epoch %d", start_epoch)
    elif job.resume and list(directory.iterdir()):
        raise ValueError("nonempty seed directory has no resumable last.pt")

    def save_last(epoch):
        atomic_checkpoint(last_path, dict(epoch=epoch, model=cpu_weights(model),
                          optimizer=optimizer.state_dict(), rng=rng_state(),
                          history=history, best=best, best_weights=best_weights, best_intervals=best_intervals,
                          config_hash=config_hash))

    if not start_epoch:
        atomic_json(directory / "training_config.json", dict(
            trainer_args=vars(train_args), model_config=model.get_extra_state(),
            training_source="real", data_provenance=dataset.data_provenance,
            selection=SELECTION, validation_split=False, test_used_for_selection=False,
            microbatch_tasks=job.microbatch_tasks, checkpoint_blocks=job.checkpoint_blocks,
            identity_ce="unchanged: all samples use the target writer label",
        ))
        if hasattr(model, "interval_diagnostics"):
            atomic_json(directory / "initial_intervals.json", model.interval_diagnostics())
        save_last(0)
    logger.info("Seed %d: %d writers; %d batches/epoch; all data retained; selection=%s",
                train_args.seed, len(dataset), len(loader), SELECTION)
    for epoch in range(start_epoch, train_args.epochs):
        rate = learning_rate(epoch, train_args.epochs, train_args.warmup_epochs, train_args.lr)
        for group in optimizer.param_groups:
            group["lr"] = rate
        epoch_args = SimpleEpochArgs(job.microbatch_tasks, train_args.grad_clip_norm)
        metrics = train_epoch(model, loader, optimizer, epoch_args, epoch, logger,
                              lambda current: status(dict(phase="training", seed=train_args.seed, **current)))
        history.append(metrics)
        if hasattr(model, "interval_diagnostics"):
            atomic_json(directory / "interval_diagnostics.json", dict(
                epoch=epoch + 1, neurons=model.interval_diagnostics()))
        if best is None or metrics["total_loss"] < best["total_loss"]:
            best = dict(metrics, selection=SELECTION, seed=train_args.seed)
            # The full checkpoint stores the authoritative best snapshot for
            # crash recovery; the plain state_dict is the evaluator's input.
            best_weights = cpu_weights(model)
            atomic_checkpoint(directory / "best_train_loss.pt", best_weights)
            atomic_json(directory / "best_train_loss.json", best)
            if hasattr(model, "interval_diagnostics"):
                best_intervals = dict(epoch=epoch + 1, neurons=model.interval_diagnostics())
                atomic_json(directory / "best_train_intervals.json", best_intervals)
            logger.info("NEW BEST TRAIN LOSS epoch=%d mean=%.6f", epoch + 1, best["total_loss"])
        atomic_json(directory / "history.json", history)
        save_last(epoch + 1)
        if (epoch + 1) % train_args.save_interval == 0:
            atomic_checkpoint(directory / ("epoch_%03d.pt" % (epoch + 1)), cpu_weights(model))
        logger.info("EPOCH SUMMARY %s", json.dumps(metrics))
        status(dict(phase="training", seed=train_args.seed, epoch=epoch + 1,
                    best_epoch=best["epoch"], best_train_loss=best["total_loss"], epoch_metrics=metrics))
    atomic_checkpoint(directory / "epochEnd", cpu_weights(model))
    optimizer.zero_grad(set_to_none=True)
    return best


class SimpleEpochArgs:
    def __init__(self, microbatch_tasks, grad_clip_norm):
        self.microbatch_tasks, self.grad_clip_norm = microbatch_tasks, grad_clip_norm


def child_command(command, logfile, logger):
    logger.info("RUN %s", subprocess.list2cmdline(command))
    with Path(logfile).open("a", encoding="utf-8") as output:
        output.write("\n[%s] %s\n" % (now(), subprocess.list2cmdline(command)))
        output.flush()
        subprocess.run(command, cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT,
                       check=True, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))


def evaluate_seed(train_args, job, log_dir, result_dir, logger, status):
    recorded = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
    for name, checksum in recorded["source_sha256"].items():
        if hashlib.sha256((PROJECT / "src" / name).read_bytes()).hexdigest() != checksum:
            raise RuntimeError("source changed during the experiment; refusing inconsistent evaluation: " + name)
    seed = train_args.seed
    export_tag = job.run_tag + "-best-train-loss"
    checkpoint = PROJECT / "models" / job.run_tag / str(seed) / "best_train_loss.pt"
    status(dict(phase="exporting_best_train_loss", seed=seed))
    child_command([
        sys.executable, "-u", "-B", str(PROJECT / "src/evaluate_stylus.py"),
        "--model-name", train_args.model_name, "--data-source", "real",
        "--data-root", train_args.data_root, "--eval-datasets", "deepsigndb",
        "--checkpoint", str(checkpoint), "--run-tag", job.run_tag,
        "--export-tag", export_tag, "--seed", str(seed), "--epoch", "BestTrainLoss",
    ], log_dir / ("seed%d-evaluation.log" % seed), logger)
    outcomes = {}
    for references in (1, 4):
        status(dict(phase="verifying_%dvs1" % references, seed=seed))
        output = result_dir / ("seed%d-%dvs1.json" % (seed, references))
        if not output.exists():
            child_command([
                sys.executable, "-u", "-B", str(PROJECT / "src/verifier_deepsign_central.py"),
                "--data-root", train_args.data_root, "--run-tag", export_tag,
                "--epoch", "BestTrainLoss", "--seeds", str(seed),
                "--references", str(references), "--output", str(output),
            ], log_dir / ("seed%d-evaluation.log" % seed), logger)
        outcomes[str(references)] = json.loads(output.read_text(encoding="utf-8"))
    return outcomes


def summarize(completed):
    result = {}
    for references in (1, 4):
        values = [entry["evaluation"][str(references)]["mean_eer_percent"] for entry in completed]
        if values:
            result["%dvs1" % references] = dict(
                seeds=[entry["seed"] for entry in completed],
                mean_eer_percent=float(np.mean(values)),
                sample_std_percent=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--config", type=Path, default=train_integer_tcn.DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444, 5555])
    parser.add_argument("--epochs", type=int, help="optional explicit budget override")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--microbatch-tasks", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-checkpoint-blocks", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    job = build_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job.run_tag):
        raise ValueError("run-tag must be a single safe directory name")
    if len(set(job.seeds)) != len(job.seeds) or not job.seeds:
        raise ValueError("seeds must be distinct")
    if min(job.cpu_threads, job.microbatch_tasks) < 1:
        raise ValueError("threads/microbatch_tasks must be positive")
    job.checkpoint_blocks = not job.no_checkpoint_blocks
    torch.set_num_threads(job.cpu_threads)
    if job.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU training")
    os.chdir(PROJECT)
    train_args = train_integer_tcn.build_parser(job.config).parse_args([])
    train_args.run_tag = job.run_tag
    if job.epochs is not None:
        train_args.epochs = job.epochs
    if train_args.epochs < 1 or train_args.save_interval < 1 or train_args.grad_clip_norm <= 0:
        raise ValueError("positive epochs/checkpoint interval/gradient clip are required")
    if train_args.train_datasets != "deepsigndb" or train_args.training_source != "real":
        raise ValueError("this experiment is centralized real DeepSignDB only")
    result_dir = PROJECT / "results" / job.run_tag
    log_dir = PROJECT / "logs" / job.run_tag
    if not job.resume and any(path.exists() for path in (result_dir, log_dir, PROJECT / "models" / job.run_tag)):
        raise FileExistsError("run already exists; use --resume only for this exact experiment")
    if job.resume and not (result_dir / "run_config.json").exists():
        raise FileNotFoundError("no recorded experiment to resume")
    result_dir.mkdir(parents=True, exist_ok=job.resume)
    log_dir.mkdir(parents=True, exist_ok=job.resume)
    state = dict(phase="initializing", started_at=now(), run_tag=job.run_tag,
                 seeds=job.seeds, epochs_per_seed=train_args.epochs, completed=[])
    if job.resume and (result_dir / "status.json").exists():
        state.update(json.loads((result_dir / "status.json").read_text(encoding="utf-8")))

    def status(values):
        state.update(values)
        state["pid"] = os.getpid()
        state["updated_at"] = now()
        atomic_json(result_dir / "status.json", state)

    logger = make_logger(log_dir / "experiment.log", 0)
    try:
        config = dict(trainer_args=copy.deepcopy(vars(train_args)), seeds=job.seeds, device=job.device,
                      microbatch_tasks=job.microbatch_tasks, checkpoint_blocks=job.checkpoint_blocks,
                      cpu_threads=job.cpu_threads, selection=SELECTION, validation_split=False,
                      test_used_for_selection=False,
                      baseline={"name": "DsDTW", "1vs1": [4.04, 0.04], "4vs1": [2.54, 0.11]},
                      torch_version=torch.__version__, python_version=sys.version)
        # Bind resumption to source code as well as settings and data provenance.
        config["source_sha256"] = {
            name: hashlib.sha256((PROJECT / "src" / name).read_bytes()).hexdigest()
            for name in ("network_integer_tcn.py", "run_integer_tcn_experiment.py",
                         "dataset/datasetRealForgery.py", "optimAP.py", "main.py",
                         "model_factory.py", "integer_tcn_config.py", "train_integer_tcn.py",
                         "deepsigndb_data.py", "real_forgery_data.py", "evaluate_stylus.py",
                         "verifier_deepsign_central.py", "verifier_stylus_embedding.py",
                         "verifier_stylus_snn_tvs.py")}
        if train_args.model_name in training.integer_tcn_config.MAMBA_MODEL_NAMES:
            config["source_sha256"].update({
                name: hashlib.sha256((PROJECT / "src" / name).read_bytes()).hexdigest()
                for name in ("mamba2_torch.py", "network_integer_tcn_mamba.py")
            })
        if train_args.model_name == "integer-tcn-mamba2-interval":
            config["source_sha256"].update({
                name: hashlib.sha256((PROJECT / "src" / name).read_bytes()).hexdigest()
                for name in ("learnable_interval_spike.py", "network_interval_mamba.py")
            })
        status(dict(phase="loading_real_training_data"))
        dataset, loader = training.load_training_data(train_args)
        config["data_provenance"] = dataset.data_provenance
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if job.resume:
            recorded = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
            if recorded != config:
                raise ValueError("code, data or effective training configuration changed")
        else:
            atomic_json(result_dir / "run_config.json", config)
        lengths = np.asarray([len(feature) for feature in dataset.feats])
        logger.info("Data prepared: %d signatures, length min/median/max=%d/%d/%d, GPU=%s",
                    len(lengths), lengths.min(), np.median(lengths), lengths.max(),
                    torch.cuda.get_device_name(0) if job.device == "cuda" else "CPU")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        for seed in job.seeds:
            if any(item["seed"] == seed for item in state["completed"]):
                continue
            train_args.seed = seed
            seed_logger = make_logger(log_dir / ("seed%d-training.log" % seed), seed)
            best = train_seed(dataset, loader, train_args, job, seed_logger, status, config_hash)
            gc.collect()
            if job.device == "cuda":
                torch.cuda.empty_cache()
            outcomes = evaluate_seed(train_args, job, log_dir, result_dir, seed_logger, status)
            state["completed"].append(dict(seed=seed, best_train=best, evaluation=outcomes))
            atomic_json(result_dir / "summary.json", summarize(state["completed"]))
            status(dict(phase="seed_completed", seed=seed))
        status(dict(phase="completed", finished_at=now(), summary=summarize(state["completed"])))
        logger.info("EXPERIMENT COMPLETE %s", json.dumps(state["summary"]))
    except BaseException as exc:
        logger.exception("EXPERIMENT FAILED")
        status(dict(phase="failed", error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
