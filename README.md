# Signature SNN: best verified centralized scheme

Start with [BEST_SCHEME.md](BEST_SCHEME.md) for the archived best scheme:
**integer-activation TCN + Mamba, AP + genuine-only CE, local DTW192**.
Recorded DeepSignDB stylus skilled-forgery EER: **5.0392% (1vs1), 2.7627% (4vs1)**.
This is mixed floating-point computation with integer-valued activations,
not full-integer inference. Data, weights and generated outputs are excluded.

The material below is the original workspace documentation. Its federated,
synthetic and historical branches are not all included in this focused Git
snapshot, and its early experiment-status statements are historical.

# FedBurst-Sig Standalone Project (historical documentation)

This subproject packages the **formal SNN method** (`FedBurst-Sig`) together with its federated training strategy and evaluation flow.

It is intended to be distributable as a standalone directory. All runner scripts target the local `src/`, `data/`, `configs/`, and `sigma_lognormal/` folders inside `projects/fedburst_sig/`.

For the full step-by-step process from raw data to final tables, see `PROJECT_WORKFLOW.md`.

## Independent Integer-TCN experiment

The new `integer-tcn` backend implements **S=1 integer counts + parallel causal
residual TCN**, inspired by SpikeYOLO's clipped-STE quantizer. It is separate
from the formal FedBurst method and from the inhibitory `ilif` comparison.
See [INTEGER_TCN_PLAN.md](INTEGER_TCN_PLAN.md) for the recorded design, limitations,
configuration, training/export commands and tests. The default configuration is
`configs/integer_tcn.json`; start with `python src/train_integer_tcn.py --help`.
No existing training results or formal federated runner defaults are replaced.
Integer-TCN now defaults to **centralized DeepSignDB stylus real-forgery training**,
using the DsDTW 528-writer development subset and independent copies under
`data_real_forgery/deepsigndb_stylus_v1/`. See
[DeepSignDB protocol and ANN baselines](DEEPSIGNDB_CENTRAL_BASELINES.md).
No SLN parameters or synthetic caches are needed. The MSDS copy and preset
`configs/integer_tcn_msds.json` are retained for later work. The original `data/`,
synthetic pipeline and formal forgery-free FedBurst defaults are unchanged.

An independent `integer-tcn-mamba2` variant adds two continuous Mamba-2-style
mixers after the integer TCN. Its portable PyTorch backend requires no new
packages on the current Windows environment. See [MAMBA2_VARIANT.md](MAMBA2_VARIANT.md)
and `configs/integer_tcn_mamba2.json`. It is not a fully spiking Mamba or the
official fused backend; no new formal training has been launched.

The independent `integer-tcn-mamba2-interval` prototype adds signed count
activations with ordered learnable threshold intervals at Mamba projection
boundaries. It includes fixed-interval controls and interval diagnostics.
See [LEARNABLE_INTERVAL_MAMBA.md](LEARNABLE_INTERVAL_MAMBA.md) and
`configs/integer_tcn_mamba2_interval.json`. Its SSM arithmetic remains floating;
no new EER or full-integer inference result is claimed.

Included:

- centralized training and evaluation
- federated non-IID local-only training
- common verifier, TVS verifier, and TVS+HN verifier
- one-shot hard-negative bank construction after training
- fixed five-seed protocol: `1111,2222,3333,4444,5555`

Excluded:

- ANN baselines
- historical SGR branches
- burstk ablation variants
- all-learnable comparison variants

## Formal method

FedBurst-Sig uses the formal SNN method only:

- channel-wise learnable tau
- burst spike with raw count output
- burst K per layer: `2,2,2,5,8,10`
- nonlinear thresholds per layer: `1,2;1,2;1,2;1,2,3,5,7;1,2,3,4,6,8,10,12;1,2,3,4,5,7,9,11,13,15`
- learnable threshold intervals in the final burst layer only
- identity straight-through estimator (STE) for every hard spike backward pass
- the historical triangular surrogate remains available with `--spike-gradient surrogate`;
  its sigma is `1.0` for centralized and `2.5` for non-IID training
- no SGR in the formal method

## Federated strategy

Training is **pure local**: each client trains independently with no parameter sharing during optimization.

The formal TVS+HN path performs cross-client sharing only after training:

1. export each client's real-genuine and Sigma--Lognormal synthetic-forgery embeddings
2. build a one-shot hard-negative bank from those writer-disjoint synthetic candidates
3. run `TVS+HN-Synthetic` verification using the shared bank

No real skilled forgery is consumed by this default path, so it remains fully
forgery-free.  `--hn-source real-dev` is retained only as the separately named
`TVS+HN-Real-dev` calibration-assisted experiment; it must not be reported as the
main forgery-free result.

## FedSNN baseline

The project also includes an optional FedSNN-style parameter-aggregation baseline. It uses
random client participation, local epochs, straggler simulation, update-scaled Gaussian
noise, and unweighted FedAvg. Because signature clients own disjoint writer identities,
their classification heads are personalized; FedAvg is applied to the shared SNN encoder
and neuron parameters only.

```powershell
.\projects\fedburst_sig\scripts\train_fedsnn.ps1 `
  -Partition fed_partition_alpha0.5_K5.json `
  -Rounds 100 -LocalEpochs 2 -ClientFraction 1.0
```

Checkpoints are written under `src/models/<run-tag>/<seed>/round_NNNN/`. Each round stores
the global shared state, every personalized client model, and participation/deviation metrics.

Interrupted federated runs can resume from the latest complete round. Checkpoints store
trainer/RNG state; Local-only v2 checkpoints additionally store each client's persistent
optimizer and momentum. Legacy Local-only checkpoints cannot resume exactly and must be
retrained under a v2 run tag.

```bash
RESUME=1 CHECKPOINT_INTERVAL=10 METHOD=fedavg SEED=3333 \
  RUN_TAG=fedburst-ste-v3-alpha01-fedavg SPIKE_GRADIENT=ste \
  bash scripts/train_federated.sh
```

STE changes only the backward estimator: the hard binary/burst forward values
and checkpoint tensor schema are unchanged. Do not resume a triangular-surrogate
run with STE; the trainer records the gradient mode and rejects mixed resumes.

`CHECKPOINT_INTERVAL=10` updates lightweight `history.json` every round, saves full models every
10 rounds, and always saves the final round. This is recommended for multi-method, multi-seed runs.

The formal horizontal-FL matrix contains exactly five strategies:

- `fedavg`: sample-count weighted FedAvg
- `fedprox`: weighted FedAvg plus a shared-encoder proximal objective
- `fednova`: momentum-aware normalized averaging from the official FedNova implementation
- `fedsnn`: FedSNN-style unweighted averaging with optional stragglers/update noise
- `fedsnn-sparse`: magnitude-only **FedSNN-Sparse (corrected)**

`local-only` is retained separately as a non-federated lower bound. All six protocols keep
the writer-specific classifier local and compare the same FedBurst-Sig encoder. Shared model
states are canonicalized before aggregation so repeated `spike_neurons`/`convNet` aliases are
not uploaded or counted twice.

`scaffold`, `fedlec-forced`, `flea-adapted`, and `bkt-dsnn-adapted` are excluded from the
formal trainer, batch runners, monitors, and evaluator. Historical checkpoint directories are
not deleted, but they are not included in formal result tables.

See `BASELINE_SOURCES.md` for upstream papers, local source files, restored behavior,
and every signature-task adaptation.

Run every horizontal baseline with the same protocol:

```powershell
.\projects\fedburst_sig\scripts\run_federated_baselines.ps1 `
  -Partition fed_partition_alpha0.5_K5.json `
  -Rounds 100 -LocalEpochs 2
```

Linux single-method run:

```bash
METHOD=fedavg SEED=1111 ROUNDS=100 LOCAL_EPOCHS=2 \
RUN_TAG=fedburst-v2-alpha05-fedavg \
  bash projects/fedburst_sig/scripts/train_federated.sh
```

Linux full baseline matrix:

```bash
bash projects/fedburst_sig/scripts/run_federated_baselines.sh
```

Formal evaluation defaults to the five global federated encoders; optionally add
the five Local-only client models as a separate control group:

```powershell
.\scripts\eval_federated_baselines.ps1 `
  -RunPrefix fedburst-v2-alpha05 -IncludeLocalOnly
.\scripts\summarize_federated_results.ps1 `
  -Logs logs/federated-v2-eval.log
```

The federated summary first averages Local-only clients within each seed and then
reports the five-seed mean and sample standard deviation. Historical adapted method
tags are ignored.

## Scripts

- `scripts/check_env.ps1`
- `scripts/prepare_dataset.ps1`
- `scripts/extract_sln_params.ps1`
- `scripts/build_partition.ps1`
- `scripts/train_central.ps1`
- `scripts/train_niid.ps1`
- `scripts/eval_central.ps1`
- `scripts/eval_niid.ps1`
- `scripts/build_hn_niid.ps1`
- `scripts/eval_tvs_hn_niid.ps1`
- `scripts/eval_federated.ps1`
- `scripts/eval_federated_baselines.ps1`
- `scripts/summarize_federated_results.ps1`
- `scripts/build_hn_federated.ps1`
- `scripts/eval_tvs_hn_federated.ps1`
- `scripts/summarize_results.ps1`
- `scripts/run_all_niid.ps1`

## Quick start

### First check the environment

```powershell
.\projects\fedburst_sig\scripts\check_env.ps1
```

### Build a non-IID partition from the local training pickles

```powershell
.\projects\fedburst_sig\scripts\build_partition.ps1
```

### Prepare dataset pickles from a manifest

```powershell
.\projects\fedburst_sig\scripts\prepare_dataset.ps1 -Manifest configs/dataset_manifest.example.json
```

### Extract sigma-lognormal parameters

```powershell
.\projects\fedburst_sig\scripts\extract_sln_params.ps1 -Config configs/sln_extraction_config.example.json
```

### Centralized

```powershell
.\projects\fedburst_sig\scripts\train_central.ps1
.\projects\fedburst_sig\scripts\eval_central.ps1
```

### Non-IID

```powershell
.\projects\fedburst_sig\scripts\train_niid.ps1 -Partition fed_partition_alpha0.5_K5.json
.\projects\fedburst_sig\scripts\eval_niid.ps1
.\projects\fedburst_sig\scripts\build_hn_niid.ps1
.\projects\fedburst_sig\scripts\eval_tvs_hn_niid.ps1
```

Or run the full non-IID pipeline:

```powershell
.\projects\fedburst_sig\scripts\run_all_niid.ps1 -Partition fed_partition_alpha0.5_K5.json
```

### Summarize results

```powershell
.\projects\fedburst_sig\scripts\summarize_results.ps1 -CentralLog <central-stdout.log> -NiidLog <niid-stdout.log> -NiidTvsHnLog <tvs-hn-stdout.log>
```

## Adapting to a new dataset

To reuse FedBurst-Sig on another dataset, keep the method fixed and replace only the dataset-facing assets:

1. dataset preparation pipeline
2. train/eval pickle files in `data/`
3. sigma-lognormal parameter folders in `sigma_lognormal/params/`
4. user/client partition file
5. optional run prefix for experiment naming

See `DATASET_ADAPTATION.md` for the exact processing and classification assumptions.

The bundled `src/` directory already contains the Python entrypoints and verifier logic required by the formal method. To adapt to a new dataset, you primarily update the local `data/`, `sigma_lognormal/params/`, and partition files.

## Default seeds

The default five-seed protocol is stored in `configs/seeds.txt`:

- `1111`
- `2222`
- `3333`
- `4444`
- `5555`

## Local-only baseline

`local-only` uses the same five client partitions, initialization, optimizer,
100-round schedule, and two local epochs per round as the federated runs, but
performs no client-to-server upload, server aggregation, or server-to-client
overwrite.  Client models therefore continue training independently across
rounds.  The final reusable client weights are stored as
`src/models/<run-prefix>-local-only/<seed>/round_0100/client_<id>.pt`.

Run all three partitions (`alpha=0.5`, `alpha=0.1`, and IID) for one dataset:

```bash
bash scripts/run_local_only_dataset.sh original
bash scripts/run_local_only_dataset.sh msds
bash scripts/run_local_only_dataset.sh gpds
```

The three commands can be assigned to separate GPUs with
`CUDA_VISIBLE_DEVICES`.  Runs resume by default and save full client weights
every 10 rounds plus the final round.

## DeepSignDB 已核验方案记录

当前最优模型与匹配器的对应组合、检查点和结果证据见
[BEST_VERIFIED_SCHEMES.md](BEST_VERIFIED_SCHEMES.md)。服务器实验遵循 `AGENTS.md`，
仅使用物理 GPU 1。
