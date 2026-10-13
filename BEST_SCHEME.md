# Best verified integer-activation signature scheme

Archived on 2026-10-13 from the completed September 2026 experiments.
This code snapshot does not launch training or include datasets, checkpoints,
logs, signature features, or per-signature scores.

## Fixed method

- DeepSignDB stylus, centralized, real skilled forgeries (SF); 528 training
  writers and 442 evaluation writers; seed 1111.
- 12-dimensional input; width-128 causal residual TCN (kernel 5, dilations
  1/2/4/8), followed by two Mamba2-style blocks; 256-dimensional embedding.
- TCN: eight unsigned integer-count activations, range 0..4, clipped STE.
- Mamba: four signed learnable-interval activations at input/output projection
  boundaries, range -4..4; original normalized-triangle gradients for input
  and thresholds. `interval_tcn=false`, `interval_sites=both`.
- AP uses the complete episode; identity CE uses only genuine signatures.
  100 epochs, AdamW; select the lowest complete-epoch mean training loss.
  No held-out validation set. Best recorded checkpoint: epoch 99.
- DTW: interpolate local features to 192 points, per-vector L2 normalization,
  cosine cost, relative band 0.25, minimum-cost path divided by path length.
  Original min/mean reference scoring and globally pooled EER are preserved.

These are integer-valued activations with floating-point arithmetic, not a
fully integer backend or membrane-state ILIF dynamics. Local DTW features
remain continuous.

## Recorded results (EER %)

| Model / matcher | SF 1vs1 | SF 4vs1 |
|---|---:|---:|
| Integer activation / embedding | 7.8224229652 | 4.8261298815 |
| Integer activation / pointwise192 | 15.5868930279 | 11.3929150376 |
| **Integer activation / DTW192** | **5.0392309967** | **2.7627150337** |
| Integer activation / fusion alpha=0.5 | 4.9581800966 | 3.3533072136 |

The selected scheme uses pure DTW, not fusion. Results are single-seed,
evaluation-set-exploratory observations, not significance claims. Official
Comparison Files have not been checked pair by pair. External paper results
must not be described as strictly matched reproductions.

Training run: `dtw192-genuinece-s1111-20260916-v1-genuine-ce`.
Evaluation run: `genuine-ce-dtw192-s1111-20260916-v1`.
Checkpoint SHA256:
`6353f9102a6620ae8d1d00cddf1a8c2861610e717f5ec2bcc6456692bab59308`.
Matched evaluator SHA256:
`4b208b5b16a5558faf011e59ce5e26258e5d9e65c22cdeb64f91dbc5a656b054`.

See [GENUINE_CE_DTW192.md](GENUINE_CE_DTW192.md) for the historical evaluation,
audits and scoped temporary-export cleanup.

## Code entry points

| Purpose | File |
|---|---|
| Fixed model / loss configuration | `configs/mamba_genuine_only_ce.json` |
| Genuine-only CE training | `src/run_genuine_ce_experiment.py` |
| TCN and Mamba backbone | `src/network_integer_tcn.py`, `src/network_integer_tcn_mamba.py` |
| Interval activation model | `src/network_interval_mamba.py`, `src/learnable_interval_spike.py` |
| Portable parallel Mamba backend | `src/mamba2_torch.py` |
| Checkpoint-compatible feature export | `src/evaluate_stylus.py` |
| Audited shared scorer | `src/verify_fusion_matched.py` |
| Historical fixed-checkpoint DTW192 job | `src/evaluate_genuine_ce_dtw192.py` |
| Project-scoped export cleanup | `src/scoped_export_cleanup.py` |

## Running with separately provisioned data

Obtain DeepSignDB through its authorized distribution. Existing preprocessed
files and manifests belong under `data_real_forgery/deepsigndb_stylus_v1/`;
they are intentionally not in Git. Preparation from existing source pickles
is available through `src/prepare_deepsigndb_real.py --help` and documented in
[DEEPSIGNDB_CENTRAL_BASELINES.md](DEEPSIGNDB_CENTRAL_BASELINES.md).

On the designated server, use environment `zmj` and physical GPU 1 only:

```bash
conda activate zmj
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg
python -u -B src/run_genuine_ce_experiment.py \
  --config configs/mamba_genuine_only_ce.json \
  --run-tag genuine-ce-reproduction-s1111 \
  --seeds 1111 --device cuda --microbatch-tasks 1 --cpu-threads 4
```

Use a new run tag; do not overwrite historical results. This archived trainer
first evaluates DTW96. The selected result additionally requires DTW192:
`verify_fusion_matched.py --length 192 --band 0.25`. The configuration's
`primary=dtw96` is historical and does not describe the final DTW192 result.

`evaluate_genuine_ce_dtw192.py` is an integrity-locked historical replay,
not a generic evaluator for newly trained weights. It requires the exact
checkpoint, previous result reports, baseline artifacts and export files
named in the script; a code-only clone cannot execute it immediately.
It refuses mismatched hashes and existing output directories. Keep these
checks; do not change hashes simply to bypass them.

For a new run, use `evaluate_stylus.py` to export that run's local features,
then `verify_fusion_matched.py` with its corresponding embedding export and
audit reference. Both embedding-only audits must pass before accepting DTW
results. The generic exporter does not automatically remove exports; only
the guarded historical runners perform scoped, audited cleanup.

## Scope and attribution

Archive validation on 2026-10-13: exported only the Git-staged files into an
isolated project-local directory and ran all eight included test modules on
CPU: **108 passed, 1 skipped** (Windows symbolic-link permissions). This
checks code packaging, model/gradient behavior, data handling, checkpoint
compatibility, matched scoring and cleanup guards; it does not rerun the
full training or reproduce EER without the separately retained artifacts.

This initial Git snapshot includes the selected centralized experiment and
its shared runtime dependencies/tests, not the entire historical federated,
synthetic-forgery or failed-ablation workspace. Legacy branches may remain
inside shared modules but are not the supported reproduction entry points.

Research context: [DsDTW](https://github.com/KAKAFEI123/DsDTW),
[SynSig2Vec](https://github.com/LaiSongxuan/SynSig2Vec), and
[SpikeYOLO](https://github.com/BICLab/SpikeYOLO). Preserve existing source
attribution; no new blanket license or ownership claim is added by this
archive. HoLoSig research code is not included.
