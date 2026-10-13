#!/usr/bin/env bash
set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate zmj
cd /data/userdata/s_qnq/code/nature_communication/fedburst_sig
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs/integer_experiment_drivers
tag="genuine-ce-dtw192-s1111-20260916-v1"
log="logs/integer_experiment_drivers/$tag.log"
if [[ -e "results/$tag" || -e "$log" ]]; then
    echo 'Existing evaluation: inspect status and logs; refusing overwrite.'
    exit 1
fi
nohup python -u -B src/evaluate_genuine_ce_dtw192.py > "$log" 2>&1 &
printf 'Evaluation PID: %s\nLog: %s\n' "$!" "$log"
