#!/usr/bin/env bash
# One-shot driver for the last 2 trajectory_tri runs with triplet OFF.
set -uo pipefail

ROOT=/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet
cd "$ROOT"

TOTAL_CORES=$(nproc)
HALF_CORES=$(( TOTAL_CORES / 2 ))
THREADS=$(( HALF_CORES > 32 ? 32 : HALF_CORES ))

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home/sohyun/.conda/envs/chameleonnet/bin/python
CPU_LAST=$(( HALF_CORES - 1 ))
TASKSET="taskset -c 0-${CPU_LAST}"

LOG_DIR="$ROOT/runs/_logs"
mkdir -p "$LOG_DIR"

run_one() {
    local cfg="$1"
    local tag="$2"
    local log="$LOG_DIR/${tag}.log"
    echo "[$(date -Is)] starting ${tag} (cfg=${cfg}); threads=${THREADS}, cpu_bind=0-${CPU_LAST}" | tee -a "$log"
    set +e
    $TASKSET "$PYTHON" -m scripts.train --config "$cfg" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    echo "[$(date -Is)] finished ${tag} rc=${rc}" | tee -a "$log"
}

run_one configs/v2_cliff_ratio_tri.yaml v2_cliff_ratio_tri
run_one configs/v2_cliff_pair_tri.yaml  v2_cliff_pair_tri

echo "[$(date -Is)] remaining 2 _tri runs complete"
