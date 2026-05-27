#!/usr/bin/env bash
# Weighted-centroid + extended-descriptor runs across all 5 splits.
# GPU 7, CPU upper half. Triplet off (matches centw).
set -uo pipefail

ROOT=/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet
cd "$ROOT"

TOTAL_CORES=$(nproc)
HALF_CORES=$(( TOTAL_CORES / 2 ))
THREADS=$(( HALF_CORES > 32 ? 32 : HALF_CORES ))

export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home/sohyun/.conda/envs/chameleonnet/bin/python
CPU_START=$HALF_CORES
CPU_END=$(( TOTAL_CORES - 1 ))
TASKSET="taskset -c ${CPU_START}-${CPU_END}"

LOG_DIR="$ROOT/runs/_logs"
mkdir -p "$LOG_DIR"

run_one() {
    local cfg="$1"
    local tag="$2"
    local log="$LOG_DIR/${tag}.log"
    echo "[$(date -Is)] starting ${tag} (cfg=${cfg}); GPU=7, threads=${THREADS}, cpu_bind=${CPU_START}-${CPU_END}" | tee -a "$log"
    set +e
    $TASKSET "$PYTHON" -m scripts.train --config "$cfg" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    echo "[$(date -Is)] finished ${tag} rc=${rc}" | tee -a "$log"
}

run_one configs/v2_id_centwext.yaml          v2_id_centwext
run_one configs/v2_od_centwext.yaml          v2_od_centwext
run_one configs/v2_od_murcko_centwext.yaml   v2_od_murcko_centwext
run_one configs/v2_cliff_ratio_centwext.yaml v2_cliff_ratio_centwext
run_one configs/v2_cliff_pair_centwext.yaml  v2_cliff_pair_centwext

echo "[$(date -Is)] all 5 weighted-centroid + extended-descriptor runs complete"
