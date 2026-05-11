#!/usr/bin/env bash
# Weighted-centroid runs across all 5 splits.
# GPU 7 only, CPU bound to second half of the cores (192..N-1) so we
# don't contend with the in-progress GPU-0 trajectory_tri run.
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
# Pin to the upper half of the cores so we don't fight the GPU-0 driver
# that's still bound to 0..HALF-1.
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

run_one configs/v2_id_centw.yaml          v2_id_centw
run_one configs/v2_od_centw.yaml          v2_od_centw
run_one configs/v2_od_murcko_centw.yaml   v2_od_murcko_centw
run_one configs/v2_cliff_ratio_centw.yaml v2_cliff_ratio_centw
run_one configs/v2_cliff_pair_centw.yaml  v2_cliff_pair_centw

echo "[$(date -Is)] all 5 weighted-centroid runs complete"
