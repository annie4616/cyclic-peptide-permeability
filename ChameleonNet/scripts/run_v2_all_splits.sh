#!/usr/bin/env bash
# Run ChameleonNet V2 on ID, OD, Cliff_ratio, Cliff_pair splits sequentially.
# GPU: 0 only. CPU: capped to <= half of available cores.
# Note: deliberately NOT using `set -e` — a failure in one split should be
# logged but must not block the remaining splits.
set -uo pipefail

ROOT=/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet
cd "$ROOT"

TOTAL_CORES=$(nproc)
HALF_CORES=$(( TOTAL_CORES / 2 ))
# Cap thread count below half. Leave headroom.
THREADS=$(( HALF_CORES > 32 ? 32 : HALF_CORES ))

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pin to the conda env python so the script works regardless of whether the
# parent shell has the env active. rc=127 means the bare `python` lookup
# missed in PATH — has bitten this script once already.
PYTHON=/home/sohyun/.conda/envs/chameleonnet/bin/python

# Bind the process tree to the first HALF_CORES CPUs as a hard cap.
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
    set -e 2>/dev/null || true
    echo "[$(date -Is)] finished ${tag} rc=${rc}" | tee -a "$log"
}

# Trajectory runs already completed: v2_od, v2_cliff_ratio, v2_cliff_pair.
# Remaining: trajectory v2_od_murcko + 5 centroid + 5 trajectory_tri = 11 runs.
run_one configs/v2_od_murcko.yaml   v2_od_murcko

# Cluster-centroid runs across all 5 split schemes.
# Triplet uses r=4, n_bits=4096, sim_high=0.5 (configured in each cent yaml)
# so mined pairs reflect real near-neighbors instead of fingerprint-collision noise.
run_one configs/v2_id_cent.yaml          v2_id_cent
run_one configs/v2_od_cent.yaml          v2_od_cent
run_one configs/v2_od_murcko_cent.yaml   v2_od_murcko_cent
run_one configs/v2_cliff_ratio_cent.yaml v2_cliff_ratio_cent
run_one configs/v2_cliff_pair_cent.yaml  v2_cliff_pair_cent

# Trajectory + improved-triplet re-runs across all 5 split schemes.
run_one configs/v2_id_tri.yaml          v2_id_tri
run_one configs/v2_od_tri.yaml          v2_od_tri
run_one configs/v2_od_murcko_tri.yaml   v2_od_murcko_tri
run_one configs/v2_cliff_ratio_tri.yaml v2_cliff_ratio_tri
run_one configs/v2_cliff_pair_tri.yaml  v2_cliff_pair_tri

echo "[$(date -Is)] all 11 remaining runs complete"
