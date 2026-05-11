#!/usr/bin/env bash
# Run eval_by_group on all completed runs (trajectory, centroid, trajectory_tri).
# GPU 3 (free); CPU upper half (192..N-1) to stay clear of the active GPU 7
# training driver.
set -uo pipefail

ROOT=/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet
cd "$ROOT"

TOTAL_CORES=$(nproc)
HALF_CORES=$(( TOTAL_CORES / 2 ))
THREADS=$(( HALF_CORES > 16 ? 16 : HALF_CORES ))

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

PYTHON=/home/sohyun/.conda/envs/chameleonnet/bin/python
CPU_START=$HALF_CORES
CPU_END=$(( TOTAL_CORES - 1 ))
TASKSET="taskset -c ${CPU_START}-${CPU_END}"

CONFIGS=(
    v2_id v2_od v2_od_murcko v2_cliff_ratio v2_cliff_pair
    v2_id_cent v2_od_cent v2_od_murcko_cent v2_cliff_ratio_cent v2_cliff_pair_cent
    v2_id_tri v2_od_tri v2_od_murcko_tri v2_cliff_ratio_tri v2_cliff_pair_tri
)

for tag in "${CONFIGS[@]}"; do
    cfg="configs/${tag}.yaml"
    out_dir="runs/${tag}"
    if [ ! -f "$out_dir/best.pt" ]; then
        echo "[$(date -Is)] skip $tag (no best.pt)"
        continue
    fi
    echo "[$(date -Is)] eval $tag"
    $TASKSET "$PYTHON" -m scripts.eval_by_group --config "$cfg" 2>&1 | tail -100
    echo "----"
done

echo "[$(date -Is)] all eval_by_group runs complete"
