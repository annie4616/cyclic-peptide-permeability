#!/usr/bin/env bash
# Multi-seed sweep on the OD_Murcko 0.300 split, v1.
# Runs the given seeds sequentially on one GPU; only `seed`, output_dir and
# wandb_run_name change per run.
#
# Parametrized by env:
#   GPU          CUDA device index            (default 0)
#   CORES        taskset core range, "0-95"   (default 0-95)
#   SEED_LIST    space-separated seeds        (default "1 2")
#   CFG          config yaml path             (default v1_od_murcko_traj_notri.yaml)
#   NAME_PREFIX  run-name / output-dir prefix (default v1_od_murcko_traj_notri)
set -uo pipefail

ROOT=/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet
PY=/home/sohyun/.conda/envs/chameleonnet/bin/python

GPU=${GPU:-0}
CORES=${CORES:-0-95}
SEED_LIST=${SEED_LIST:-"1 2"}
CFG=${CFG:-"$ROOT/configs/v1_od_murcko_traj_notri.yaml"}
NAME_PREFIX=${NAME_PREFIX:-"v1_od_murcko_traj_notri"}

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

for S in $SEED_LIST; do
  NAME="${NAME_PREFIX}_seed${S}"
  echo "============================================================"
  echo "[sweep gpu=$GPU cores=$CORES] START seed=$S -> runs/$NAME  $(date -u +%H:%M:%S)"
  echo "============================================================"
  taskset -c "$CORES" "$PY" -m scripts.train \
    --config "$CFG" \
    --seed "$S" \
    --output_dir "$ROOT/runs/$NAME" \
    --wandb_run_name "$NAME"
  echo "[sweep gpu=$GPU] DONE seed=$S rc=$? $(date -u +%H:%M:%S)"
done

echo "[sweep gpu=$GPU] COMPLETE prefix=$NAME_PREFIX seeds=[$SEED_LIST]"
