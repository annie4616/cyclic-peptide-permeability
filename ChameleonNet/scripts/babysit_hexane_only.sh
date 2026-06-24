#!/usr/bin/env bash
# Wait for all four runs to finish, then summarize performance and sync each
# offline wandb run to the cloud.
#   hexane_only:  OD PID 2499515 (gpu5), ID PID 2501834 (gpu4)
#   xfmr_pool:    OD PID 2515148 (gpu6), ID PID 2515149 (gpu7)
set -uo pipefail
ROOT=/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet
PY=/home/sohyun/.conda/envs/chameleonnet/bin/python
cd "$ROOT"

PIDS=${PIDS:-"2499515 2501834 2515148 2515149"}

echo "[babysit] $(date -u +%H:%M:%S) waiting for PIDs: $PIDS"
while :; do
  any=0
  for p in $PIDS; do kill -0 "$p" 2>/dev/null && any=1; done
  [ "$any" = "0" ] && break
  sleep 30
done
echo "[babysit] $(date -u +%H:%M:%S) all runs finished."

echo "[babysit] summarizing..."
"$PY" scripts/summarize_hexane_only.py

echo "[babysit] syncing wandb offline runs..."
export WANDB_MODE=online
for d in runs/final_extdesc_hexane_only/wandb/offline-run-* \
         runs/final_extdesc_hexane_only_id/wandb/offline-run-* \
         runs/final_extdesc_xfmrpool/wandb/offline-run-* \
         runs/final_extdesc_xfmrpool_id/wandb/offline-run-*; do
  [ -d "$d" ] || continue
  echo "[babysit] wandb sync $d"
  "$PY" -m wandb sync "$d"
done
echo "[babysit] $(date -u +%H:%M:%S) DONE."
