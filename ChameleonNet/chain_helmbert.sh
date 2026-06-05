#!/usr/bin/env bash
# Wait for the running PeptideCLM seed sweep to finish, then launch the
# HELM-BERT seed sweep (seeds 1-4) with the same GPU/CPU layout:
#   GPU 0, cores 0-95   -> seeds 1,2
#   GPU 1, cores 96-191 -> seeds 3,4
set -uo pipefail
ROOT=/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet
cd "$ROOT"

echo "[chain] $(date -u +%H:%M:%S) waiting for PeptideCLM sweep to finish..."
# Block until no PeptideCLM seed-sweep training process remains.
while pgrep -f "scripts.train.*v1_od_murcko_traj_notri_seed[0-9]" >/dev/null 2>&1; do
  sleep 30
done
echo "[chain] $(date -u +%H:%M:%S) PeptideCLM sweep done. Launching HELM-BERT sweep."

CFG="$ROOT/configs/v1_od_murcko_traj_notri_helmbert.yaml"
PREFIX="v1_od_murcko_traj_notri_helmbert"

GPU=0 CORES=0-95   SEED_LIST="1 2" CFG="$CFG" NAME_PREFIX="$PREFIX" \
  nohup bash run_seed_sweep.sh > logs/seed_sweep_helmbert_gpu0.log 2>&1 &
echo "[chain] helmbert GPU0 driver PID $!"

GPU=1 CORES=96-191 SEED_LIST="3 4" CFG="$CFG" NAME_PREFIX="$PREFIX" \
  nohup bash run_seed_sweep.sh > logs/seed_sweep_helmbert_gpu1.log 2>&1 &
echo "[chain] helmbert GPU1 driver PID $!"

echo "[chain] $(date -u +%H:%M:%S) HELM-BERT sweep launched."
