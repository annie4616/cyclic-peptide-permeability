#!/usr/bin/env bash
# _launch_one.sh GPU CPU_RANGE SPLIT TARGET EPOCHS BATCH RUN_NAME
# Spawns one MultiCycPermea run via run_mcp.sh in the background and writes its pid file.
set -euo pipefail
GPU=$1
CPUS=$2
SPLIT=$3
TARGET=$4
EPOCHS=$5
BATCH=$6
RUN_NAME=$7

ROOT=/ssd0/sohyun/cyclic_peptide/cyclic_peptide_permeability
cd "$ROOT"
mkdir -p logs
LOG="logs/${RUN_NAME}.log"
PIDF="logs/${RUN_NAME}.pid"

CUDA_VISIBLE_DEVICES=$GPU CPU_AFFINITY=$CPUS \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    USE_AUTHOR_SPLIT=0 SPLIT=$SPLIT TARGET=$TARGET EPOCHS=$EPOCHS BATCH_SIZE=$BATCH \
    RUN_NAME=$RUN_NAME \
    nohup bash "$ROOT/run_mcp.sh" > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "[launch] $RUN_NAME GPU=$GPU CPUS=$CPUS pid=$(cat "$PIDF") log=$LOG"
