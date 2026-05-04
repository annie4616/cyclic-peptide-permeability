#!/usr/bin/env bash
# Launch MultiCycPermea training runs 2-at-a-time on GPUs 0 and 1.
#
# Phase 1: GPU0/cpu 0-63   = author_repro (Permeability target)
#          GPU1/cpu 64-127 = ours_ID     (PAMPA target)
# Phase 2: GPU0/cpu 0-63   = ours_OD     (PAMPA target)
#          GPU1/cpu 64-127 = ours_Cliff  (PAMPA target)
#
# EPOCHS defaults to 50.  Each run logs to logs/<run>.log and writes its PID
# to logs/<run>.pid so this script can wait on them.

set -euo pipefail

ROOT=/ssd0/sohyun/cyclic_peptide_permeability
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

EPOCHS=${EPOCHS:-50}

launch() {
    local gpu=$1
    local cpus=$2
    local run_name=$3
    local use_author=$4
    local split=$5
    local target=$6
    local logfile="$LOGDIR/${run_name}.log"
    local pidfile="$LOGDIR/${run_name}.pid"

    echo "[launch] GPU=$gpu CPUS=$cpus run=$run_name split=$split target=$target author=$use_author"
    echo "         -> $logfile"

    CUDA_VISIBLE_DEVICES=$gpu \
        CPU_AFFINITY=$cpus \
        USE_AUTHOR_SPLIT=$use_author \
        SPLIT=$split \
        TARGET=$target \
        EPOCHS=$EPOCHS \
        RUN_NAME=$run_name \
        nohup bash "$ROOT/run_mcp.sh" > "$logfile" 2>&1 &
    echo $! > "$pidfile"
    echo "[launch] pid=$(cat $pidfile)"
}

wait_for_pid() {
    local pidfile=$1
    local pid
    pid=$(cat "$pidfile")
    echo "[wait] waiting for pid $pid ($(basename "$pidfile"))"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "[wait] pid $pid finished"
}

STAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "Phase 1: author_repro (GPU0) + ours_ID (GPU1), $EPOCHS epochs each"
echo "========================================================================"
launch 0 0-63   "author_repro_${STAMP}" 1 ID    Permeability
launch 1 64-127 "ours_ID_${STAMP}"      0 ID    PAMPA

wait_for_pid "$LOGDIR/author_repro_${STAMP}.pid"
wait_for_pid "$LOGDIR/ours_ID_${STAMP}.pid"

echo "========================================================================"
echo "Phase 2: ours_OD (GPU0) + ours_Cliff (GPU1), $EPOCHS epochs each"
echo "========================================================================"
launch 0 0-63   "ours_OD_${STAMP}"    0 OD    PAMPA
launch 1 64-127 "ours_Cliff_${STAMP}" 0 Cliff PAMPA

wait_for_pid "$LOGDIR/ours_OD_${STAMP}.pid"
wait_for_pid "$LOGDIR/ours_Cliff_${STAMP}.pid"

echo "=== All 4 runs complete ==="
