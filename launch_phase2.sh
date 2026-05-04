#!/usr/bin/env bash
# Phase-2 launcher: starts OD + Cliff runs on GPUs 0 and 1 respectively
# once the earlier phase's runs finish.
#
# Usage: EPOCHS=50 bash launch_phase2.sh

set -euo pipefail

ROOT=/ssd0/sohyun/cyclic_peptide_permeability
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
EPOCHS=${EPOCHS:-50}
STAMP=$(date +%Y%m%d_%H%M%S)

launch() {
    local gpu=$1
    local cpus=$2
    local run_name=$3
    local split=$4
    local target=$5
    local logfile="$LOGDIR/${run_name}.log"
    local pidfile="$LOGDIR/${run_name}.pid"

    echo "[phase2] GPU=$gpu CPUS=$cpus run=$run_name split=$split target=$target"
    CUDA_VISIBLE_DEVICES=$gpu \
        CPU_AFFINITY=$cpus \
        USE_AUTHOR_SPLIT=0 \
        SPLIT=$split \
        TARGET=$target \
        EPOCHS=$EPOCHS \
        RUN_NAME=$run_name \
        nohup bash "$ROOT/run_mcp.sh" > "$logfile" 2>&1 &
    echo $! > "$pidfile"
    echo "[phase2] pid=$(cat "$pidfile")"
}

launch 0 0-63   "ours_OD_${STAMP}"    OD    PAMPA
launch 1 64-127 "ours_Cliff_${STAMP}" Cliff PAMPA

echo "[phase2] both runs started; monitor with: tail -f $LOGDIR/*.log"
