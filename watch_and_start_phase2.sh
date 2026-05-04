#!/usr/bin/env bash
# Poll the phase-1 pids every 2 minutes; once both have exited, launch phase 2.

set -euo pipefail
ROOT=/ssd0/sohyun/cyclic_peptide_permeability
LOGDIR="$ROOT/logs"

P1="$LOGDIR/author_repro_20260424_071132.pid"
P2="$LOGDIR/ours_ID_20260424_070856.pid"

echo "[watcher] waiting on $(cat $P1) and $(cat $P2)"
while :; do
    alive=0
    for f in "$P1" "$P2"; do
        pid=$(cat "$f" 2>/dev/null || echo 0)
        if [[ "$pid" != "0" ]] && kill -0 "$pid" 2>/dev/null; then
            alive=1
            break
        fi
    done
    if [[ "$alive" == "0" ]]; then
        break
    fi
    sleep 120
done

echo "[watcher] phase 1 done at $(date); launching phase 2"
EPOCHS=50 bash "$ROOT/launch_phase2.sh"
