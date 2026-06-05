"""Tiny GPU job queue: run a list of training configs across GPUs 0,1,2.

Each job = a config path (optionally with a seed). At most one job per GPU at a
time; when a GPU frees up the next queued job starts on it. CPU threads per job
are capped so total stays well under half of the machine's cores.

Usage:
    python scripts/gpu_queue.py job1.yaml job2.yaml ...            # seed from cfg
    python scripts/gpu_queue.py --seeds 42,1,2 jobA.yaml jobB.yaml # cross seeds
"""
from __future__ import annotations
import subprocess, sys, time, os, threading, queue, json
from pathlib import Path

ROOT = "/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet"
PY = "/home/sohyun/.conda/envs/chameleonnet/bin/python"
# GPUs from $QUEUE_GPUS (comma-sep), else default 0,1,2.
GPUS = [int(g) for g in os.environ.get("QUEUE_GPUS", "0,1,2").split(",") if g != ""]
THREADS_PER_JOB = 32          # jobs * 32 stays < 192 (half of 384 cores)
LOGDIR = Path(ROOT) / "runs" / "_logs" / "campaign"
LOGDIR.mkdir(parents=True, exist_ok=True)


def run_job(gpu: int, cfg: str, seed: int | None):
    name = Path(cfg).stem + (f"_seed{seed}" if seed is not None else "")
    out = f"{ROOT}/runs/campaign/{name}"
    log = LOGDIR / f"{name}.log"
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS=str(THREADS_PER_JOB), MKL_NUM_THREADS=str(THREADS_PER_JOB),
        OPENBLAS_NUM_THREADS=str(THREADS_PER_JOB), NUMEXPR_NUM_THREADS=str(THREADS_PER_JOB),
        TOKENIZERS_PARALLELISM="false",
    )
    cmd = [PY, "scripts/train.py", "--config", cfg,
           "--output_dir", out, "--wandb_run_name", name]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"# gpu={gpu} cfg={cfg} seed={seed}\n# cmd={' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    return name, rc, dt


def worker(gpu: int, jobs: "queue.Queue", results: list, lock: threading.Lock):
    while True:
        try:
            cfg, seed = jobs.get_nowait()
        except queue.Empty:
            return
        name, rc, dt = run_job(gpu, cfg, seed)
        with lock:
            results.append((name, rc, round(dt, 1)))
            print(f"[done gpu{gpu}] {name} rc={rc} {dt/60:.1f}min", flush=True)
        jobs.task_done()


def main():
    args = sys.argv[1:]
    seeds = [None]
    if args and args[0] == "--seeds":
        seeds = [int(s) for s in args[1].split(",")]
        args = args[2:]
    cfgs = args
    jobs: "queue.Queue" = queue.Queue()
    for cfg in cfgs:
        for s in seeds:
            jobs.put((cfg, s))
    n = jobs.qsize()
    print(f"[queue] {n} jobs over GPUs {GPUS} (seeds={seeds})", flush=True)
    results: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, jobs, results, lock)) for g in GPUS]
    for t in threads: t.start()
    for t in threads: t.join()
    print("\n[ALL DONE]")
    for name, rc, dt in sorted(results):
        print(f"  {name:42s} rc={rc} {dt/60:.1f}min")
    json.dump(results, open(LOGDIR / "queue_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
