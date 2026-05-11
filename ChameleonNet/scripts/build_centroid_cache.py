"""Pre-compute cluster-centroid conformer ensembles from MD logs + trajectories.

For each peptide we:
  1. Parse the gromos clustering log (Hexane/Logs, Water/Logs) to get
     (cluster_size, middle_frame_time_ns) per cluster.
  2. Parse the trajectory PDB to get coords for all frames and a ns-per-frame
     map from each frame's TITLE timestamp.
  3. Look up the closest frame to each cluster's middle-time, take its coords,
     and stack into (K_clusters, N_atoms, 3).
  4. Save .npz with {coords, z, res, weights} where weights[i] = cluster_size[i].

Output layout (mirrors Trajectories/):
  CycPeptMPDB-4D/Water/Centroids/{Source}_{pid}_H2O_Cent.npz
  CycPeptMPDB-4D/Hexane/Centroids/{Source}_{pid}_Hexane_Cent.npz

Run:
  python -m scripts.build_centroid_cache \
      --csv  /hdd0/.../data/CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv \
      --pdb-root /hdd0/.../CycPeptMPDB-4D \
      --workers 64
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from chameleonnet.data.pdb_parser import parse_pdb
from chameleonnet.data.residue_vocab import ResidueVocab


_CL_HEADER_RE = re.compile(r"^\s*\d+\s*\|")
_TITLE_T_RE = re.compile(r"^TITLE.*?t=\s*([0-9.eE+-]+)", re.IGNORECASE)


def parse_cluster_log(log_path: Path) -> Optional[List[Tuple[int, float]]]:
    """Return list of (cluster_size, middle_time_ns) ordered by cluster id.

    The log table looks like:
      cl. | #st  rmsd | middle rmsd | cluster members
        1 |  42  0.094 |     32 .078 |   26 ...
        2 |  16  0.100 |     41 .090 |   26.3 ...
       12 |   1       |   29.6      |   29.6
    """
    try:
        text = log_path.read_text()
    except FileNotFoundError:
        return None

    out: List[Tuple[int, float]] = []
    in_table = False
    for raw in text.splitlines():
        if raw.lstrip().startswith("cl."):
            in_table = True
            continue
        if not in_table:
            continue
        # The table rows start with a cluster id followed by |.
        # Continuation lines (member lists) start with whitespace then |.
        m = _CL_HEADER_RE.match(raw)
        if not m:
            continue
        # split on |
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 4:
            continue
        try:
            cl_id = int(parts[0])
        except ValueError:
            continue
        # parts[1] = "#st rmsd" e.g. "42  0.094" or "1"
        size_tok = parts[1].split()
        try:
            size = int(size_tok[0])
        except (IndexError, ValueError):
            continue
        # parts[2] = "middle rmsd" e.g. "32 .078" or "29.6" (singletons have no rmsd)
        mid_tok = parts[2].split()
        try:
            middle_ns = float(mid_tok[0])
        except (IndexError, ValueError):
            continue
        out.append((size, middle_ns))
        # sanity: cluster ids increment by 1
        if cl_id != len(out):
            # log is malformed; bail
            return None
    return out if out else None


def title_times_ns(pdb_path: Path) -> List[float]:
    """Read each TITLE line's t= value, in ns (the file stores ps)."""
    times: List[float] = []
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("TITLE"):
                continue
            m = _TITLE_T_RE.match(line)
            if not m:
                continue
            try:
                times.append(float(m.group(1)) / 1000.0)  # ps -> ns
            except ValueError:
                continue
    return times


def nearest_frame(times_ns: np.ndarray, target_ns: float) -> int:
    return int(np.argmin(np.abs(times_ns - target_ns)))


def build_one(
    pid: int,
    source: str,
    pdb_root: Path,
    vocab_paths: Tuple[str, ...],
    overwrite: bool,
) -> Tuple[int, str]:
    """Process one peptide (both envs). Returns (pid, status)."""
    vocab = ResidueVocab.from_csvs(*vocab_paths)
    base = f"{source}_{pid}"
    statuses = []
    for env, ext_pdb, ext_log in (
        ("Water", "_H2O_Traj.pdb", "_H2O.log"),
        ("Hexane", "_Hexane_Traj.pdb", "_Hexane.log"),
    ):
        traj = pdb_root / env / "Trajectories" / f"{base}{ext_pdb}"
        log = pdb_root / env / "Logs" / f"{base}{ext_log}"
        out_dir = pdb_root / env / "Centroids"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{base}_{ 'H2O' if env=='Water' else 'Hexane'}_Cent.npz"

        if out_path.exists() and not overwrite:
            statuses.append(f"{env}=skip")
            continue
        if not traj.exists() or not log.exists():
            statuses.append(f"{env}=missing")
            continue

        clusters = parse_cluster_log(log)
        if not clusters:
            statuses.append(f"{env}=log-empty")
            continue
        try:
            coords, z, res, _ = parse_pdb(traj, vocab, max_models=None)
        except Exception as e:
            statuses.append(f"{env}=parse-fail({type(e).__name__})")
            continue
        times = title_times_ns(traj)
        if len(times) != coords.shape[0]:
            # Fall back to evenly-spaced times spanning the trajectory if TITLE
            # parsing fell short (rare).
            times = list(np.linspace(0.0, float(coords.shape[0] - 1), coords.shape[0]))
        times_arr = np.asarray(times, dtype=np.float32)

        sizes = np.asarray([c[0] for c in clusters], dtype=np.int32)
        mids = np.asarray([c[1] for c in clusters], dtype=np.float32)
        idxs = np.asarray([nearest_frame(times_arr, m) for m in mids], dtype=np.int64)
        cent_coords = coords[idxs]  # (K, N, 3)

        np.savez_compressed(
            out_path,
            coords=cent_coords.astype(np.float32),
            z=z.astype(np.int64),
            res=res.astype(np.int64),
            weights=sizes.astype(np.float32),
            frame_idx=idxs.astype(np.int64),
            mid_ns=mids.astype(np.float32),
        )
        statuses.append(f"{env}=ok({len(idxs)})")
    return pid, ",".join(statuses)


def iter_peptides(csv_path: Path) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["CycPeptMPDB_ID"])
            except (KeyError, ValueError):
                continue
            src = (row.get("Source") or "").strip()
            if src:
                out.append((pid, src))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--pdb-root", required=True, type=Path)
    ap.add_argument(
        "--vocab",
        nargs="+",
        default=[
            "/hdd0/sohyun/cyclic-peptide-permeability/eda/water_residue_vocab.csv",
            "/hdd0/sohyun/cyclic-peptide-permeability/eda/hexane_residue_vocab.csv",
        ],
    )
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N peptides (debug).")
    args = ap.parse_args()

    peptides = iter_peptides(args.csv)
    if args.limit > 0:
        peptides = peptides[: args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] {len(peptides)} peptides; {args.workers} workers")

    vocab_tuple = tuple(args.vocab)
    pdb_root = args.pdb_root
    overwrite = args.overwrite

    ok = miss = fail = skip = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(build_one, pid, src, pdb_root, vocab_tuple, overwrite)
            for pid, src in peptides
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            pid, status = fut.result()
            if "missing" in status:
                miss += 1
            elif "fail" in status or "log-empty" in status:
                fail += 1
            elif "skip" in status and "ok" not in status:
                skip += 1
            else:
                ok += 1
            if i % 200 == 0 or i == len(peptides):
                dt = time.time() - t0
                rate = i / max(dt, 1e-6)
                print(
                    f"[{time.strftime('%H:%M:%S')}] {i}/{len(peptides)} "
                    f"ok={ok} skip={skip} miss={miss} fail={fail} "
                    f"({rate:.1f}/s, last pid={pid}: {status})"
                )

    print(
        f"done. ok={ok} skip={skip} miss={miss} fail={fail} "
        f"elapsed={time.time()-t0:.0f}s"
    )


if __name__ == "__main__":
    main()
