"""Round-0 classic-ML screen on descriptors for the OD_Murcko (0.300) split.

Permeability is often predicted well OOD by descriptor-based models; this gives
a fast, strong reference for the deep-model campaign. We train on the OD_Murcko
train ids and evaluate on its test ids, reporting MAE/MSE/R2/Pearson. Several
models are run; tree models are repeated over 3 seeds for mean/std.

Leakage guard: every assay/permeability-derived column is dropped from the
feature set — we keep only physicochemical + MD (4D) descriptors.
"""
from __future__ import annotations
import csv, json, sys
import numpy as np

CSV = "/hdd0/sohyun/cyclic-peptide-permeability/data/CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv"
SPLITS = "/hdd0/sohyun/cyclic-peptide-permeability/splits"
SCHEME = "OD_Murcko"
TARGET = "PAMPA"

# Columns that leak the label or are non-numeric identifiers — never features.
LEAK = {
    "CycPeptMPDB_ID", "Source", "Original_Name_in_Source_Literature",
    "Structurally_Unique_ID", "SMILES", "HELM", "Sequence", "HELM_URL",
    "Year", "Version", "Molecule_Shape",
    # label + assay-derived (leakage)
    "PAMPA", "PAMPA-4D", "Permeability", "Caco2", "MDCK", "RRCK",
    "R_PAMAP", "R_Caco2", "R_MDCK", "R_RRCK", "T_PAMPA",
    "Detection_Limit_1", "Detection_Limit_2",
}


def load_ids(fold):
    out = []
    with open(f"{SPLITS}/{SCHEME}_{fold}_ids.csv") as f:
        for r in csv.DictReader(f):
            out.append(int(r["CycPeptMPDB_ID"]))
    return set(out)


def load_table():
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    header = [c for c in rows[0].keys()]
    feat_cols = [c for c in header if c not in LEAK]

    def fnum(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return np.nan

    data = {}
    for r in rows:
        try:
            pid = int(r["CycPeptMPDB_ID"])
        except (KeyError, ValueError):
            continue
        y = fnum(r.get(TARGET, ""))
        if not np.isfinite(y):
            continue
        x = np.array([fnum(r.get(c, "")) for c in feat_cols], dtype=np.float64)
        data[pid] = (x, y)
    return feat_cols, data


def make_xy(ids, data):
    pids = [p for p in ids if p in data]
    X = np.stack([data[p][0] for p in pids])
    y = np.array([data[p][1] for p in pids])
    return X, y


def _rank(a):
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(c)); np.add.at(s, inv, r)
    return (s / c)[inv]


def metrics(yt, yp):
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    var = float(np.var(yt))
    r2 = float(1 - mse / var) if var > 0 else float("nan")
    ok = yt.std() > 0 and yp.std() > 0
    pear = float(np.corrcoef(yt, yp)[0, 1]) if ok else float("nan")
    spear = float(np.corrcoef(_rank(yt), _rank(yp))[0, 1]) if ok else float("nan")
    return {"mae": mae, "mse": mse, "r2": r2, "pearson": pear, "spearman": spear}


def main():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import (
        RandomForestRegressor, ExtraTreesRegressor,
        GradientBoostingRegressor, HistGradientBoostingRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

    feat_cols, data = load_table()
    Xtr, ytr = make_xy(load_ids("train"), data)
    Xte, yte = make_xy(load_ids("test"), data)
    print(f"[data] features={len(feat_cols)} train={len(ytr)} test={len(yte)} "
          f"target_std(test)={yte.std():.3f}", flush=True)

    # Clean inf / wild magnitudes (BCUT, Ipc, etc. overflow float32) BEFORE
    # imputation, and drop columns that are all-NaN on train.
    def sanitize(X):
        X = np.where(np.isfinite(X), X, np.nan)
        return np.clip(X, -1e8, 1e8)
    Xtr, Xte = sanitize(Xtr), sanitize(Xte)
    keep = ~np.all(np.isnan(Xtr), axis=0)
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    print(f"[clean] kept {int(keep.sum())}/{len(feat_cols)} features", flush=True)

    # Deterministic preprocessing shared by all models.
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr_i, Xte_i = imp.transform(Xtr), imp.transform(Xte)

    def deterministic(name, est):
        m = est.fit(Xtr_i, ytr)
        return name, metrics(yte, m.predict(Xte_i))

    def seeded(name, ctor, seeds=(0, 1, 2)):
        maes, mets = [], []
        for s in seeds:
            m = ctor(s).fit(Xtr_i, ytr)
            d = metrics(yte, m.predict(Xte_i))
            maes.append(d["mae"]); mets.append(d)
        mae = np.array(maes)
        return name, {"mae": float(mae.mean()), "mae_std": float(mae.std(ddof=1)),
                      "r2": float(np.mean([m["r2"] for m in mets])),
                      "pearson": float(np.mean([m["pearson"] for m in mets]))}

    results = {}
    # scaled linear / kernel / knn (deterministic)
    for name, est in [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("krr_rbf", make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3))),
        ("svr_rbf", make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1))),
        ("knn", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance"))),
    ]:
        try:
            results.update(dict([deterministic(name, est)]))
            print(f"[{name}] {results[name]}", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)

    # GP (deterministic enough; can be slow — wrap in try)
    try:
        kern = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(0.5)
        gp = make_pipeline(StandardScaler(),
                           GaussianProcessRegressor(kernel=kern, alpha=1e-6, normalize_y=True))
        results.update(dict([deterministic("gpr", gp)]))
        print(f"[gpr] {results['gpr']}", flush=True)
    except Exception as e:
        print(f"[gpr] FAILED: {e}", flush=True)

    # tree ensembles (seeded -> mean/std), CPU-parallel but capped
    NJ = 48  # well under half of 384 cores
    for name, ctor in [
        ("rf", lambda s: RandomForestRegressor(n_estimators=600, n_jobs=NJ, random_state=s)),
        ("extratrees", lambda s: ExtraTreesRegressor(n_estimators=600, n_jobs=NJ, random_state=s)),
        ("gbr", lambda s: GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                                    learning_rate=0.03, subsample=0.8, random_state=s)),
        ("histgb", lambda s: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                          l2_regularization=1.0, random_state=s)),
    ]:
        try:
            results.update(dict([seeded(name, ctor)]))
            print(f"[{name}] {results[name]}", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)

    best = min(results.items(), key=lambda kv: kv[1]["mae"])
    print("\n=== SUMMARY (OD_Murcko test) — target MAE to beat = 0.5330 ===")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["mae"]):
        std = f" ± {v['mae_std']:.3f}" if "mae_std" in v else ""
        print(f"  {k:12s} MAE={v['mae']:.4f}{std}  R2={v['r2']:+.3f}  r={v['pearson']:.3f}")
    print(f"\nBEST: {best[0]}  MAE={best[1]['mae']:.4f}")
    json.dump(results, open("/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet/runs/_logs/classic_ml_screen.json", "w"), indent=2)


if __name__ == "__main__":
    main()
