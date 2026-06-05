"""Out-of-fold GBR predictions per peptide, for the deep GBR-residual / stacking.

GBR (descriptor-based gradient boosting) is the strong OOD reference from
Round 0. To let the deep model fit the GBR *residual* without leakage, we need
GBR predictions for TRAIN molecules that the GBR did NOT see — hence 5-fold
out-of-fold (OOF) preds on train. For val/test we fit GBR on ALL train.

Output: runs/_logs/gbr_preds_<scheme>.json  ->  {pid: gbr_pred, ...} for every
train/val/test molecule of the scheme. Also prints the GBR OD test metrics.
"""
from __future__ import annotations
import csv, json
import numpy as np

import sys
ROOT = "/hdd0/sohyun/cyclic-peptide-permeability"
CSV = f"{ROOT}/data/CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv"
SPLITS = f"{ROOT}/splits"
SCHEME = sys.argv[1] if len(sys.argv) > 1 else "OD_Murcko"   # ID | Cliff_ratio | OD_Murcko | ...
TARGET = "PAMPA"
OUT = f"{ROOT}/ChameleonNet/runs/_logs/gbr_preds_{SCHEME}.json"

LEAK = {
    "CycPeptMPDB_ID", "Source", "Original_Name_in_Source_Literature",
    "Structurally_Unique_ID", "SMILES", "HELM", "Sequence", "HELM_URL",
    "Year", "Version", "Molecule_Shape",
    "PAMPA", "PAMPA-4D", "Permeability", "Caco2", "MDCK", "RRCK",
    "R_PAMAP", "R_Caco2", "R_MDCK", "R_RRCK", "T_PAMPA",
    "Detection_Limit_1", "Detection_Limit_2",
}


def load_ids(fold):
    with open(f"{SPLITS}/{SCHEME}_{fold}_ids.csv") as f:
        return [int(r["CycPeptMPDB_ID"]) for r in csv.DictReader(f)]


def load_table():
    rows = list(csv.DictReader(open(CSV, newline="")))
    feat_cols = [c for c in rows[0].keys() if c not in LEAK]

    def fnum(s):
        try: return float(s)
        except (TypeError, ValueError): return np.nan

    data = {}
    for r in rows:
        try: pid = int(r["CycPeptMPDB_ID"])
        except (KeyError, ValueError): continue
        y = fnum(r.get(TARGET, ""))
        if not np.isfinite(y): continue
        x = np.array([fnum(r.get(c, "")) for c in feat_cols], dtype=np.float64)
        data[pid] = (x, y)
    return data


def metrics(yt, yp):
    err = yp - yt
    return dict(mae=float(np.mean(np.abs(err))),
                mse=float(np.mean(err**2)),
                r2=float(1 - np.mean(err**2)/np.var(yt)),
                pearson=float(np.corrcoef(yt, yp)[0,1]))


def main():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold

    data = load_table()
    tr = [p for p in load_ids("train") if p in data]
    va = [p for p in load_ids("val") if p in data]
    te = [p for p in load_ids("test") if p in data]

    def XY(ids):
        X = np.stack([data[p][0] for p in ids]); y = np.array([data[p][1] for p in ids])
        return np.clip(np.where(np.isfinite(X), X, np.nan), -1e8, 1e8), y

    Xtr, ytr = XY(tr); Xva, yva = XY(va); Xte, yte = XY(te)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xva, Xte = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)

    def gbr(seed=0):
        return GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                         learning_rate=0.03, subsample=0.8, random_state=seed)

    preds = {}
    # OOF predictions for train
    oof = np.zeros(len(tr))
    for fold, (itr, ite) in enumerate(KFold(5, shuffle=True, random_state=0).split(Xtr)):
        m = gbr().fit(Xtr[itr], ytr[itr])
        oof[ite] = m.predict(Xtr[ite])
    for p, v in zip(tr, oof): preds[int(p)] = float(v)
    print(f"[oof-train] MAE={metrics(ytr, oof)['mae']:.4f}", flush=True)

    # val/test from GBR fit on all train
    full = gbr().fit(Xtr, ytr)
    for p, v in zip(va, full.predict(Xva)): preds[int(p)] = float(v)
    pte = full.predict(Xte)
    for p, v in zip(te, pte): preds[int(p)] = float(v)
    print(f"[gbr OD_Murcko TEST] {metrics(yte, pte)}", flush=True)

    json.dump(preds, open(OUT, "w"))
    print(f"[saved] {OUT}  ({len(preds)} peptides)")


if __name__ == "__main__":
    main()
