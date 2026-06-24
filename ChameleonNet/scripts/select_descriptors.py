"""Select top descriptors via RF + XGBoost importance on the OD_Murcko TRAIN set.

Feature selection is done train-only (no test leakage) and on the OOD split we
care about. We combine RandomForest impurity importance, XGBoost gain, and RF
permutation importance into a consensus rank, then emit the top-N descriptor
names for the deep model to ingest.

Output: runs/_logs/selected_descriptors.json
"""
from __future__ import annotations
import csv, json, sys
import numpy as np

ROOT = "/hdd0/sohyun/cyclic-peptide-permeability"
CSV = f"{ROOT}/data/CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv"
SPLITS = f"{ROOT}/splits"
SCHEME = "OD_Murcko"; TARGET = "PAMPA"
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 20

LEAK = {
    "CycPeptMPDB_ID","Source","Original_Name_in_Source_Literature","Structurally_Unique_ID",
    "SMILES","HELM","Sequence","Sequence_LogP","Sequence_TPSA","HELM_URL","Year","Version","Molecule_Shape",
    "PAMPA","PAMPA-4D","Permeability","Caco2","MDCK","RRCK","R_PAMAP","R_Caco2","R_MDCK","R_RRCK","T_PAMPA",
    "Detection_Limit_1","Detection_Limit_2",
}

def load_ids(fold):
    return {int(r["CycPeptMPDB_ID"]) for r in csv.DictReader(open(f"{SPLITS}/{SCHEME}_{fold}_ids.csv"))}

def main():
    rows = list(csv.DictReader(open(CSV)))
    feat = [c for c in rows[0].keys() if c not in LEAK]
    def fnum(s):
        try: return float(s)
        except: return np.nan
    tr = load_ids("train")
    X=[]; y=[]
    for r in rows:
        try: pid=int(r["CycPeptMPDB_ID"])
        except: continue
        if pid not in tr: continue
        yy=fnum(r.get(TARGET,""))
        if not np.isfinite(yy): continue
        X.append([fnum(r.get(c,"")) for c in feat]); y.append(yy)
    X=np.clip(np.where(np.isfinite(X),X,np.nan),-1e8,1e8); y=np.array(y)
    from sklearn.impute import SimpleImputer
    keep=~np.all(np.isnan(X),axis=0)
    feat=[f for f,k in zip(feat,keep) if k]; X=X[:,keep]
    X=SimpleImputer(strategy="median").fit_transform(X)
    print(f"[select] {len(feat)} features, {len(y)} train molecules", flush=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    rf=RandomForestRegressor(n_estimators=600,n_jobs=48,random_state=0).fit(X,y)
    rf_imp=rf.feature_importances_
    perm=permutation_importance(rf,X,y,n_repeats=5,random_state=0,n_jobs=48).importances_mean
    import xgboost as xgb
    xr=xgb.XGBRegressor(n_estimators=600,max_depth=4,learning_rate=0.03,subsample=0.8,
                        colsample_bytree=0.8,random_state=0,n_jobs=48).fit(X,y)
    xg_imp=xr.feature_importances_

    def rank(a):  # higher importance -> rank 1
        order=np.argsort(-a); r=np.empty(len(a)); r[order]=np.arange(1,len(a)+1); return r
    consensus=(rank(rf_imp)+rank(xg_imp)+rank(perm))/3.0
    idx=np.argsort(consensus)[:TOPN]
    top=[feat[i] for i in idx]
    table=[{"feature":feat[i],"rf":float(rf_imp[i]),"xgb":float(xg_imp[i]),
            "perm":float(perm[i]),"consensus_rank":float(consensus[i])} for i in idx]
    print(f"\n=== Top {TOPN} consensus descriptors (RF+XGB+perm, OD_Murcko train) ===")
    for t in table: print(f"  {t['feature']:24s} rf={t['rf']:.3f} xgb={t['xgb']:.3f} perm={t['perm']:.4f}")
    json.dump({"scheme":SCHEME,"topn":TOPN,"features":top,"table":table},
              open(f"{ROOT}/ChameleonNet/runs/_logs/selected_descriptors.json","w"),indent=2)
    print(f"\n[saved] runs/_logs/selected_descriptors.json")
    print("PYLIST:", top)

if __name__=="__main__":
    main()
