"""Final report for the OD_Murcko campaign.

Given the 3 final extended-desc seed runs, compute the three candidates:
  1. ChameleonNet-ExtDesc (deep)            -> per-seed metrics -> mean/std
  2. ChameleonNet-ExtDesc + GBR stacking    -> per-seed (alpha tuned on val) -> mean/std
  3. 3-seed deep ensemble, and ensemble+GBR -> single numbers (ensembles have no seed-std)
All vs baseline (table) and GBR reference. Metric of record = MAE (lower better).
"""
from __future__ import annotations
import sys, json
import numpy as np
from predict_eval import predict, GBR   # reuse loader/predictor

SEEDS = [42, 1, 2]
RUNS = {s: f"runs/campaign/final_extdesc_seed{s}" for s in SEEDS}
BASELINE = dict(mae=0.5330, mse=0.5221, r2=0.2954, pearson=0.5956, spearman=0.5574)


def _rank(a):
    a = np.asarray(a, float); o = a.argsort(kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(1, len(a)+1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(c)); np.add.at(s, inv, r); return (s/c)[inv]

def met(yt, yp):
    e = yp - yt
    ok = yt.std() > 0 and yp.std() > 0
    return dict(mae=float(np.mean(np.abs(e))), mse=float(np.mean(e**2)),
                r2=float(1 - np.mean(e**2)/np.var(yt)),
                pearson=float(np.corrcoef(yt, yp)[0,1]) if ok else float("nan"),
                spearman=float(np.corrcoef(_rank(yt), _rank(yp))[0,1]) if ok else float("nan"))

def ms(dlist):
    out = {}
    for k in ("mae","mse","r2","pearson","spearman"):
        v = np.array([d[k] for d in dlist]); out[k] = (float(v.mean()), float(v.std(ddof=1)))
    return out

def main():
    dev = "cuda"
    # per-seed test preds (+ val for alpha tuning) aligned by pid
    deep_test, blend_test = [], []
    test_pred_by_seed, ty_ref, tg_ref = {}, None, None
    for s in SEEDS:
        vp, vd, vy = predict(RUNS[s], "val", dev)
        tp, td, ty = predict(RUNS[s], "test", dev)
        vg = np.array([GBR[str(p)] for p in vp]); tg = np.array([GBR[str(p)] for p in tp])
        deep_test.append(met(ty, td))
        a = min(np.linspace(0,1,21), key=lambda a: met(vy, a*vd+(1-a)*vg)["mae"])
        blend_test.append(met(ty, a*td+(1-a)*tg))
        test_pred_by_seed[s] = (tp, td, ty, tg)
        print(f"[seed {s}] deep MAE={deep_test[-1]['mae']:.4f}  blend(a={a:.2f}) MAE={blend_test[-1]['mae']:.4f}", flush=True)

    # 3-seed ensemble (align by pid using seed 42 order)
    tp0, _, ty0, tg0 = test_pred_by_seed[SEEDS[0]]
    order = {p: i for i, p in enumerate(tp0)}
    ens = np.zeros(len(tp0))
    for s in SEEDS:
        tp, td, _, _ = test_pred_by_seed[s]
        idx = np.array([order[p] for p in tp]); ens[idx] += td
    ens /= len(SEEDS)
    ens_deep = met(ty0, ens)
    a_e = min(np.linspace(0,1,21), key=lambda a: a)  # placeholder; tune below on val ensemble
    # tune ensemble+GBR alpha on val ensemble
    val_ens = np.zeros(0)
    vp0, _, vy0 = predict(RUNS[SEEDS[0]], "val", dev)
    vorder = {p: i for i, p in enumerate(vp0)}; ve = np.zeros(len(vp0))
    for s in SEEDS:
        vp, vd, _ = predict(RUNS[s], "val", dev)
        ve[np.array([vorder[p] for p in vp])] += vd
    ve /= len(SEEDS); vg0 = np.array([GBR[str(p)] for p in vp0])
    a_e = min(np.linspace(0,1,21), key=lambda a: met(vy0, a*ve+(1-a)*vg0)["mae"])
    ens_blend = met(ty0, a_e*ens + (1-a_e)*tg0)
    gbr_only = met(ty0, tg0)

    D, B = ms(deep_test), ms(blend_test)
    def line(name, d):
        if isinstance(next(iter(d.values())), tuple):
            return (f"{name:42s} MAE {d['mae'][0]:.4f}±{d['mae'][1]:.4f}  R2 {d['r2'][0]:+.3f}  "
                    f"PCC {d['pearson'][0]:.3f}  SCC {d['spearman'][0]:.3f}")
        return (f"{name:42s} MAE {d['mae']:.4f}          R2 {d['r2']:+.3f}  "
                f"PCC {d['pearson']:.3f}  SCC {d['spearman']:.3f}")
    print("\n================ FINAL (OD_Murcko test) ================")
    print(line("1. ChameleonNet-ExtDesc (deep, 3 seeds)", D))
    print(line("2. ChameleonNet-ExtDesc + GBR stack (3 seeds)", B))
    print(line(f"3a. 3-seed deep ensemble", ens_deep))
    print(line(f"3b. ensemble + GBR (a={a_e:.2f})", ens_blend))
    print(line("-- GBR (descriptor reference)", gbr_only))
    print(line("-- baseline (paper table)", BASELINE))
    json.dump(dict(deep=D, blend=B, ens_deep=ens_deep, ens_blend=ens_blend,
                   gbr=gbr_only, baseline=BASELINE, alpha_ens=float(a_e)),
              open("runs/_logs/final_report.json","w"), indent=2)
    print("\n[saved] runs/_logs/final_report.json")

if __name__ == "__main__":
    main()
