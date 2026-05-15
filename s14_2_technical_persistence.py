#!/usr/bin/env python3
"""
s14_2_technical_persistence.py — Windowed effect persistence analysis for Technical subgroup.

Following the evaluation design of Morgan & Halfaker (2018), partitions the
post-reply period into non-overlapping windows and estimates the ATT for
1+ and 5+ mainspace edit thresholds within each window.

Windows: 0-14d, 15-28d (3-4 weeks), 29-60d (1-2 months), 61-180d (2-6 months)

Output:
  data/s14/tables/technical_persistence.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = BASE / "data" / "s12" / "psm_data" / "psm_dataset.npz"
OUT_TBL = BASE / "data" / "s14" / "tables"
OUT_TBL.mkdir(parents=True, exist_ok=True)

N_BOOT = 500
K = 5
SEP = "=" * 70


def strat_att(y_d, y_out, st):
    us = np.unique(st); w_sum, att_sum = 0.0, 0.0
    for s in us:
        sm = st == s
        yt = y_out[sm & (y_d == 1)]; yc = y_out[sm & (y_d == 0)]
        if len(yt) < 2 or len(yc) < 2:
            continue
        w = len(yt)
        att_sum += w * (yt.mean() - yc.mean())
        w_sum += w
    return att_sum / w_sum if w_sum else np.nan


def cluster_boot_ci(y_d, y_out, st, mids, mid_to_arr, n_boot=N_BOOT):
    rng = np.random.default_rng(42)
    unique_m = np.unique(mids); n_cl = len(unique_m)
    atts = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_m, n_cl, replace=True)
        bi = np.concatenate([mid_to_arr[m] for m in sampled])
        a = strat_att(y_d[bi], y_out[bi], st[bi])
        if not np.isnan(a):
            atts.append(a)
    if len(atts) > 50:
        return np.percentile(atts, 2.5), np.percentile(atts, 97.5)
    return np.nan, np.nan


def main():
    print(f"\n{SEP}")
    print("  s14_2: TECHNICAL SUBGROUP — EFFECT PERSISTENCE (WINDOWED)")
    print(SEP)

    D = np.load(DATA, allow_pickle=True)
    y_treat = D["y_treat"]
    mentee_ids = D["mentee_ids"]
    X_Qtype = D["X_Qtype"]

    # Technical mask: substantive=1, referent=0, appraisal=0, normative=0
    tech = (X_Qtype[:, 0] == 1) & (X_Qtype[:, 1] == 0) & (X_Qtype[:, 2] == 0) & (X_Qtype[:, 3] == 0)

    # Covariates (exclude Qtype)
    X = np.hstack([D["X_E"], D["X_Qtext"], D["X_Qpersp"], D["X_emb20"], D["X_temporal"]])
    Xs = StandardScaler().fit_transform(X[tech])
    Ts = y_treat[tech]
    Ms = mentee_ids[tech]

    print(f"  Technical: N={tech.sum()}, T={int(Ts.sum())}, C={int((1 - Ts).sum())}")

    # Fit PS and stratify (matching s14_1 methodology)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ps = cross_val_predict(
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42),
        Xs, Ts, cv=cv, method="predict_proba"
    )[:, 1]
    auc = roc_auc_score(Ts, ps)
    strata = pd.qcut(ps, K, labels=False, duplicates="drop")
    print(f"  PS AUC: {auc:.4f}")

    # Build mentee-to-row index
    mid_to_arr = {}
    for i, m in enumerate(Ms):
        mid_to_arr.setdefault(m, []).append(i)
    for m in mid_to_arr:
        mid_to_arr[m] = np.array(mid_to_arr[m])

    # Outcomes
    # 0-14d: primary (1+ edit), n_mainspace_edits_14d (for 5+ threshold)
    Y_1plus_14d = D["oc_primary"][tech]
    n_edits_14d = D["oc_n_mainspace_edits_14d"][tech]
    Y_5plus_14d = (n_edits_14d >= 5).astype(float)

    # Teahouse windows
    th_windows = ["15_28d", "29_60d", "61_180d"]
    outcomes = {
        ("1+ edit", "0-14d"): Y_1plus_14d,
        ("5+ edits", "0-14d"): Y_5plus_14d,
    }
    for w in th_windows:
        outcomes[("1+ edit", w.replace("_", "-"))] = D[f"oc_th_1plus_{w}"][tech]
        outcomes[("5+ edits", w.replace("_", "-"))] = D[f"oc_th_5plus_{w}"][tech]

    # Compute ATT for each
    window_order = ["0-14d", "15-28d", "29-60d", "61-180d"]
    window_labels = ["0-14d", "3-4 weeks", "1-2 months", "2-6 months"]
    rows = []

    print(f"\n  {'Threshold':<10} {'Window':<12} {'Ctrl':>7} {'Treat':>7} {'ATT':>8} {'CI':>20} {'Sig':>4}")
    print("  " + "-" * 72)

    for threshold in ["1+ edit", "5+ edits"]:
        for w, wlabel in zip(window_order, window_labels):
            Y = outcomes[(threshold, w)]
            att = strat_att(Ts, Y, strata)
            lo, hi = cluster_boot_ci(Ts, Y, strata, Ms, mid_to_arr)
            ctrl = Y[Ts == 0].mean()
            treat = Y[Ts == 1].mean()
            sig = "*" if (lo > 0 or hi < 0) else ""

            rows.append({
                "Threshold": threshold,
                "Window": wlabel,
                "N": int(tech.sum()),
                "Ctrl_mean": round(ctrl, 4),
                "Treat_mean": round(treat, 4),
                "ATT": round(att, 4),
                "CI_lo": round(lo, 4),
                "CI_hi": round(hi, 4),
                "Sig": sig,
            })

            print(f"  {threshold:<10} {wlabel:<12} {ctrl * 100:>6.1f}% {treat * 100:>6.1f}% "
                  f"{att * 100:>+7.1f}pp [{lo * 100:>+5.1f}, {hi * 100:>+5.1f}] {sig:>4}")

    df = pd.DataFrame(rows)
    out_path = OUT_TBL / "technical_persistence.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(SEP)


if __name__ == "__main__":
    main()
