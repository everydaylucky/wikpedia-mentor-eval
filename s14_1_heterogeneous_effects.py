#!/usr/bin/env python3
"""
s14_heterogeneous_effects.py — Within-subgroup PSM analysis by question type.

For each Morrison (1993) subgroup + Q5 split:
  1. Re-estimate propensity score WITHIN each subgroup
  2. Stratified ATT with cluster bootstrap CI (mentee-level)
  3. All 5 DVs

Sensitivity analyses per subgroup:
  S1. Strata count variation (K=3,5,7,10)
  S2. PS trimming ([0.05,0.95], [0.10,0.90])
  S3. Covariate balance (SMD table)
  S4. Control-group baseline comparison across subgroups

Output:
  data/s14/tables/*.csv
  data/s14/figures/*.pdf
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = BASE / "data" / "s12" / "psm_data" / "psm_dataset.npz"
OUT_FIG = BASE / "data" / "s14" / "figures"
OUT_TBL = BASE / "data" / "s14" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

N_BOOT = 500
K_DEFAULT = 5
rng = np.random.RandomState(42)
SEP = "=" * 70

# ══════════════════════════════════════════════════════════════════════════════
# Estimator functions
# ══════════════════════════════════════════════════════════════════════════════

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


def cluster_boot_ci(y_d, y_out, st, mids, mid_to_arr, alpha=0.05, n_boot=N_BOOT):
    unique_m = np.unique(mids); n_cl = len(unique_m)
    atts = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_m, n_cl, replace=True)
        bi = np.concatenate([mid_to_arr[m] for m in sampled])
        a = strat_att(y_d[bi], y_out[bi], st[bi])
        if not np.isnan(a):
            atts.append(a)
    if len(atts) > 50:
        return np.percentile(atts, 100 * alpha / 2), np.percentile(atts, 100 * (1 - alpha / 2))
    return np.nan, np.nan


def compute_smd_table(X, col_names, y_d, strata_arr):
    """Compute weighted SMD for each covariate, return DataFrame."""
    rows = []
    for j in range(X.shape[1]):
        vals = X[:, j]
        tv = vals[y_d == 1]; cv = vals[y_d == 0]
        pooled = np.sqrt((tv.var() + cv.var()) / 2)
        if pooled == 0:
            continue
        # Raw SMD
        raw_smd = (tv.mean() - cv.mean()) / pooled
        # Weighted (stratified) SMD
        w_diffs, w_total = 0.0, 0.0
        for s in np.unique(strata_arr):
            sm = strata_arr == s
            t_s = vals[sm & (y_d == 1)]; c_s = vals[sm & (y_d == 0)]
            if len(t_s) > 0 and len(c_s) > 0:
                w = sm.sum()
                w_diffs += w * (t_s.mean() - c_s.mean())
                w_total += w
        adj_smd = (w_diffs / w_total) / pooled if w_total > 0 else 0
        rows.append({
            "feature": col_names[j] if j < len(col_names) else f"X{j}",
            "raw_smd": raw_smd, "adj_smd": adj_smd,
            "abs_raw": abs(raw_smd), "abs_adj": abs(adj_smd),
        })
    return pd.DataFrame(rows).sort_values("abs_adj", ascending=False)


def fit_ps_and_stratify(X, y_d, K, cv):
    """Fit PS within a subgroup, return ps, strata, auc."""
    X_s = StandardScaler().fit_transform(X)
    ps = cross_val_predict(
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42),
        X_s, y_d, cv=cv, method="predict_proba"
    )[:, 1]
    auc = roc_auc_score(y_d, ps)
    strata = pd.qcut(ps, K, labels=False, duplicates="drop")
    return ps, strata, auc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print(f"\n{SEP}")
    print("  s14: WITHIN-SUBGROUP PSM — HETEROGENEOUS EFFECTS")
    print(SEP)

    # ── Load ──────────────────────────────────────────────────────────────
    D = np.load(DATA, allow_pickle=True)
    y_treat = D["y_treat"]
    mentee_ids = D["mentee_ids"]

    X_E = D["X_E"]; X_Qtext = D["X_Qtext"]; X_Qpersp = D["X_Qpersp"]
    X_Qtype = D["X_Qtype"]; X_emb20 = D["X_emb20"]; X_temporal = D["X_temporal"]

    E_cols = list(D["E_cols"])
    Qtext_cols = list(D["Qtext_cols"])
    Qpersp_cols = list(D["Qpersp_cols"])
    Qtype_cols = list(D["Qtype_cols"])

    OC = {}
    for k in ["primary", "n_mainspace_edits_14d", "active_days_30d", "unique_ns", "reverted_any", "cross_day_any_14d"]:
        OC[k] = D[f"oc_{k}"]

    N = len(y_treat)
    print(f"  N = {N:,}")

    # Feature matrix WITHOUT Qtype (since we condition on question type)
    X_sub = np.hstack([X_E, X_Qtext, X_Qpersp, X_emb20, X_temporal])
    sub_col_names = E_cols + Qtext_cols + Qpersp_cols + [f"emb{i}" for i in range(X_emb20.shape[1])] + [f"t{i}" for i in range(X_temporal.shape[1])]
    print(f"  Covariates per subgroup: {X_sub.shape[1]}")

    # Pre-compute mentee cluster index
    mid_to_arr = {}
    for i, m in enumerate(mentee_ids):
        if m not in mid_to_arr:
            mid_to_arr[m] = []
        mid_to_arr[m].append(i)
    for m in mid_to_arr:
        mid_to_arr[m] = np.array(mid_to_arr[m])

    # ── Define subgroups ──────────────────────────────────────────────────
    q_sub = X_Qtype[:, 0]
    q_ref = X_Qtype[:, 1]
    q_apr = X_Qtype[:, 2]
    q_nor = X_Qtype[:, 3]
    q_own = X_Qtype[:, 4]

    subgroups = [
        ("Full sample",     np.ones(N, dtype=bool)),
        ("Technical",       (q_sub == 1) & (q_ref == 0) & (q_apr == 0) & (q_nor == 0)),
        ("Referent (Q2)",   q_ref == 1),
        ("Appraisal (Q3)",  q_apr == 1),
        ("Normative (Q4)",  q_nor == 1),
        ("Non-substantive", q_sub == 0),
        ("Own work (Q5=Y)", q_own == 1),
        ("No own work (Q5=N)", q_own == 0),
    ]

    DIMS = [
        ("1+_mainspace_14d",    "primary"),
        ("n_edits_14d",         "n_mainspace_edits_14d"),
        ("active_days_30d",     "active_days_30d"),
        ("unique_ns_14d",       "unique_ns"),
        ("reverted_any_14d",    "reverted_any"),
        ("2+_active_days_14d",  "cross_day_any_14d"),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Subgroup overview ─────────────────────────────────────────────────
    print(f"\n  {'Subgroup':<22s} {'N':>7s} {'T':>7s} {'C':>7s} {'T%':>6s} {'Ctrl base':>10s}")
    print("  " + "-" * 65)
    for name, mask in subgroups:
        n = mask.sum(); nt = int(y_treat[mask].sum()); nc = n - nt
        ctrl_base = OC["primary"][mask & (y_treat == 0)].mean()
        print(f"  {name:<22s} {n:>7,} {nt:>7,} {nc:>7,} {nt/n*100:>5.1f}% {ctrl_base:>10.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: MAIN RESULTS — Within-subgroup PSM (K=5)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PART 1: WITHIN-SUBGROUP PSM (K={})".format(K_DEFAULT))
    print(f"{'='*70}")

    all_results = []
    all_diagnostics = []

    for sg_name, sg_mask in subgroups:
        idx = np.where(sg_mask)[0]
        y_sg = y_treat[idx]
        mid_sg = mentee_ids[idx]
        X_sg = X_sub[idx]
        n_sg = len(idx)
        n_t = int(y_sg.sum()); n_c = n_sg - n_t

        if n_c < 50 or n_t < 50:
            print(f"\n  SKIP {sg_name}: T={n_t}, C={n_c}")
            continue

        print(f"\n{SEP}")
        print(f"  {sg_name}: N={n_sg:,} (T={n_t:,}, C={n_c:,})")
        print(SEP)

        K_use = min(K_DEFAULT, max(3, n_c // 30))
        ps_sg, strata_sg, auc_sg = fit_ps_and_stratify(X_sg, y_sg, K_use, cv)
        actual_K = len(np.unique(strata_sg))

        # Build local mid_to_arr for this subgroup
        local_mid_arr = {}
        for li, gi in enumerate(idx):
            m = mentee_ids[gi]
            if m not in local_mid_arr:
                local_mid_arr[m] = []
            local_mid_arr[m].append(li)
        for m in local_mid_arr:
            local_mid_arr[m] = np.array(local_mid_arr[m])

        # SMD count
        smd_df = compute_smd_table(X_sg, sub_col_names, y_sg, strata_sg)
        n_bad = (smd_df["abs_adj"] > 0.1).sum()

        print(f"  PS AUC={auc_sg:.4f}  K={actual_K}  SMD>0.1: {n_bad}/{len(smd_df)}")

        diag = {
            "Subgroup": sg_name, "N": n_sg, "Treated": n_t, "Control": n_c,
            "T_pct": round(n_t / n_sg * 100, 1), "AUC": round(auc_sg, 4),
            "K": actual_K, "SMD_bad": n_bad,
        }
        all_diagnostics.append(diag)

        # ATT for each DV
        for dim_name, dim_key in DIMS:
            y_out = OC[dim_key][idx]
            att = strat_att(y_sg, y_out, strata_sg)
            ci_lo, ci_hi = cluster_boot_ci(y_sg, y_out, strata_sg, mid_sg, local_mid_arr)
            sig = "*" if not np.isnan(ci_lo) and (ci_lo > 0 or ci_hi < 0) else ""

            ctrl_mean = y_out[y_sg == 0].mean()
            treat_mean = y_out[y_sg == 1].mean()
            sd_ctrl = y_out[y_sg == 0].std()
            cohen_d = att / sd_ctrl if sd_ctrl > 0 else 0.0

            all_results.append({
                "Subgroup": sg_name, "N": n_sg, "T": n_t, "C": n_c,
                "Outcome": dim_name,
                "Ctrl_mean": round(ctrl_mean, 4),
                "Treat_mean": round(treat_mean, 4),
                "ATT": round(att, 4), "CI_lo": round(ci_lo, 4), "CI_hi": round(ci_hi, 4),
                "Cohen_d": round(cohen_d, 4), "Sig": sig,
            })
            print(f"    {dim_name:<22s}  ATT={att:+.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}] {sig}"
                  f"  (ctrl={ctrl_mean:.3f})")

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUT_TBL / "subgroup_att.csv", index=False)

    diag_df = pd.DataFrame(all_diagnostics)
    diag_df.to_csv(OUT_TBL / "subgroup_diagnostics.csv", index=False)
    print(f"\n  Saved: subgroup_att.csv, subgroup_diagnostics.csv")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: SENSITIVITY — Strata count variation (primary + 2+ active days)
    # ══════════════════════════════════════════════════════════════════════
    SENS_DVS = [
        ("primary", "primary"),
        ("n_mainspace_edits_14d", "n_edits_14d"),
        ("active_days_30d", "active_days_30d"),
        ("unique_ns", "unique_ns_14d"),
        ("reverted_any", "reverted_any_14d"),
        ("cross_day_any_14d", "2+_active_days_14d"),
    ]
    print(f"\n{'='*70}")
    print("  PART 2: SENSITIVITY — Strata count (K=3,5,7,10)")
    print(f"{'='*70}")

    K_VALS = [3, 5, 7, 10]
    strata_rows = []

    for sg_name, sg_mask in subgroups:
        idx = np.where(sg_mask)[0]
        y_sg = y_treat[idx]; mid_sg = mentee_ids[idx]; X_sg = X_sub[idx]
        n_c = (y_sg == 0).sum()
        if n_c < 50 or y_sg.sum() < 50:
            continue

        # Build local mid_to_arr
        local_mid_arr = {}
        for li, gi in enumerate(idx):
            m = mentee_ids[gi]
            if m not in local_mid_arr:
                local_mid_arr[m] = []
            local_mid_arr[m].append(li)
        for m in local_mid_arr:
            local_mid_arr[m] = np.array(local_mid_arr[m])

        for dv_key, dv_label in SENS_DVS:
            print(f"\n  {sg_name} [{dv_label}]:")
            y_out = OC[dv_key][idx]

            for K in K_VALS:
                if n_c < K * 15:
                    continue
                ps_sg, strata_sg, auc = fit_ps_and_stratify(X_sg, y_sg, K, cv)
                att = strat_att(y_sg, y_out, strata_sg)
                ci_lo, ci_hi = cluster_boot_ci(y_sg, y_out, strata_sg, mid_sg, local_mid_arr, n_boot=200)
                sig = "*" if not np.isnan(ci_lo) and (ci_lo > 0 or ci_hi < 0) else ""

                strata_rows.append({
                    "Subgroup": sg_name, "DV": dv_label, "K": K, "ATT": round(att, 4),
                    "CI_lo": round(ci_lo, 4), "CI_hi": round(ci_hi, 4), "Sig": sig,
                })
                print(f"    K={K:>2d}  ATT={att:+.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}] {sig}")

    strata_df = pd.DataFrame(strata_rows)
    strata_df.to_csv(OUT_TBL / "sensitivity_strata.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: SENSITIVITY — PS trimming (primary + 2+ active days, K=5)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PART 3: SENSITIVITY — PS trimming")
    print(f"{'='*70}")

    TRIMS = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]
    trim_rows = []

    for sg_name, sg_mask in subgroups:
        idx = np.where(sg_mask)[0]
        y_sg = y_treat[idx]; mid_sg = mentee_ids[idx]; X_sg = X_sub[idx]
        n_c = (y_sg == 0).sum()
        if n_c < 50 or y_sg.sum() < 50:
            continue

        K_use = min(K_DEFAULT, max(3, n_c // 30))
        ps_sg, _, _ = fit_ps_and_stratify(X_sg, y_sg, K_use, cv)

        for dv_key, dv_label in SENS_DVS:
            y_out = OC[dv_key][idx]
            print(f"\n  {sg_name} [{dv_label}]:")

            for lo, hi in TRIMS:
                keep = (ps_sg >= lo) & (ps_sg <= hi)
                n_keep = keep.sum()
                if keep.sum() < 100:
                    continue

                strata_t = pd.qcut(ps_sg[keep], K_use, labels=False, duplicates="drop")

                # Local mid_to_arr for trimmed
                local_mid_arr = {}
                local_idx = np.where(keep)[0]
                for li, orig_li in enumerate(local_idx):
                    m = mentee_ids[idx[orig_li]]
                    if m not in local_mid_arr:
                        local_mid_arr[m] = []
                    local_mid_arr[m].append(li)
                for m in local_mid_arr:
                    local_mid_arr[m] = np.array(local_mid_arr[m])

                att = strat_att(y_sg[keep], y_out[keep], strata_t)
                ci_lo, ci_hi = cluster_boot_ci(
                    y_sg[keep], y_out[keep], strata_t, mid_sg[keep], local_mid_arr, n_boot=200)
                sig = "*" if not np.isnan(ci_lo) and (ci_lo > 0 or ci_hi < 0) else ""

                trim_rows.append({
                    "Subgroup": sg_name, "DV": dv_label, "Trim": f"[{lo},{hi}]", "N": n_keep,
                    "ATT": round(att, 4), "CI_lo": round(ci_lo, 4), "CI_hi": round(ci_hi, 4),
                    "Sig": sig,
                })
                print(f"    [{lo:.2f},{hi:.2f}] N={n_keep:,}  ATT={att:+.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}] {sig}")

    trim_df = pd.DataFrame(trim_rows)
    trim_df.to_csv(OUT_TBL / "sensitivity_trimming.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # PART 4: SENSITIVITY — Covariate balance detail (top imbalanced)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PART 4: COVARIATE BALANCE (top 10 largest SMD per subgroup)")
    print(f"{'='*70}")

    balance_rows = []
    for sg_name, sg_mask in subgroups:
        idx = np.where(sg_mask)[0]
        y_sg = y_treat[idx]; X_sg = X_sub[idx]
        n_c = (y_sg == 0).sum()
        if n_c < 50 or y_sg.sum() < 50:
            continue

        K_use = min(K_DEFAULT, max(3, n_c // 30))
        _, strata_sg, _ = fit_ps_and_stratify(X_sg, y_sg, K_use, cv)
        smd_df = compute_smd_table(X_sg, sub_col_names, y_sg, strata_sg)

        print(f"\n  {sg_name}:")
        print(f"    {'Feature':<40s} {'Raw SMD':>10s} {'Adj SMD':>10s}")
        print(f"    {'-'*40} {'-'*10} {'-'*10}")
        for _, row in smd_df.head(10).iterrows():
            flag = "!" if row["abs_adj"] > 0.1 else ""
            print(f"    {row['feature']:<40s} {row['raw_smd']:>+10.4f} {row['adj_smd']:>+10.4f} {flag}")
            balance_rows.append({
                "Subgroup": sg_name, "Feature": row["feature"],
                "Raw_SMD": round(row["raw_smd"], 4), "Adj_SMD": round(row["adj_smd"], 4),
            })

    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(OUT_TBL / "covariate_balance.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # PART 5: CONTROL-GROUP BASELINE COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  PART 5: CONTROL-GROUP BASELINES ACROSS SUBGROUPS")
    print(f"{'='*70}")

    baseline_rows = []
    print(f"\n  {'Subgroup':<22s}", end="")
    for dim_name, _ in DIMS:
        print(f" {dim_name:>16s}", end="")
    print()
    print("  " + "-" * (22 + 16 * len(DIMS)))

    for sg_name, sg_mask in subgroups:
        ctrl = sg_mask & (y_treat == 0)
        if ctrl.sum() < 10:
            continue
        row_data = {"Subgroup": sg_name, "N_ctrl": int(ctrl.sum())}
        print(f"  {sg_name:<22s}", end="")
        for dim_name, dim_key in DIMS:
            val = OC[dim_key][ctrl].mean()
            row_data[dim_name] = round(val, 4)
            print(f" {val:>16.4f}", end="")
        print()
        baseline_rows.append(row_data)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(OUT_TBL / "control_baselines.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  FIGURES")
    print(f"{'='*70}")

    colors = {
        "Full sample": "#333333",
        "Technical": "#1f77b4", "Referent (Q2)": "#ff7f0e",
        "Appraisal (Q3)": "#2ca02c", "Normative (Q4)": "#d62728",
        "Non-substantive": "#9467bd",
        "Own work (Q5=Y)": "#8c564b", "No own work (Q5=N)": "#e377c2",
    }

    # ── Figure 1: Forest plot — primary DV ─────────────────────────────
    primary = res_df[res_df["Outcome"] == "1+_mainspace_14d"].copy()
    primary = primary[primary["Subgroup"] != "Full sample"]
    primary = primary.sort_values("ATT", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, len(primary) * 0.8 + 2))
    for i, (_, r) in enumerate(primary.iterrows()):
        c = colors.get(r["Subgroup"], "#333")
        if r["Sig"] != "*":
            c = "#bbbbbb"
        ax.plot([r["CI_lo"], r["CI_hi"]], [i, i], color=c, lw=2.5, solid_capstyle="round")
        ax.plot(r["ATT"], i, "o", color=c, markersize=9, zorder=5)
        ax.annotate(f"ATT={r['ATT']:+.4f} {r['Sig']}  (N={r['N']:,})",
                    xy=(r["CI_hi"] + 0.003, i), fontsize=8, va="center")
    ax.axvline(0, color="black", lw=0.8)
    # Add full-sample reference line
    fs = res_df[(res_df["Outcome"] == "1+_mainspace_14d") & (res_df["Subgroup"] == "Full sample")]
    if len(fs):
        ax.axvline(fs.iloc[0]["ATT"], color="gray", ls="--", alpha=0.5, label=f"Full sample ATT={fs.iloc[0]['ATT']:+.4f}")
        ax.legend(fontsize=8)
    ax.set_yticks(range(len(primary)))
    ax.set_yticklabels(primary["Subgroup"], fontsize=10)
    ax.set_xlabel("ATT (1+ mainspace edit within 14 days)")
    ax.set_title("Within-Subgroup PSM: Treatment Effect by Question Type\n"
                 "Each subgroup has its own PS model · 95% cluster bootstrap CI",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "forest_primary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved forest_primary.pdf")

    # ── Figure 2: Heatmap — all DVs × subgroups ──────────────────────
    sg_order = ["Technical", "Referent (Q2)", "Appraisal (Q3)", "Normative (Q4)",
                "Non-substantive", "Own work (Q5=Y)", "No own work (Q5=N)"]
    dv_order = [d[0] for d in DIMS]

    pivot_att = res_df[res_df["Subgroup"] != "Full sample"].pivot(
        index="Subgroup", columns="Outcome", values="ATT")
    pivot_sig = res_df[res_df["Subgroup"] != "Full sample"].pivot(
        index="Subgroup", columns="Outcome", values="Sig")

    sg_present = [s for s in sg_order if s in pivot_att.index]
    dv_present = [d for d in dv_order if d in pivot_att.columns]
    pivot_att = pivot_att.loc[sg_present, dv_present]
    pivot_sig = pivot_sig.loc[sg_present, dv_present]

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(pivot_att.values, cmap="RdBu_r", aspect="auto", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(dv_present)))
    ax.set_xticklabels([d.replace("_", "\n") for d in dv_present], fontsize=9)
    ax.set_yticks(range(len(sg_present)))
    ax.set_yticklabels(sg_present, fontsize=10)
    for i in range(len(sg_present)):
        for j in range(len(dv_present)):
            val = pivot_att.iloc[i, j]
            sig = pivot_sig.iloc[i, j]
            txt = f"{val:+.3f}{sig}"
            color = "white" if abs(val) > 0.03 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label="ATT", shrink=0.8)
    ax.set_title("Within-Subgroup ATT by Question Type × Outcome\n* = 95% CI excludes zero",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "heatmap_att.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved heatmap_att.pdf")

    # ── Figure 3: Bar chart — primary DV with error bars ──────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    sg_plot = [s for s in sg_order if s in primary["Subgroup"].values]
    x = np.arange(len(sg_plot))
    for i, sg in enumerate(sg_plot):
        row = primary[primary["Subgroup"] == sg]
        if row.empty:
            continue
        r = row.iloc[0]
        c = colors.get(sg, "#333")
        ax.bar(i, r["ATT"], 0.6, color=c, alpha=0.85)
        ax.errorbar(i, r["ATT"],
                    yerr=[[r["ATT"] - r["CI_lo"]], [r["CI_hi"] - r["ATT"]]],
                    fmt="none", color="black", capsize=5, lw=1.5)
        if r["Sig"] == "*":
            ax.text(i, r["CI_hi"] + 0.003, "*", ha="center", fontsize=14, fontweight="bold")
    if len(fs):
        ax.axhline(fs.iloc[0]["ATT"], color="gray", ls="--", alpha=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sg_plot, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("ATT (1+ mainspace edit 14d)")
    ax.set_title("Within-Subgroup Treatment Effect (Primary DV)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "bar_primary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved bar_primary.pdf")

    # ── Figure 4: Strata sensitivity — key subgroups ──────────────────
    key_sg = ["Full sample", "Referent (Q2)", "Own work (Q5=Y)", "No own work (Q5=N)", "Technical"]
    primary_strata = strata_df[strata_df["DV"] == "primary"]
    fig, axes = plt.subplots(1, len(key_sg), figsize=(4 * len(key_sg), 4), sharey=True)
    for ax_i, sg in enumerate(key_sg):
        ax = axes[ax_i]
        sg_rows = primary_strata[primary_strata["Subgroup"] == sg]
        if sg_rows.empty:
            ax.set_visible(False)
            continue
        ks = sg_rows["K"].values
        atts = sg_rows["ATT"].values
        lo = sg_rows["CI_lo"].values
        hi = sg_rows["CI_hi"].values
        ax.errorbar(ks, atts, yerr=[atts - lo, hi - atts], fmt="o-", capsize=4, color=colors.get(sg, "#333"))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("K (strata)")
        ax.set_title(sg, fontsize=9)
        if ax_i == 0:
            ax.set_ylabel("ATT (primary)")
    fig.suptitle("Strata Count Sensitivity (Primary DV)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sensitivity_strata.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved sensitivity_strata.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print(f"  s14 COMPLETE ({elapsed:.0f}s)")
    print(SEP)

    print("\n  Diagnostics:")
    print(diag_df.to_string(index=False))

    print(f"\n  Primary DV summary:")
    for _, r in res_df[res_df["Outcome"] == "1+_mainspace_14d"].iterrows():
        print(f"    {r['Subgroup']:<22s}  ATT={r['ATT']:+.4f}  [{r['CI_lo']:+.4f}, {r['CI_hi']:+.4f}] {r['Sig']}"
              f"  (ctrl={r['Ctrl_mean']:.3f})")

    print(f"\n  Tables: {OUT_TBL}")
    print(f"  Figures: {OUT_FIG}")
    print(SEP)


if __name__ == "__main__":
    main()
