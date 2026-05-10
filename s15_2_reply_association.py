#!/usr/bin/env python3
"""
s15_2_reply_association.py — Association between mentor reply features and
newcomer retention, within the treated group, controlling for pre-treatment
covariates.

Design:
  - Sample: treated newcomers only (N ≈ 26,270)
  - Outcome: primary (1+ mainspace edit 14d) and secondary DVs
  - Reply features: standardized (z-scored) so coefficients are comparable
  - Controls: all 164 pre-treatment covariates from the PS model
  - Model: logistic regression (primary DV) and OLS (continuous DVs)
  - Also runs by subgroup to see if associations differ

This is ASSOCIATIONAL, not causal. Reply features are post-treatment.

Input:
  data/s12/psm_data/psm_dataset.npz  — covariates + outcomes + treatment
  data/s15/reply_features.csv        — reply features from s15
  data/s11/s11_features.jsonl        — conversation_id order

Output:
  data/s15/association_results.csv
  data/s15/association_by_subgroup.csv
  data/s15/figures/association_forest.pdf
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
PSM_FILE = BASE / "data" / "s12" / "psm_data" / "psm_dataset.npz"
S11_FILE = BASE / "data" / "s11" / "s11_features.jsonl"
S15_FILE = BASE / "data" / "s15" / "reply_features.csv"
OUT_DIR = BASE / "data" / "s15"
OUT_FIG = OUT_DIR / "figures"

SEP = "=" * 70

# Reply features to analyze (interpretable, non-redundant)
REPLY_FEATURES = [
    ("r_words",              "Reply word count"),
    ("r_pronoun_you_rate",   "\"You\" rate (per word)"),
    ("r_pronoun_i_rate",     "\"I\" rate (per word)"),
    ("r_pronoun_we_rate",    "\"We\" rate (per word)"),
    ("r_vader_compound",     "VADER compound sentiment"),
    ("r_tb_subjectivity",    "TextBlob subjectivity"),
    ("r_flesch_kincaid",     "Flesch-Kincaid grade"),
    ("r_n_imperatives",      "Imperative sentences"),
    ("r_n_resources",        "Resource links (total)"),
    ("r_n_wikilink",         "Wikilinks"),
    ("r_n_policy",           "Policy references"),
    ("r_n_help",             "Help page references"),
    ("r_reply_lag_hours",    "Reply lag (hours)"),
    ("r_reply_q_word_ratio", "Reply/question word ratio"),
    ("r_has_greeting",       "Has greeting"),
    ("r_polite_gratitude",   "Gratitude expression"),
    ("r_polite_has_positive","Positive language"),
    ("r_polite_deference",   "Deference"),
    ("r_has_list",           "Has list"),
    ("r_n_question_marks",   "Question marks in reply"),
]

DVS = [
    ("primary",               "1+ mainspace edit (14d)"),
    ("n_mainspace_edits_14d", "N mainspace edits (14d)"),
    ("active_days_30d",       "Active days (30d)"),
    ("unique_ns",             "Unique namespaces (14d)"),
    ("reverted_any",          "Reverted any (14d)"),
]


def remove_collinear(X, threshold=0.999):
    """Remove near-constant and perfectly collinear columns."""
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.std() < 1e-10:
            continue
        keep.append(j)
    X = X[:, keep]
    # QR to find rank
    Q, R = np.linalg.qr(X, mode="reduced")
    diag = np.abs(np.diag(R))
    mask = diag > diag.max() * 1e-10
    return X[:, mask], np.array(keep)[mask]


def ols_with_controls(y, X_reply, X_controls):
    """
    OLS: y ~ X_reply + X_controls.
    Uses QR decomposition to handle collinearity.
    Returns coefficients, SEs, t-stats, p-values for reply features only.
    """
    k_reply = X_reply.shape[1]

    # Remove collinear columns from controls only
    X_c_clean, _ = remove_collinear(X_controls)

    X = np.hstack([np.ones((X_reply.shape[0], 1)), X_reply, X_c_clean])
    n, p = X.shape

    try:
        # QR-based OLS
        Q, R = np.linalg.qr(X, mode="reduced")
        beta = np.linalg.solve(R, Q.T @ y)
        resid = y - X @ beta
        sigma2 = (resid ** 2).sum() / max(n - p, 1)
        R_inv = np.linalg.inv(R)
        var_beta = sigma2 * (R_inv @ R_inv.T).diagonal()
        se = np.sqrt(np.maximum(var_beta, 0))
    except np.linalg.LinAlgError:
        return (np.full(k_reply, np.nan),) * 4

    # reply features are columns 1..k_reply (after intercept)
    beta_r = beta[1:k_reply+1]
    se_r = se[1:k_reply+1]
    t_r = beta_r / np.where(se_r > 0, se_r, np.nan)
    p_r = 2 * (1 - stats.t.cdf(np.abs(t_r), df=max(n - p, 1)))

    return beta_r, se_r, t_r, p_r


def main():
    t0 = time.time()
    print(f"\n{SEP}")
    print("  s15b: REPLY FEATURE → RETENTION ASSOCIATION ANALYSIS")
    print(SEP)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading PSM dataset...")
    psm = np.load(PSM_FILE, allow_pickle=True)
    y_treat = psm["y_treat"]
    X_E = psm["X_E"]
    X_Qtext = psm["X_Qtext"]
    X_Qpersp = psm["X_Qpersp"]
    X_Qtype = psm["X_Qtype"]
    X_emb20 = psm["X_emb20"]
    X_temporal = psm["X_temporal"]
    X_all = np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_emb20, X_temporal])

    OC = {}
    for k in ["primary", "n_mainspace_edits_14d", "active_days_30d", "unique_ns", "reverted_any"]:
        OC[k] = psm[f"oc_{k}"]

    # Get conversation_id order from s11
    s11_cids = []
    with open(S11_FILE) as f:
        for line in f:
            s11_cids.append(json.loads(line)["conversation_id"])

    N = len(y_treat)
    treated_mask = y_treat == 1
    treated_indices = np.where(treated_mask)[0]
    treated_cid_map = {s11_cids[i]: i for i in treated_indices}
    print(f"  N = {N:,}, Treated = {treated_mask.sum():,}")

    # Load reply features
    print("  Loading reply features...")
    rf = pd.read_csv(S15_FILE)
    print(f"  Reply features: {len(rf):,} rows, {len(rf.columns)} columns")

    # Align: map reply features to PSM row indices
    rf_cols = [c for c, _ in REPLY_FEATURES if c in rf.columns]
    rf_names = [n for c, n in REPLY_FEATURES if c in rf.columns]
    print(f"  Using {len(rf_cols)} reply features: {rf_cols}")

    # Build aligned arrays
    valid_psm_idx = []
    valid_rf_idx = []
    for ri, cid in enumerate(rf["conversation_id"]):
        if cid in treated_cid_map:
            valid_psm_idx.append(treated_cid_map[cid])
            valid_rf_idx.append(ri)

    valid_psm_idx = np.array(valid_psm_idx)
    valid_rf_idx = np.array(valid_rf_idx)
    print(f"  Matched: {len(valid_psm_idx):,} treated conversations")

    # Extract aligned matrices
    X_controls = X_all[valid_psm_idx]
    X_controls_s = StandardScaler().fit_transform(X_controls)

    X_reply_raw = rf.iloc[valid_rf_idx][rf_cols].values.astype(float)
    # Handle NaN/inf
    X_reply_raw = np.nan_to_num(X_reply_raw, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip reply lag outliers (cap at 99th percentile)
    for j, col in enumerate(rf_cols):
        if col == "r_reply_lag_hours":
            p99 = np.percentile(X_reply_raw[:, j], 99)
            X_reply_raw[:, j] = np.clip(X_reply_raw[:, j], 0, p99)

    reply_scaler = StandardScaler()
    X_reply_s = reply_scaler.fit_transform(X_reply_raw)

    # Question type labels for subgroup analysis
    q_types = rf.iloc[valid_rf_idx][["Q0", "Q2", "Q3", "Q4", "Q5"]].values

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: FULL TREATED SAMPLE — All DVs
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PART 1: FULL TREATED SAMPLE — Reply Feature Associations")
    print(SEP)

    all_results = []
    for dv_key, dv_name in DVS:
        y = OC[dv_key][valid_psm_idx]

        beta, se, t, p = ols_with_controls(y, X_reply_s, X_controls_s)

        print(f"\n  DV: {dv_name} (N = {len(y):,}, mean = {y.mean():.4f})")
        print(f"  {'Feature':<30s} {'β (std)':>10s} {'SE':>8s} {'t':>8s} {'p':>8s} {'Sig':>5s}")
        print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

        for j, (col, name) in enumerate(zip(rf_cols, rf_names)):
            sig = ""
            if p[j] < 0.001:
                sig = "***"
            elif p[j] < 0.01:
                sig = "**"
            elif p[j] < 0.05:
                sig = "*"

            print(f"  {name:<30s} {beta[j]:>+10.4f} {se[j]:>8.4f} {t[j]:>8.2f} {p[j]:>8.4f} {sig:>5s}")

            all_results.append({
                "DV": dv_name, "DV_key": dv_key,
                "Feature": name, "Feature_col": col,
                "Beta_std": round(beta[j], 4),
                "SE": round(se[j], 4),
                "t": round(t[j], 2),
                "p": round(p[j], 4),
                "Sig": sig,
                "Subgroup": "Full treated",
            })

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUT_DIR / "association_results.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'association_results.csv'}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: BY SUBGROUP — Primary DV only
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PART 2: BY SUBGROUP — Primary DV Association")
    print(SEP)

    q_sub = q_types[:, 0] == "Y"
    q_ref = q_types[:, 1] == "Y"
    q_apr = q_types[:, 2] == "Y"
    q_nor = q_types[:, 3] == "Y"
    q_own = q_types[:, 4] == "Y"

    subgroups = [
        ("Technical",       q_sub & ~q_ref & ~q_apr & ~q_nor),
        ("Referent (Q2)",   q_ref),
        ("Normative (Q4)",  q_nor),
        ("Non-substantive", q_types[:, 0] == "N"),
        ("Own work (Q5=Y)", q_own),
        ("No own work (Q5=N)", ~q_own),
    ]

    sg_results = []
    for sg_name, sg_mask in subgroups:
        n_sg = sg_mask.sum()
        if n_sg < 200:
            print(f"\n  SKIP {sg_name}: N={n_sg}")
            continue

        X_r_sg = X_reply_s[sg_mask]
        X_c_sg = X_controls_s[sg_mask]

        print(f"\n  {'='*60}")
        print(f"  {sg_name} (N={n_sg:,})")
        print(f"  {'='*60}")

        for dv_key, dv_name in DVS:
            y_sg = OC[dv_key][valid_psm_idx[sg_mask]]
            beta, se, t_val, p = ols_with_controls(y_sg, X_r_sg, X_c_sg)

            # Collect all results
            feat_results = []
            for j, (col, name) in enumerate(zip(rf_cols, rf_names)):
                sig = ""
                if p[j] < 0.001: sig = "***"
                elif p[j] < 0.01: sig = "**"
                elif p[j] < 0.05: sig = "*"
                feat_results.append((name, col, beta[j], se[j], t_val[j], p[j], sig))
                sg_results.append({
                    "Subgroup": sg_name, "N": n_sg, "DV": dv_name, "DV_key": dv_key,
                    "Feature": name, "Feature_col": col,
                    "Beta_std": round(beta[j], 4), "SE": round(se[j], 4),
                    "t": round(t_val[j], 2), "p": round(p[j], 4), "Sig": sig,
                })

            # Print: show significant first, sorted by |β|
            sig_feats = [(n, b, pv, s) for n, _, b, _, _, pv, s in feat_results if s]
            sig_feats.sort(key=lambda x: abs(x[1]), reverse=True)

            if sig_feats:
                print(f"\n    {dv_name} (mean={y_sg.mean():.4f})")
                print(f"    {'Feature':<30s} {'β (std)':>10s} {'p':>8s} {'Sig':>5s}")
                print(f"    {'-'*30} {'-'*10} {'-'*8} {'-'*5}")
                for name, b, pv, s in sig_feats:
                    print(f"    {name:<30s} {b:>+10.4f} {pv:>8.4f} {s:>5s}")

    sg_df = pd.DataFrame(sg_results)
    sg_df.to_csv(OUT_DIR / "association_by_subgroup.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'association_by_subgroup.csv'}")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  FIGURES")
    print(SEP)

    # Figure 1: Forest plot — primary DV, full sample
    primary_res = res_df[res_df["DV_key"] == "primary"].copy()
    primary_res = primary_res.sort_values("Beta_std").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, len(primary_res) * 0.45 + 2))
    for i, (_, r) in enumerate(primary_res.iterrows()):
        ci_lo = r["Beta_std"] - 1.96 * r["SE"]
        ci_hi = r["Beta_std"] + 1.96 * r["SE"]
        color = "#1f77b4" if r["Sig"] else "#bbbbbb"
        ax.plot([ci_lo, ci_hi], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.plot(r["Beta_std"], i, "o", color=color, markersize=7, zorder=5)
        label = f"β={r['Beta_std']:+.4f} {r['Sig']}"
        ax.annotate(label, xy=(ci_hi + 0.002, i), fontsize=7, va="center")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(primary_res)))
    ax.set_yticklabels(primary_res["Feature"], fontsize=8)
    ax.set_xlabel("Standardized β (1 SD increase in reply feature → Δ retention probability)")
    ax.set_title("Reply Feature → Retention Association (Treated Group, N={:,})\n"
                 "Controlling for 164 pre-treatment covariates · OLS"
                 .format(len(valid_psm_idx)),
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "association_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved association_forest.pdf")

    # Figure 2: Heatmap — primary DV by subgroup
    sg_primary = sg_df[sg_df["DV_key"] == "primary"].copy()
    sg_pivot = sg_primary.pivot(index="Feature", columns="Subgroup", values="Beta_std")
    sg_sig = sg_primary.pivot(index="Feature", columns="Subgroup", values="Sig")

    feat_order = [n for _, n in zip(rf_cols, rf_names)]
    feat_present = [f for f in feat_order if f in sg_pivot.index]
    sg_order = ["Technical", "Referent (Q2)", "Normative (Q4)",
                "Non-substantive", "Own work (Q5=Y)", "No own work (Q5=N)"]
    sg_present = [s for s in sg_order if s in sg_pivot.columns]

    sg_pivot = sg_pivot.loc[feat_present, sg_present]
    sg_sig = sg_sig.loc[feat_present, sg_present]

    fig, ax = plt.subplots(figsize=(12, len(feat_present) * 0.4 + 2))
    finite_vals = sg_pivot.values[np.isfinite(sg_pivot.values)]
    if len(finite_vals) == 0:
        print("  No finite values for heatmap, skipping.")
        return
    vmax = max(abs(finite_vals.min()), abs(finite_vals.max()))
    vmax = min(vmax, 0.05)
    im = ax.imshow(sg_pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(sg_present)))
    ax.set_xticklabels(sg_present, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(feat_present)))
    ax.set_yticklabels(feat_present, fontsize=8)
    for i in range(len(feat_present)):
        for j in range(len(sg_present)):
            val = sg_pivot.iloc[i, j]
            sig = sg_sig.iloc[i, j] if pd.notna(sg_sig.iloc[i, j]) else ""
            if np.isfinite(val):
                txt = f"{val:+.3f}{sig}"
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)
    plt.colorbar(im, ax=ax, label="Standardized β", shrink=0.8)
    ax.set_title("Reply Feature → Retention (Primary DV) by Subgroup\n"
                 "OLS controlling for 164 pre-treatment covariates",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "association_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved association_heatmap.pdf")

    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print(f"  s15b COMPLETE ({elapsed:.0f}s)")
    print(SEP)


if __name__ == "__main__":
    main()
