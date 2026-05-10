#!/usr/bin/env python3
"""
s13_psm_analysis.py — Full PSM analysis pipeline.

Loads s12 psm_dataset.npz and runs:
  Part A: Diagnostics + Feature Ablation
    A1  PS estimation (5-fold CV Logistic Regression)
    A2  PS distribution (histogram + boxplot)
    A3  Stratification diagnostics table
    A4  Covariate balance Love plot (SMD before/after)
    A5  Common support / positivity
    A6  Feature importance (top 25 coefficients)
    A7  Feature ablation (AUC + ATT stability, with/without Qtype)

  Part B: Main Effects
    B1  5 outcome dimensions × {Naive, ATT, DR}
    B2  Cluster bootstrap CI (mentee-level, 500 resamples)
    B3  Cohen's d effect sizes
    B4  Forest plot

  Part C: Sensitivity & Robustness
    C1  Strata sensitivity (K=5/10/20)
    C2  D=48h treatment definition
    C3  PS trimming (0.01-0.99 … 0.15-0.85)
    C4  Window sensitivity (7/14/21/28/30/60/180d)
    C5  Rosenbaum bounds + E-value
    C6  Robustness summary table + forest plot

Output:
  data/s13/figures/*.pdf
  data/s13/tables/*.csv
  Console summary

Dependencies:
  pip install numpy pandas scikit-learn scipy matplotlib
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.linalg import lstsq as np_lstsq
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
BASE = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = BASE / "data" / "s12" / "psm_data" / "psm_dataset.npz"
OUT_FIG = BASE / "data" / "s13" / "figures"
OUT_TBL = BASE / "data" / "s13" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

N_STRATA = 10
N_BOOT = 500
rng = np.random.RandomState(42)

SEP = "=" * 70

# ══════════════════════════════════════════════════════════════════════════════
# Estimator functions
# ══════════════════════════════════════════════════════════════════════════════

def strat_att(y_d, y_out, st):
    us = np.unique(st); w_sum, att_sum = 0, 0
    for s in us:
        sm = st == s
        yt = y_out[sm & (y_d == 1)]; yc = y_out[sm & (y_d == 0)]
        if len(yt) < 2 or len(yc) < 2:
            continue
        w = len(yt)
        att_sum += w * (yt.mean() - yc.mean())
        w_sum += w
    return att_sum / w_sum if w_sum else np.nan


def strat_dr(y_d, y_out, st, X_cov):
    us = np.unique(st); w_sum, dr_sum = 0, 0
    for s in us:
        sm = st == s; y_s = y_out[sm]; d_s = y_d[sm].astype(float)
        if d_s.sum() < 2 or (1 - d_s).sum() < 2:
            continue
        X_s = np.column_stack([d_s, X_cov[sm], np.ones(sm.sum())])
        try:
            beta, _, _, _ = np_lstsq(X_s, y_s, rcond=None)
        except Exception:
            continue
        w = sm.sum()
        dr_sum += w * beta[0]
        w_sum += w
    return dr_sum / w_sum if w_sum else np.nan


def cluster_boot_ci(y_d, y_out, st, mids, alpha=0.05, X_cov=None, est="att"):
    unique_m = np.unique(mids); n_cl = len(unique_m)
    mid_to_idx = defaultdict(list)
    for i, m in enumerate(mids):
        mid_to_idx[m].append(i)
    atts = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique_m, n_cl, replace=True)
        bi = np.concatenate([mid_to_idx[m] for m in sampled])
        if est == "dr" and X_cov is not None:
            a = strat_dr(y_d[bi], y_out[bi], st[bi], X_cov[bi])
        else:
            a = strat_att(y_d[bi], y_out[bi], st[bi])
        if not np.isnan(a):
            atts.append(a)
    if len(atts) > 50:
        return np.percentile(atts, 100 * alpha / 2), np.percentile(atts, 100 * (1 - alpha / 2))
    return np.nan, np.nan


def rosenbaum_bounds(y_d, y_out, st_arr, gamma_values):
    results = []
    for gamma in gamma_values:
        T_sum, var_sum = 0, 0
        for s in np.unique(st_arr):
            sm = st_arr == s
            yt = y_out[sm & (y_d == 1)]; yc = y_out[sm & (y_d == 0)]
            if len(yt) == 0 or len(yc) == 0:
                continue
            n_t, n_c = len(yt), len(yc)
            T_sum += n_t * (yt.mean() - yc.mean())
            if n_t > 1 and n_c > 1:
                var_sum += n_t * (yt.var() / n_t + yc.var() / n_c)
        p_val = 1 - _norm.cdf(T_sum / np.sqrt(var_sum)) if var_sum > 0 else np.nan
        results.append({"gamma": gamma, "p_value": p_val})
    return pd.DataFrame(results)


def compute_ps(X, y, cv):
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    ps = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    return ps


# ══════════════════════════════════════════════════════════════════════════════
# Readable feature names
# ══════════════════════════════════════════════════════════════════════════════

READABLE = {
    "n_edits": "Prior edits (total)", "n_edits_7d": "Prior edits (7d)", "n_edits_1d": "Prior edits (1d)",
    "ns0_mainspace": "Mainspace edits", "ns2_userpage": "User-page edits",
    "ns3_usertalk": "User-talk edits", "ns4_wp": "Wikipedia-ns edits", "ns118_draft": "Draft-ns edits",
    "n_unique_ns": "Unique namespaces", "mainspace_ratio": "Mainspace ratio", "draft_ratio": "Draft ratio",
    "avg_sizediff": "Avg byte change", "std_sizediff": "Std byte change",
    "max_sizediff": "Max byte change", "neg_sizediff_ratio": "Neg-change ratio",
    "hours_since_last_edit": "Hours since last edit", "active_span_hours": "Active span (h)",
    "account_age_hours": "Account age (h)", "n_reverts": "Prior reverts", "n_ai_reverts": "AI reverts",
    "revert_rate_pre": "Pre-Q revert rate",
    "tag_visualeditor": "VisualEditor", "tag_mobile": "Mobile edit",
    "tag_newcomer_task": "Newcomer Task", "tag_editcheck_newref": "EditCheck ref",
    "tag_mw_reverted": "Reverted (tag)", "tag_discussion": "Discussion comment",
    "log_create": "Page creations", "log_thanks": "Thanks received",
    "n_abuse": "Abuse filter hits", "n_abuse_warn": "Abuse warnings",
    "q_weekday": "Q weekday", "q_hour_utc": "Q hour (UTC)", "q_is_weekend": "Q on weekend",
    "q_has_sig": "Q has signature", "q_has_unsigned": "Q unsigned",
    "q_link_count": "Q links", "q_templatt_count": "Q templates", "q_has_url": "Q has URL",
    "q_mentions_wp": "Q mentions WP", "q_mentions_help": "Q mentions help", "q_is_indented": "Q indented",
    "q_body_char_len": "Q length (chars)", "q_body_word_count": "Q length (words)",
    "q_vader_neg": "Sentiment (neg)", "q_vader_neu": "Sentiment (neu)",
    "q_vader_pos": "Sentiment (pos)", "q_vader_compound": "Sentiment (compound)",
    "q_tb_polarity": "Polarity (TextBlob)", "q_tb_subjectivity": "Subjectivity",
    "q_persp_toxicity": "Persp: Toxicity", "q_persp_severe_toxicity": "Persp: Severe tox",
    "q_persp_identity_attack": "Persp: Identity attack", "q_persp_insult": "Persp: Insult",
    "q_persp_profanity": "Persp: Profanity", "q_persp_threat": "Persp: Threat",
    "q_persp_sexually_explicit": "Persp: Sexual", "q_persp_flirtation": "Persp: Flirtation",
    "q_persp_affinity_experimental": "Persp: Affinity", "q_persp_compassion_experimental": "Persp: Compassion",
    "q_persp_curiosity_experimental": "Persp: Curiosity", "q_persp_nuance_experimental": "Persp: Nuance",
    "q_persp_personal_story_experimental": "Persp: Personal story",
    "q_persp_reasoning_experimental": "Persp: Reasoning", "q_persp_respect_experimental": "Persp: Respect",
    "q_has_article_context": "Q from article context",
    "q_substantive": "Q-type: Substantive", "q_referent": "Q-type: Referent",
    "q_appraisal": "Q-type: Appraisal", "q_normative": "Q-type: Normative",
    "q_own_work": "Q-type: Own Work",
}


def readable(name):
    if name in READABLE:
        return READABLE[name]
    if name.startswith("emb_pc"):
        return f"Embedding PC{name.replace('emb_pc', '')}"
    if name.startswith("ym_"):
        return f"YM: {name.replace('ym_', '')}"
    if "politeness" in name.lower():
        tag = name.split("==")[-2] if "==" in name else name
        return f"Polite: {tag}"
    return name


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0_total = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  s13: PSM ANALYSIS")
    print(SEP)
    print(f"\nLoading {DATA}...")

    D = np.load(DATA, allow_pickle=True)

    y_treat = D["y_treat"]
    y_treat_48h = D["y_treat_48h"]
    mentee_ids = D["mentee_ids"]
    mentor_ids = D["mentor_ids"] if "mentor_ids" in D else None
    cid_order = D["cid_order"]

    X_E = D["X_E"]; X_Qtext = D["X_Qtext"]; X_Qpersp = D["X_Qpersp"]
    X_Qtype = D["X_Qtype"]; X_emb20 = D["X_emb20"]
    X_emb_full = D["X_emb_full"]; X_temporal = D["X_temporal"]; X_M = D["X_M"]

    E_cols = list(D["E_cols"]); Qtext_cols = list(D["Qtext_cols"])
    Qpersp_cols = list(D["Qpersp_cols"]); Qtype_cols = list(D["Qtype_cols"])
    M_cols = list(D["M_cols"]); temporal_cols = list(D["temporal_cols"])
    emb_cols = [f"emb_pc{i}" for i in range(20)]

    q_year_month = D["q_year_month"].astype(str)

    # Outcomes
    OC_KEYS = ["primary", "n_mainspace_edits_14d", "primary_constructive", "sec2",
               "constructive_edit_15_60d", "reverted_any", "active_days_30d",
               "constructive_days_30d", "unique_ns", "cross_day_constructive_14d"]
    WINDOW_KEYS = [f"mainspace_{w}d" for w in [7, 14, 21, 28, 30, 60, 180]]
    OC = {}
    for k in OC_KEYS + WINDOW_KEYS:
        OC[k] = D[f"oc_{k}"]

    N = len(y_treat)
    n_treated = int(y_treat.sum())
    n_control = N - n_treated

    print(f"  N = {N:,}  (Treated={n_treated:,}, Control={n_control:,})")
    print(f"  Treated (48h): {int(y_treat_48h.sum()):,}")

    # Build main feature matrix
    X_main = np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_emb20, X_temporal])
    ALL_COLS = E_cols + Qtext_cols + Qpersp_cols + Qtype_cols + emb_cols + temporal_cols
    X_main_s = StandardScaler().fit_transform(X_main)

    print(f"  X_main: {X_main.shape[1]} features")

    # Outcome dimensions for main effects
    DIMS = [
        ("1+_mainspace_edit_14d",  "primary",              "primary"),        # retention (binary)
        ("n_mainspace_edits_14d",  "n_mainspace_edits_14d","supplementary"),  # productivity (count)
        ("active_days_30d",        "active_days_30d",      "supplementary"),  # engagement persistence
        ("unique_namespace_14d",   "unique_ns",            "supplementary"),  # participation breadth
        ("reverted_any_14d",       "reverted_any",         "supplementary"),  # edit quality
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ══════════════════════════════════════════════════════════════════════
    # PART A: DIAGNOSTICS + FEATURE ABLATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PART A: PS DIAGNOSTICS + FEATURE ABLATION")
    print(SEP)

    # ── A1. PS estimation ──────────────────────────────────────────────────
    print("\n[A1] Propensity score estimation (5-fold CV LR)...")
    ps = compute_ps(X_main_s, y_treat, cv)
    auc_main = roc_auc_score(y_treat, ps)
    strata = pd.qcut(ps, N_STRATA, labels=False, duplicates="drop")
    n_strata = len(np.unique(strata))
    X_cov_dr = PCA(n_components=10, random_state=42).fit_transform(X_main_s)

    ps_t = ps[y_treat == 1]
    ps_c = ps[y_treat == 0]

    print(f"  AUC = {auc_main:.4f},  Strata = {n_strata}")
    print(f"\n{'':>10s} {'min':>8s} {'p5':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p95':>8s} {'max':>8s}")
    for label, arr in [("Treated", ps_t), ("Control", ps_c)]:
        print(f"{label:>10s} {arr.min():>8.4f} {np.percentile(arr,5):>8.4f} "
              f"{np.percentile(arr,25):>8.4f} {np.median(arr):>8.4f} "
              f"{np.percentile(arr,75):>8.4f} {np.percentile(arr,95):>8.4f} {arr.max():>8.4f}")

    # ── A2. PS distribution plot ───────────────────────────────────────────
    print("\n[A2] PS distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.hist(ps_c, bins=60, alpha=0.55, label=f"Control (N={len(ps_c):,})", color="#d62728", density=True)
    ax.hist(ps_t, bins=60, alpha=0.55, label=f"Treated (N={len(ps_t):,})", color="#1f77b4", density=True)
    ax.axvline(x=0.05, color="gray", ls=":", lw=0.8)
    ax.axvline(x=0.95, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Propensity Score"); ax.set_ylabel("Density")
    ax.set_title(f"PS Distribution (AUC = {auc_main:.3f})")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.boxplot([ps_c, ps_t], labels=["Control", "Treated"], widths=0.5)
    ax.axhline(y=0.05, color="red", ls=":", alpha=0.5)
    ax.axhline(y=0.95, color="red", ls=":", alpha=0.5)
    ax.set_ylabel("Propensity Score")
    ax.set_title("PS by Group")

    fig.tight_layout()
    fig.savefig(OUT_FIG / "A2_ps_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved A2_ps_distribution.pdf")

    # ── A3. Stratification diagnostics ─────────────────────────────────────
    print("\n[A3] Stratification diagnostics...")
    print(f"{'S':>3s} {'N':>7s} {'T':>7s} {'C':>7s} {'T%':>7s} {'PS_lo':>8s} {'PS_hi':>8s}")
    print("-" * 55)
    strata_rows = []
    for s in range(n_strata):
        sm = strata == s
        yt = y_treat[sm]
        row = {"Stratum": s, "N": int(sm.sum()), "Treated": int(yt.sum()),
               "Control": int(len(yt) - yt.sum()), "T_pct": yt.mean() * 100,
               "PS_lo": ps[sm].min(), "PS_hi": ps[sm].max()}
        strata_rows.append(row)
        print(f"{s:>3d} {sm.sum():>7,} {int(yt.sum()):>7,} {int(len(yt)-yt.sum()):>7,} "
              f"{yt.mean()*100:>6.1f}% {ps[sm].min():>8.4f} {ps[sm].max():>8.4f}")
    pd.DataFrame(strata_rows).to_csv(OUT_TBL / "A3_strata_diagnostics.csv", index=False)

    # ── A4. Covariate balance Love plot ────────────────────────────────────
    print("\n[A4] Covariate balance (Love plot)...")
    X_all_unscaled = np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_emb20, X_temporal])

    smd_rows = []
    for i, col in enumerate(ALL_COLS):
        vals = X_all_unscaled[:, i]
        tv = vals[y_treat == 1]; cv_vals = vals[y_treat == 0]
        pooled = np.sqrt((tv.var() + cv_vals.var()) / 2)
        smd_raw = abs(tv.mean() - cv_vals.mean()) / pooled if pooled > 0 else 0
        w_diffs, w_total = [], 0
        for s in range(n_strata):
            sm = strata == s
            t_s = vals[sm & (y_treat == 1)]; c_s = vals[sm & (y_treat == 0)]
            if len(t_s) > 0 and len(c_s) > 0:
                w = sm.sum(); w_diffs.append(w * (t_s.mean() - c_s.mean())); w_total += w
        smd_strat = abs(sum(w_diffs) / w_total) / pooled if pooled > 0 and w_total > 0 else 0
        smd_rows.append({"var": col, "raw": smd_raw, "strat": smd_strat})

    smd_df = pd.DataFrame(smd_rows)
    smd_df["label"] = smd_df["var"].map(readable)
    n_bad = (smd_df["strat"] > 0.1).sum()
    smd_df.to_csv(OUT_TBL / "A4_covariate_balance.csv", index=False)

    smd_plot = smd_df.sort_values("raw", ascending=True).reset_index(drop=True)
    n_vars = len(smd_plot)
    row_h = 0.35
    fig_h = max(16, n_vars * row_h)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    y_pos = range(n_vars)
    ax.scatter(smd_plot["raw"], y_pos, marker="x", color="#d62728", s=30, label="Raw", zorder=3)
    ax.scatter(smd_plot["strat"], y_pos, marker="o", color="#1f77b4", s=30, label="Stratified", zorder=3)
    for i in range(n_vars):
        ax.plot([smd_plot["raw"].iloc[i], smd_plot["strat"].iloc[i]], [i, i],
                color="gray", lw=0.4, alpha=0.5)
    ax.axvline(x=0.1, color="gray", ls="--", alpha=0.6, label="SMD = 0.1")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(smd_plot["label"].tolist(), fontsize=7)
    ax.set_xlabel("Absolute SMD", fontsize=11)
    ax.set_title(f"Covariate Balance: Love Plot ({len(ALL_COLS)} PS features)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, n_vars - 0.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "A4_love_plot.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"  Total covariates: {len(smd_df)}")
    print(f"  Imbalanced after stratification (SMD > 0.1): {n_bad}/{len(smd_df)}")
    if n_bad > 0:
        print("  Imbalanced variables:")
        for _, r in smd_df[smd_df["strat"] > 0.1].sort_values("strat", ascending=False).iterrows():
            print(f"    {r['var']:<40s} raw={r['raw']:.4f}  strat={r['strat']:.4f}")

    # ── A5. Common support / Positivity ────────────────────────────────────
    print("\n[A5] Common support...")
    common_lo = max(ps_t.min(), ps_c.min())
    common_hi = min(ps_t.max(), ps_c.max())
    n_outside = N - ((ps >= common_lo) & (ps <= common_hi)).sum()

    print(f"  Treated PS range:  [{ps_t.min():.4f}, {ps_t.max():.4f}]")
    print(f"  Control PS range:  [{ps_c.min():.4f}, {ps_c.max():.4f}]")
    print(f"  Common support:    [{common_lo:.4f}, {common_hi:.4f}]")
    print(f"  Outside common support: {n_outside:,} ({n_outside/N*100:.2f}%)")

    print(f"\n  {'Threshold':<12s} {'Treated':>10s} {'Control':>10s} {'Total':>10s}")
    print("  " + "-" * 48)
    for thresh in [0.01, 0.05, 0.10]:
        nt = (ps_t < thresh).sum(); nc = (ps_c < thresh).sum()
        print(f"  PS < {thresh:.2f}    {nt:>10,} {nc:>10,} {nt+nc:>10,}")
    for thresh in [0.90, 0.95, 0.99]:
        nt = (ps_t > thresh).sum(); nc = (ps_c > thresh).sum()
        print(f"  PS > {thresh:.2f}    {nt:>10,} {nc:>10,} {nt+nc:>10,}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ps_c, bins=50, alpha=0.5, label="Control", color="#d62728", range=(0, 1))
    ax.hist(ps_t, bins=50, alpha=0.5, label="Treated", color="#1f77b4", range=(0, 1))
    ax.axvspan(0, common_lo, alpha=0.15, color="gray", label="Outside common support")
    ax.axvspan(common_hi, 1, alpha=0.15, color="gray")
    ax.set_xlabel("Propensity Score"); ax.set_ylabel("Count")
    ax.set_title("Common Support Region")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "A5_common_support.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved A5_common_support.pdf")

    # ── A6. Feature importance ─────────────────────────────────────────────
    print("\n[A6] PS model feature importance (top 25)...")
    lr_full = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    lr_full.fit(X_main_s, y_treat)
    coefs = lr_full.coef_[0]

    fi_df = pd.DataFrame({"Feature": ALL_COLS, "Coef": coefs, "AbsCoef": np.abs(coefs)})
    fi_df["Label"] = fi_df["Feature"].map(readable)
    fi_df = fi_df.sort_values("AbsCoef", ascending=False)
    fi_df.to_csv(OUT_TBL / "A6_feature_importance.csv", index=False)

    top_k = 25
    top = fi_df.head(top_k).sort_values("AbsCoef", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in top["Coef"]]
    ax.barh(range(len(top)), top["Coef"], color=colors, height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Label"].tolist(), fontsize=8)
    ax.axvline(x=0, color="black", lw=0.8)
    ax.set_xlabel("Standardized LR Coefficient")
    ax.set_title(f"Top {top_k} Features Predicting Mentor Reply ({len(ALL_COLS)} features)")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "A6_feature_importance.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"  Top 10:")
    for _, r in fi_df.head(10).iterrows():
        d = "↑ reply" if r["Coef"] > 0 else "↓ reply"
        print(f"    {r['Label']:<40s} coef={r['Coef']:+.4f}  {d}")

    # ── A7. Feature ablation ───────────────────────────────────────────────
    print("\n[A7] Feature ablation...")

    ablation_sets = {
        "Full model (E+Qtext+Qpersp+Qtype+Emb+T)": np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_emb20, X_temporal]),
        "Full – Qtype (no LLM annotation)":         np.hstack([X_E, X_Qtext, X_Qpersp, X_emb20, X_temporal]),
        "Full – temporal":                           np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_emb20]),
        "Full – embeddings":                         np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_temporal]),
        "Full – Qtext":                              np.hstack([X_E, X_Qpersp, X_Qtype, X_emb20, X_temporal]),
        "Full – Qpersp":                             np.hstack([X_E, X_Qtext, X_Qtype, X_emb20, X_temporal]),
        "Full + full emb (1024d)":                   np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_emb_full, X_temporal]),
        "Edit history only":                         X_E,
        "E + embeddings":                            np.hstack([X_E, X_emb20]),
        "E + Qtext":                                 np.hstack([X_E, X_Qtext]),
        "E + Qpersp":                                np.hstack([X_E, X_Qpersp]),
        "E + Qtype":                                 np.hstack([X_E, X_Qtype]),
        "Question features only (Qtext+Qpersp+Qtype)": np.hstack([X_Qtext, X_Qpersp, X_Qtype]),
        "Embeddings only":                           X_emb20,
        "Qtype only":                                X_Qtype,
        "Full + mentor features":                    np.hstack([X_E, X_Qtext, X_Qpersp, X_Qtype, X_emb20, X_temporal, X_M]),
    }

    ablation_results = []
    for name, X_ab in ablation_sets.items():
        X_ab_s = StandardScaler().fit_transform(X_ab)
        ps_ab = compute_ps(X_ab_s, y_treat, cv)
        auc_ab = roc_auc_score(y_treat, ps_ab)
        st_ab = pd.qcut(ps_ab, N_STRATA, labels=False, duplicates="drop")
        att_ab = strat_att(y_treat, OC["primary"], st_ab)
        ci_lo_ab, ci_hi_ab = cluster_boot_ci(y_treat, OC["primary"], st_ab, mentee_ids)
        sig_ab = "*" if (ci_lo_ab > 0 or ci_hi_ab < 0) else ""

        # Compute SMD > 0.1 count for this ablation
        X_ab_unscaled = X_ab
        n_bad_ab = 0
        for j in range(X_ab_unscaled.shape[1]):
            vals = X_ab_unscaled[:, j]
            tv = vals[y_treat == 1]; cv2 = vals[y_treat == 0]
            pooled = np.sqrt((tv.var() + cv2.var()) / 2)
            if pooled > 0:
                w_diffs2, w_total2 = [], 0
                for s in np.unique(st_ab):
                    sm = st_ab == s
                    t_s = vals[sm & (y_treat == 1)]; c_s = vals[sm & (y_treat == 0)]
                    if len(t_s) > 0 and len(c_s) > 0:
                        w = sm.sum(); w_diffs2.append(w * (t_s.mean() - c_s.mean())); w_total2 += w
                smd_s = abs(sum(w_diffs2) / w_total2) / pooled if w_total2 > 0 else 0
                if smd_s > 0.1:
                    n_bad_ab += 1

        ablation_results.append({
            "Model": name, "p": X_ab.shape[1], "AUC": auc_ab,
            "ATT": att_ab, "CI_lo": ci_lo_ab, "CI_hi": ci_hi_ab,
            "Sig": sig_ab, "SMD_gt_0.1": n_bad_ab
        })
        print(f"  {name:<50s} p={X_ab.shape[1]:>5d}  AUC={auc_ab:.4f}  "
              f"ATT={att_ab:+.4f} [{ci_lo_ab:+.4f},{ci_hi_ab:+.4f}] {sig_ab}  "
              f"SMD>0.1={n_bad_ab}")

    abl_df = pd.DataFrame(ablation_results)
    abl_df.to_csv(OUT_TBL / "A7_feature_ablation.csv", index=False)

    # Ablation plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    abl_sorted = abl_df.sort_values("AUC")
    ax = axes[0]
    colors_ab = ["#e74c3c" if "Full model" in m else "#1f77b4" for m in abl_sorted["Model"]]
    ax.barh(range(len(abl_sorted)), abl_sorted["AUC"], color=colors_ab, alpha=0.8)
    ax.set_yticks(range(len(abl_sorted)))
    ax.set_yticklabels(abl_sorted["Model"], fontsize=7)
    ax.set_xlabel("AUC")
    ax.set_title("PS Model AUC by Feature Set")
    ax.axvline(x=auc_main, color="red", ls="--", alpha=0.5, label=f"Full={auc_main:.3f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    abl_sorted2 = abl_df.sort_values("ATT")
    for i, (_, r) in enumerate(abl_sorted2.iterrows()):
        c = "#e74c3c" if "Full model" in r["Model"] else ("#1f77b4" if r["Sig"] == "*" else "#999999")
        ax.plot([r["CI_lo"], r["CI_hi"]], [i, i], color=c, lw=2)
        ax.plot(r["ATT"], i, "o", color=c, markersize=7)
    ax.axvline(x=0, color="black", lw=0.8)
    ax.set_yticks(range(len(abl_sorted2)))
    ax.set_yticklabels(abl_sorted2["Model"], fontsize=7)
    ax.set_xlabel("ATT (PRIMARY)")
    ax.set_title("ATT Stability Across Feature Sets")

    fig.tight_layout()
    fig.savefig(OUT_FIG / "A7_feature_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved A7_feature_ablation.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # PART B: MAIN EFFECTS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PART B: MAIN EFFECTS")
    print(SEP)

    # ── B1-B3. Main results ────────────────────────────────────────────────
    print("\n[B1-B3] Computing ATT, DR, Cohen's d for 5 outcomes...")
    results_main = {}
    rows_for_table = []

    mentor_boot_rows = []
    for dim_name, dim_key, variant in DIMS:
        y_out = OC[dim_key].copy()
        sd_ctrl = y_out[y_treat == 0].std()
        naive = y_out[y_treat == 1].mean() - y_out[y_treat == 0].mean()
        att = strat_att(y_treat, y_out, strata)
        dr = strat_dr(y_treat, y_out, strata, X_cov_dr)
        ci_lo, ci_hi = cluster_boot_ci(y_treat, y_out, strata, mentee_ids)
        sig = "*" if (ci_lo > 0 or ci_hi < 0) else ""
        cohen_d = att / sd_ctrl if sd_ctrl > 0 else 0.0
        ci_lo_d = ci_lo / sd_ctrl if sd_ctrl > 0 else 0.0
        ci_hi_d = ci_hi / sd_ctrl if sd_ctrl > 0 else 0.0

        # Mentor-level cluster bootstrap
        if mentor_ids is not None:
            mci_lo, mci_hi = cluster_boot_ci(y_treat, y_out, strata, mentor_ids)
            msig = "*" if (mci_lo > 0 or mci_hi < 0) else ""
        else:
            mci_lo, mci_hi, msig = np.nan, np.nan, ""

        results_main[dim_key] = {
            "naive": naive, "att": att, "dr": dr,
            "ci": (ci_lo, ci_hi), "sig": sig, "name": dim_name,
            "sd_ctrl": sd_ctrl, "cohen_d": cohen_d,
            "ci_lo_d": ci_lo_d, "ci_hi_d": ci_hi_d,
            "mentor_ci": (mci_lo, mci_hi), "mentor_sig": msig,
        }
        rows_for_table.append({
            "Outcome": dim_name, "Naive": naive, "ATT": att, "DR": dr,
            "CI_lo": ci_lo, "CI_hi": ci_hi,
            "Cohen_d": cohen_d, "d_CI_lo": ci_lo_d, "d_CI_hi": ci_hi_d,
            "Sig": sig, "variant": variant,
        })
        mentor_boot_rows.append({
            "Outcome": dim_name, "ATT": att,
            "Mentee_CI_lo": ci_lo, "Mentee_CI_hi": ci_hi, "Mentee_Sig": sig,
            "Mentor_CI_lo": mci_lo, "Mentor_CI_hi": mci_hi, "Mentor_Sig": msig,
        })

    res_df = pd.DataFrame(rows_for_table)
    res_df.to_csv(OUT_TBL / "B_main_results.csv", index=False)
    print(res_df[["Outcome", "Naive", "ATT", "DR", "CI_lo", "CI_hi", "Cohen_d", "Sig"]].to_string(
        index=False, float_format="%+.4f"))

    # Print mentor-level bootstrap comparison
    if mentor_ids is not None:
        n_mentors = len(np.unique(mentor_ids))
        print(f"\n  Mentor-level cluster bootstrap (N_mentors={n_mentors:,}, {N_BOOT} resamples):")
        mboot_df = pd.DataFrame(mentor_boot_rows)
        mboot_df.to_csv(OUT_TBL / "B_mentor_boot.csv", index=False)
        print(f"  {'Outcome':<30s} {'ATT':>7s} {'Mentee CI':>18s} {'Sig':>4s} {'Mentor CI':>18s} {'Sig':>4s}")
        print(f"  {'-'*30} {'-'*7} {'-'*18} {'-'*4} {'-'*18} {'-'*4}")
        for r in mentor_boot_rows:
            print(f"  {r['Outcome']:<30s} {r['ATT']:>+7.4f} "
                  f"[{r['Mentee_CI_lo']:>+.4f}, {r['Mentee_CI_hi']:>+.4f}] {r['Mentee_Sig']:>4s} "
                  f"[{r['Mentor_CI_lo']:>+.4f}, {r['Mentor_CI_hi']:>+.4f}] {r['Mentor_Sig']:>4s}")

    # ── B4. Forest plot ────────────────────────────────────────────────────
    print("\n[B4] Forest plot...")
    fig, ax = plt.subplots(figsize=(9, len(res_df) * 0.7 + 1.5))
    y_positions = list(range(len(res_df)))

    for i, (_, r) in enumerate(res_df.iterrows()):
        yp = y_positions[i]
        c = "#1f77b4" if r["Sig"] == "*" else "#cccccc"
        ax.plot([r["d_CI_lo"], r["d_CI_hi"]], [yp, yp], color=c, lw=2.5, solid_capstyle="round")
        ax.plot(r["Cohen_d"], yp, "o", color=c, markersize=8, zorder=5)
        att_str = f"ATT={r['ATT']:+.4f}"
        sig_str = " *" if r["Sig"] == "*" else ""
        ax.annotate(f"{att_str}{sig_str}", xy=(r["d_CI_hi"] + 0.003, yp),
                    fontsize=8, va="center", color="#333333")

    ax.axvline(x=0, color="black", lw=0.8, ls="-")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(res_df["Outcome"].tolist(), fontsize=9)
    ax.set_xlabel("Cohen's d (ATT / SD_control)", fontsize=10)
    ax.set_title(f"PSM Treatment Effects on Newcomer Outcomes\n"
                 f"Stratified ATT · Cluster bootstrap 95% CI · N={N:,}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "B4_forest_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved B4_forest_plot.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # PART C: SENSITIVITY & ROBUSTNESS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PART C: SENSITIVITY & ROBUSTNESS")
    print(SEP)

    # ── C1. Strata sensitivity ─────────────────────────────────────────────
    print("\n[C1] Strata sensitivity (K=5/10/20)...")
    strata_sens = []
    for label, dim_key, _ in DIMS:
        y_out = OC[dim_key]
        for ns in [5, 10, 20]:
            st_alt = pd.qcut(ps, ns, labels=False, duplicates="drop")
            att_alt = strat_att(y_treat, y_out, st_alt)
            cl, ch = cluster_boot_ci(y_treat, y_out, st_alt, mentee_ids)
            sig = "*" if (cl > 0 or ch < 0) else ""
            strata_sens.append({"Outcome": label, "Strata": ns, "ATT": att_alt,
                                "CI_lo": cl, "CI_hi": ch, "Sig": sig})

    ss_df = pd.DataFrame(strata_sens)
    ss_df.to_csv(OUT_TBL / "C1_strata_sensitivity.csv", index=False)
    for outcome in ss_df["Outcome"].unique():
        sub = ss_df[ss_df["Outcome"] == outcome]
        print(f"  {outcome}:")
        for _, r in sub.iterrows():
            print(f"    K={r['Strata']:>2d}  ATT={r['ATT']:+.4f}  [{r['CI_lo']:+.4f},{r['CI_hi']:+.4f}] {r['Sig']}")

    # ── C2. D=48h treatment ────────────────────────────────────────────────
    print("\n[C2] D=48h treatment definition...")
    ps_48 = compute_ps(X_main_s, y_treat_48h, cv)
    st_48 = pd.qcut(ps_48, N_STRATA, labels=False, duplicates="drop")
    att_48 = strat_att(y_treat_48h, OC["primary"], st_48)
    ci48_lo, ci48_hi = cluster_boot_ci(y_treat_48h, OC["primary"], st_48, mentee_ids)
    sig_48 = "*" if (ci48_lo > 0 or ci48_hi < 0) else ""
    print(f"  D=48h:  ATT={att_48:+.4f}  [{ci48_lo:+.4f},{ci48_hi:+.4f}] {sig_48}")
    print(f"  D=ever: ATT={results_main['primary']['att']:+.4f}  (comparison)")

    # ── C3. PS trimming ────────────────────────────────────────────────────
    print("\n[C3] PS trimming...")
    trim_results = []
    for lo, hi in [(0.01, 0.99), (0.05, 0.95), (0.10, 0.90), (0.15, 0.85)]:
        tm = (ps >= lo) & (ps <= hi)
        n_trim = N - tm.sum()
        st_t = pd.qcut(ps[tm], N_STRATA, labels=False, duplicates="drop")
        att_t = strat_att(y_treat[tm], OC["primary"][tm], st_t)
        cl_t, ch_t = cluster_boot_ci(y_treat[tm], OC["primary"][tm], st_t, mentee_ids[tm])
        sig_t = "*" if (cl_t > 0 or ch_t < 0) else ""
        trim_results.append({"Range": f"[{lo},{hi}]", "Trimmed": n_trim, "N_remain": int(tm.sum()),
                             "ATT": att_t, "CI_lo": cl_t, "CI_hi": ch_t, "Sig": sig_t})
        print(f"  [{lo:.2f},{hi:.2f}]  trimmed={n_trim:,}  ATT={att_t:+.4f}  [{cl_t:+.4f},{ch_t:+.4f}] {sig_t}")
    pd.DataFrame(trim_results).to_csv(OUT_TBL / "C3_ps_trimming.csv", index=False)

    # ── C4. Window sensitivity ─────────────────────────────────────────────
    print("\n[C4] Window sensitivity (mainspace edits)...")
    window_sens = []
    for wd, oc_key in [(7, "mainspace_7d"), (14, "mainspace_14d"), (21, "mainspace_21d"),
                       (28, "mainspace_28d"), (30, "mainspace_30d"),
                       (60, "mainspace_60d"), (180, "mainspace_180d")]:
        y_w = OC[oc_key]
        att_w = strat_att(y_treat, y_w, strata)
        cl, ch = cluster_boot_ci(y_treat, y_w, strata, mentee_ids)
        sig = "*" if (cl > 0 or ch < 0) else ""
        window_sens.append({"window": wd, "ATT": att_w, "CI_lo": cl, "CI_hi": ch, "Sig": sig})

    ws_df = pd.DataFrame(window_sens)
    ws_df.to_csv(OUT_TBL / "C4_window_sensitivity.csv", index=False)
    for _, r in ws_df.iterrows():
        print(f"  {r['window']:>3d}d  ATT={r['ATT']:+.4f}  [{r['CI_lo']:+.4f},{r['CI_hi']:+.4f}] {r['Sig']}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ws_df["window"], ws_df["ATT"],
                yerr=[ws_df["ATT"] - ws_df["CI_lo"], ws_df["CI_hi"] - ws_df["ATT"]],
                fmt="o-", capsize=5, color="#1f77b4", lw=2, markersize=7)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Window (days)"); ax.set_ylabel("ATT")
    ax.set_title("Mainspace Edit Retention: Window Sensitivity")
    ax.set_xticks([7, 14, 21, 28, 30, 60, 180])
    fig.tight_layout()
    fig.savefig(OUT_FIG / "C4_window_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved C4_window_sensitivity.pdf")

    # ── C5. Rosenbaum bounds + E-value ─────────────────────────────────────
    print("\n[C5] Rosenbaum sensitivity bounds...")
    gammas = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    sens = rosenbaum_bounds(y_treat, OC["primary"], strata, gammas)
    sens.to_csv(OUT_TBL / "C5_rosenbaum_bounds.csv", index=False)

    critical_gamma = None
    for _, r in sens.iterrows():
        if critical_gamma is None and r["p_value"] >= 0.05:
            critical_gamma = r["gamma"]

    for _, r in sens.iterrows():
        flag = " ← critical" if r["gamma"] == critical_gamma else ""
        print(f"  Γ={r['gamma']:.1f}  p={r['p_value']:.4f}{flag}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sens["gamma"], sens["p_value"], "o-", color="#1f77b4", lw=2, markersize=7)
    ax.axhline(y=0.05, color="black", ls="--", label="α = 0.05")
    if critical_gamma:
        ax.axvline(x=critical_gamma, color="gray", ls=":", alpha=0.7,
                   label=f"Critical Γ = {critical_gamma:.1f}")
    ax.set_xlabel("Γ (sensitivity parameter)")
    ax.set_ylabel("P-value")
    ax.set_title("Rosenbaum Sensitivity: PRIMARY Outcome")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "C5_rosenbaum_bounds.pdf", bbox_inches="tight")
    plt.close(fig)

    # E-value
    att_pt = results_main["primary"]["att"]
    ci_lo_pt = results_main["primary"]["ci"][0]
    if att_pt > 0:
        rr = 1 + att_pt
        e_val = rr + np.sqrt(rr * (rr - 1))
        if ci_lo_pt > 0:
            rr_ci = 1 + ci_lo_pt
            e_val_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
        else:
            e_val_ci = 1.0
        print(f"\n  E-value (point):  {e_val:.3f}")
        print(f"  E-value (CI):     {e_val_ci:.3f}")
        print(f"  An unmeasured confounder would need RR ≥ {e_val_ci:.2f} with both")
        print(f"  treatment and outcome to explain away the observed effect.")
    else:
        e_val = e_val_ci = np.nan
        print(f"\n  ATT ≤ 0, E-value not applicable.")

    # ── C6. Robustness summary ─────────────────────────────────────────────
    print("\n[C6] Robustness summary...")
    rob_rows = []
    rob_rows.append({"Spec": "Main (K=10)", "ATT": results_main["primary"]["att"],
                     "CI_lo": results_main["primary"]["ci"][0],
                     "CI_hi": results_main["primary"]["ci"][1]})
    rob_rows.append({"Spec": "D=48h", "ATT": att_48, "CI_lo": ci48_lo, "CI_hi": ci48_hi})

    for r in trim_results:
        rob_rows.append({"Spec": f"Trim {r['Range']}", "ATT": r["ATT"],
                         "CI_lo": r["CI_lo"], "CI_hi": r["CI_hi"]})

    for _, r in pd.DataFrame(strata_sens).query("Outcome == '1+_mainspace_edit_14d'").iterrows():
        rob_rows.append({"Spec": f"K={int(r['Strata'])}", "ATT": r["ATT"],
                         "CI_lo": r["CI_lo"], "CI_hi": r["CI_hi"]})

    rob_df = pd.DataFrame(rob_rows)
    rob_df["Sig"] = rob_df.apply(lambda r: "*" if (r["CI_lo"] > 0 or r["CI_hi"] < 0) else "", axis=1)
    rob_df.to_csv(OUT_TBL / "C6_robustness_summary.csv", index=False)

    print(rob_df.to_string(index=False, float_format="%+.4f"))

    fig, ax = plt.subplots(figsize=(8, max(4, len(rob_df) * 0.45)))
    y_pos = list(range(len(rob_df)))[::-1]
    for i, (_, r) in enumerate(rob_df.iterrows()):
        yp = y_pos[i]
        is_main = "Main" in r["Spec"]
        c = "#1f77b4" if is_main else "#666666"
        lw = 2.5 if is_main else 1.5
        ms = 9 if is_main else 6
        ax.plot([r["CI_lo"], r["CI_hi"]], [yp, yp], color=c, lw=lw)
        ax.plot(r["ATT"], yp, "o" if is_main else "s", color=c, markersize=ms, zorder=5)

    ax.axvline(x=0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rob_df["Spec"].tolist(), fontsize=9)
    ax.set_xlabel("ATT (PRIMARY: 1+ mainspace edit 14d)")
    ax.set_title("Robustness: ATT Stability Across Specifications")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "C6_robustness_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved C6_robustness_summary.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0_total
    print(f"\n{SEP}")
    print(f"  s13 COMPLETE")
    print(f"  N = {N:,}  (T={n_treated:,}, C={n_control:,})")
    print(f"  PS model AUC = {auc_main:.4f}")
    print(f"  Covariates with SMD > 0.1 after stratification: {n_bad}")
    print(f"  Primary outcome ATT = {results_main['primary']['att']:+.4f} "
          f"[{results_main['primary']['ci'][0]:+.4f}, "
          f"{results_main['primary']['ci'][1]:+.4f}] "
          f"{results_main['primary']['sig']}")
    if not np.isnan(e_val_ci):
        print(f"  E-value (CI) = {e_val_ci:.3f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n  Figures: {OUT_FIG}")
    print(f"  Tables:  {OUT_TBL}")
    print(SEP)


if __name__ == "__main__":
    main()
