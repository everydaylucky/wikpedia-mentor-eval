"""
s16b: Embedding Direction Analysis
    - Which semantic directions in the question embedding predict active_days_14d?
    - Which question features (text, Perspective API, politeness) correlate with retention?
    - Frisch-Waugh residualization to partial out mentee covariates

Scope: Technical treated only (same as s16).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from scipy import stats
import json, time, warnings
warnings.filterwarnings("ignore")

t0 = time.time()
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "s12" / "psm_data" / "psm_dataset.npz"
REPLY_F = BASE / "data" / "s15" / "reply_features.csv"
TURNS = BASE / "data" / "s8" / "s8_first_turns.jsonl"
OUT = BASE / "data" / "s16"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  s16b: Embedding Direction & Question Feature Analysis")
print("=" * 60)

# ── 1. Load & filter to Technical treated ────────────────────
D = np.load(DATA, allow_pickle=True)
cids = D["cid_order"]
y_treat = D["y_treat"]
X_Qtype = D["X_Qtype"]
q1 = X_Qtype[:, 0]  # substantive/technical

tech_treated = (y_treat == 1) & (q1 == 1) & (X_Qtype[:, 1] == 0) & (X_Qtype[:, 2] == 0) & (X_Qtype[:, 3] == 0)
idx = np.where(tech_treated)[0]
print(f"\nTechnical treated: N = {len(idx)}")

emb = D["X_emb_full"][idx]           # (N, 1024)
X_Qtext = D["X_Qtext"][idx]          # (N, 37)
X_Qpersp = D["X_Qpersp"][idx]        # (N, 8)
X_E = D["X_E"][idx]                   # mentee editing history
X_M = D["X_M"][idx]                   # mentor features
X_temporal = D["X_temporal"][idx]

y_a14 = D["oc_active_days_14d"][idx].astype(float)
y_ret = D["oc_primary"][idx].astype(float)
y_a30 = D["oc_active_days_30d"][idx].astype(float)
cids_tt = cids[idx]

Qtext_cols = list(D["Qtext_cols"])
Qpersp_cols = list(D["Qpersp_cols"])
E_cols = list(D["E_cols"])
M_cols = list(D["M_cols"])

# Also load reply features for joint analysis
rf = pd.read_csv(REPLY_F).set_index("conversation_id")
REPLY_COLS = [
    "r_words", "r_sentences", "r_type_token_ratio",
    "r_n_question_marks", "r_has_greeting", "r_has_thanks",
    "r_n_policy", "r_n_wikilink", "r_n_link", "r_n_resources",
    "r_pronoun_you_rate", "r_pronoun_i_rate", "r_pronoun_we_rate",
    "r_vader_compound", "r_tb_polarity",
    "r_flesch_kincaid", "r_has_steps",
    "r_reply_lag_hours", "r_reply_q_word_ratio",
]
rf_aligned = rf.reindex(cids_tt)
valid = rf_aligned[REPLY_COLS].notna().all(axis=1).values
print(f"With reply features: N = {valid.sum()}")

# Work with full N for question-only analysis, valid subset for joint
emb_v = emb[valid]
X_Qtext_v = X_Qtext[valid]
X_Qpersp_v = X_Qpersp[valid]
X_E_v = X_E[valid]
X_M_v = X_M[valid]
y_a14_v = y_a14[valid]
y_ret_v = y_ret[valid]
R_v = rf_aligned.loc[valid, REPLY_COLS].values.astype(float)
N = valid.sum()

# ══════════════════════════════════════════════════════════════
# PART 1: Raw Question Feature Correlations with active_days_14d
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 1: Question Feature Correlations (raw)")
print("=" * 60)

all_q_cols = Qtext_cols + Qpersp_cols
X_Q_all = np.hstack([X_Qtext, X_Qpersp])

rows = []
for i, col in enumerate(all_q_cols):
    r_val, p_val = stats.spearmanr(X_Q_all[idx == idx, i] if False else X_Q_all[:, i], y_a14)
    rows.append({"feature": col, "spearman_r": r_val, "p_value": p_val})
corr_df = pd.DataFrame(rows).sort_values("spearman_r", key=abs, ascending=False)
corr_df["sig"] = corr_df["p_value"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")

print("\nAll question features ranked by |Spearman r| with active_days_14d:")
for _, r in corr_df.iterrows():
    print(f"  {r['feature']:>50s}  r={r['spearman_r']:+.4f}  p={r['p_value']:.1e} {r['sig']}")

corr_df.to_csv(OUT / "s16b_question_feature_correlations_raw.csv", index=False)

# ══════════════════════════════════════════════════════════════
# PART 2: Frisch-Waugh — partial out mentee + mentor + temporal
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 2: Frisch-Waugh Residualized Correlations")
print("=" * 60)

# Controls: mentee editing history + mentor features + temporal
X_controls = np.hstack([X_E, X_M, X_temporal])
scaler_c = StandardScaler()
X_c_scaled = scaler_c.fit_transform(X_controls)

# Residualize y
reg_y = Ridge(alpha=1.0).fit(X_c_scaled, y_a14)
y_resid = y_a14 - reg_y.predict(X_c_scaled)
print(f"  Controls R² on active_days_14d: {reg_y.score(X_c_scaled, y_a14):.4f}")

# Residualize each question feature
rows_fw = []
for i, col in enumerate(all_q_cols):
    x_i = X_Q_all[:, i]
    reg_x = Ridge(alpha=1.0).fit(X_c_scaled, x_i)
    x_resid = x_i - reg_x.predict(X_c_scaled)
    r_val, p_val = stats.spearmanr(x_resid, y_resid)
    rows_fw.append({"feature": col, "partial_spearman_r": r_val, "p_value": p_val})

fw_df = pd.DataFrame(rows_fw).sort_values("partial_spearman_r", key=abs, ascending=False)
fw_df["sig"] = fw_df["p_value"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")

print("\nPartial correlations (controlling mentee + mentor + temporal):")
for _, r in fw_df.iterrows():
    print(f"  {r['feature']:>50s}  r={r['partial_spearman_r']:+.4f}  p={r['p_value']:.1e} {r['sig']}")

fw_df.to_csv(OUT / "s16b_question_feature_correlations_FW.csv", index=False)

# ══════════════════════════════════════════════════════════════
# PART 3: Reply Feature Correlations (raw + FW)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 3: Reply Feature Correlations (raw + FW)")
print("=" * 60)

# Raw
rows_r_raw = []
for i, col in enumerate(REPLY_COLS):
    r_val, p_val = stats.spearmanr(R_v[:, i], y_a14_v)
    rows_r_raw.append({"feature": col, "spearman_r": r_val, "p_value": p_val})
r_raw_df = pd.DataFrame(rows_r_raw).sort_values("spearman_r", key=abs, ascending=False)
r_raw_df["sig"] = r_raw_df["p_value"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")

# FW for reply features
X_c_v = np.hstack([X_E_v, X_M_v])
scaler_cv = StandardScaler()
X_cv_scaled = scaler_cv.fit_transform(X_c_v)
reg_yv = Ridge(alpha=1.0).fit(X_cv_scaled, y_a14_v)
y_resid_v = y_a14_v - reg_yv.predict(X_cv_scaled)

rows_r_fw = []
for i, col in enumerate(REPLY_COLS):
    x_i = R_v[:, i]
    reg_x = Ridge(alpha=1.0).fit(X_cv_scaled, x_i)
    x_resid = x_i - reg_x.predict(X_cv_scaled)
    r_val, p_val = stats.spearmanr(x_resid, y_resid_v)
    rows_r_fw.append({"feature": col, "partial_spearman_r": r_val, "p_value": p_val})
r_fw_df = pd.DataFrame(rows_r_fw).sort_values("partial_spearman_r", key=abs, ascending=False)
r_fw_df["sig"] = r_fw_df["p_value"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")

print("\nRaw reply feature correlations with active_days_14d:")
for _, r in r_raw_df.iterrows():
    print(f"  {r['feature']:>30s}  r={r['spearman_r']:+.4f}  p={r['p_value']:.1e} {r['sig']}")

print("\nPartial (FW) reply feature correlations:")
for _, r in r_fw_df.iterrows():
    print(f"  {r['feature']:>30s}  r={r['partial_spearman_r']:+.4f}  p={r['p_value']:.1e} {r['sig']}")

r_raw_df.to_csv(OUT / "s16b_reply_feature_correlations_raw.csv", index=False)
r_fw_df.to_csv(OUT / "s16b_reply_feature_correlations_FW.csv", index=False)

# ══════════════════════════════════════════════════════════════
# PART 4: Embedding Direction Analysis
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 4: Embedding Direction Analysis")
print("=" * 60)

# 4a. Direct ridge regression: emb -> active_days_14d
scaler_emb = StandardScaler()
emb_scaled = scaler_emb.fit_transform(emb)

# Raw: embedding predicting a14
from sklearn.model_selection import cross_val_score
ridge_emb = Ridge(alpha=10.0)
scores_raw = cross_val_score(ridge_emb, emb_scaled, y_a14, cv=5, scoring="r2")
print(f"\n  Embedding -> active_days_14d (5-fold CV R²): {scores_raw.mean():.4f} ± {scores_raw.std():.4f}")

# With controls
X_all = np.hstack([emb_scaled, StandardScaler().fit_transform(X_controls)])
scores_all = cross_val_score(Ridge(alpha=10.0), X_all, y_a14, cv=5, scoring="r2")
scores_ctrl = cross_val_score(Ridge(alpha=10.0), StandardScaler().fit_transform(X_controls), y_a14, cv=5, scoring="r2")
print(f"  Controls only -> active_days_14d (5-fold CV R²): {scores_ctrl.mean():.4f} ± {scores_ctrl.std():.4f}")
print(f"  Embedding + Controls (5-fold CV R²): {scores_all.mean():.4f} ± {scores_all.std():.4f}")
print(f"  Incremental R² from embedding: {scores_all.mean() - scores_ctrl.mean():.4f}")

# 4b. Frisch-Waugh on embedding: residualize both emb and y, then find direction
print("\n  Frisch-Waugh: residualizing embedding and outcome on controls...")
X_c_s = StandardScaler().fit_transform(X_controls)
reg_y_fw = Ridge(alpha=1.0).fit(X_c_s, y_a14)
y_resid_emb = y_a14 - reg_y_fw.predict(X_c_s)

emb_resid = np.zeros_like(emb)
for j in range(emb.shape[1]):
    reg_j = Ridge(alpha=1.0).fit(X_c_s, emb[:, j])
    emb_resid[:, j] = emb[:, j] - reg_j.predict(X_c_s)

# Find the optimal direction via OLS on residualized embedding
reg_dir = Ridge(alpha=1.0).fit(emb_resid, y_resid_emb)
w = reg_dir.coef_
w_norm = w / (np.linalg.norm(w) + 1e-12)
print(f"  Direction vector norm: {np.linalg.norm(w):.4f}")
print(f"  R² of direction on residualized outcome: {reg_dir.score(emb_resid, y_resid_emb):.4f}")

# Project all observations onto this direction
proj = emb @ w_norm
r_proj, p_proj = stats.spearmanr(proj, y_a14)
print(f"  Spearman(projection, active_days_14d): r={r_proj:.4f}, p={p_proj:.1e}")

# 4c. Interpret the direction: correlate projection with question features
print("\n  Interpreting the retention-predictive direction:")
interp_rows = []
for i, col in enumerate(all_q_cols):
    r_val, p_val = stats.spearmanr(proj, X_Q_all[:, i])
    interp_rows.append({"feature": col, "r_with_direction": r_val, "p_value": p_val})

# Also correlate with reply features (valid subset)
proj_v = emb_v @ w_norm
for i, col in enumerate(REPLY_COLS):
    r_val, p_val = stats.spearmanr(proj_v, R_v[:, i])
    interp_rows.append({"feature": col, "r_with_direction": r_val, "p_value": p_val})

interp_df = pd.DataFrame(interp_rows).sort_values("r_with_direction", key=abs, ascending=False)
interp_df["sig"] = interp_df["p_value"].apply(lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")

print("\n  Features most correlated with the retention-predictive embedding direction:")
for _, r in interp_df.head(25).iterrows():
    print(f"    {r['feature']:>50s}  r={r['r_with_direction']:+.4f} {r['sig']}")

interp_df.to_csv(OUT / "s16b_embedding_direction_interpretation.csv", index=False)

# 4d. What do high vs low projection questions look like?
print("\n  Extreme questions along the retention-predictive direction:")
text_map = {}
with open(TURNS) as f:
    for line in f:
        d = json.loads(line)
        text_map[d["conversation_id"]] = d.get("question_clean", "")

proj_order = np.argsort(proj)
print("\n  TOP 15 (high projection = more retention-predictive):")
for i in proj_order[-15:][::-1]:
    cid = int(cids_tt[i])
    q = text_map.get(cid, "")[:120]
    print(f"    [a14={y_a14[i]:>2.0f}] proj={proj[i]:+.3f}  {q}")

print("\n  BOTTOM 15 (low projection):")
for i in proj_order[:15]:
    cid = int(cids_tt[i])
    q = text_map.get(cid, "")[:120]
    print(f"    [a14={y_a14[i]:>2.0f}] proj={proj[i]:+.3f}  {q}")

# 4e. Binned analysis: split projection into quintiles
print("\n  Quintile analysis of the retention direction:")
quintiles = pd.qcut(proj, 5, labels=False)
for q_bin in range(5):
    mask = quintiles == q_bin
    n = mask.sum()
    a14_mean = y_a14[mask].mean()
    ret_mean = y_ret[mask].mean()
    a30_mean = D["oc_active_days_30d"][idx][mask].mean()
    print(f"    Q{q_bin+1}: n={n:>5d}, mean_a14={a14_mean:.2f}, ret14={ret_mean:.2%}, mean_a30={a30_mean:.2f}")

# ══════════════════════════════════════════════════════════════
# PART 5: Question x Reply interaction
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PART 5: Question Specificity x Reply Quality -> Outcome")
print("=" * 60)

# Use projection as continuous measure of "question specificity/engagement"
# and r_words as proxy for reply effort
proj_v_full = proj[valid]
r_words_v = R_v[:, REPLY_COLS.index("r_words")]

# Tertiles of each
q_tert = pd.qcut(proj_v_full, 3, labels=["low_proj", "mid_proj", "high_proj"])
r_tert = pd.qcut(r_words_v, 3, labels=["short_reply", "mid_reply", "long_reply"], duplicates="drop")

interact_df = pd.DataFrame({
    "q_tertile": q_tert,
    "r_tertile": r_tert,
    "active_days_14d": y_a14_v,
    "retention_14d": y_ret_v,
})

print("\n  Mean active_days_14d by question projection x reply length:")
pivot = interact_df.pivot_table(values="active_days_14d", index="q_tertile", columns="r_tertile", aggfunc="mean")
print(pivot.round(2).to_string())

print("\n  Mean retention_14d:")
pivot_ret = interact_df.pivot_table(values="retention_14d", index="q_tertile", columns="r_tertile", aggfunc="mean")
print(pivot_ret.round(3).to_string())

print("\n  Cell counts:")
pivot_n = interact_df.pivot_table(values="active_days_14d", index="q_tertile", columns="r_tertile", aggfunc="count")
print(pivot_n.to_string())

# Save
interact_df.to_csv(OUT / "s16b_interaction_grid.csv", index=False)

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"  s16b COMPLETE ({elapsed:.0f}s)")
print(f"{'=' * 60}")
