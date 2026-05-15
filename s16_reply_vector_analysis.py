"""
s16: Reply Vector Analysis (Two-Stage Clustering)

Stage 1: Cluster Technical treated questions by embedding (HDBSCAN, tight clusters)
Stage 2: Within each question cluster, sub-cluster by mentee covariates

Output:
  data/s16/stage1_question_clusters.csv  — question clusters with sample texts
  data/s16/stage2_detail.csv             — every obs with question_cluster + mentee_subgroup
  data/s16/stage2_cell_summary.csv       — per cell (question_cluster × mentee_subgroup) summary
"""

import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

t0 = time.time()

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "s12" / "psm_data" / "psm_dataset.npz"
REPLY_F = BASE / "data" / "s15" / "reply_features.csv"
TURNS = BASE / "data" / "s8" / "s8_first_turns.jsonl"
OUT = BASE / "data" / "s16"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  s16: Reply Vector Analysis (Two-Stage)")
print("=" * 60)

# ── 1. Load data ─────────────────────────────────────────────
print("\n[1] Loading data...")
D = np.load(DATA, allow_pickle=True)
cids = D["cid_order"]
y_treat = D["y_treat"]
emb_full = D["X_emb_full"]

X_E = D["X_E"]
X_Qtext = D["X_Qtext"]
X_Qpersp = D["X_Qpersp"]
X_temporal = D["X_temporal"]
X_M = D["X_M"]
X_cov = np.hstack([X_E, X_Qtext, X_Qpersp, X_temporal, X_M])

X_Qtype = D["X_Qtype"]
q1, q2, q3, q4 = X_Qtype[:, 0], X_Qtype[:, 1], X_Qtype[:, 2], X_Qtype[:, 3]

oc_primary = D["oc_primary"]
oc_retention = D["oc_cross_day_any_14d"]
oc_active14 = D["oc_active_days_14d"]
oc_active30 = D["oc_active_days_30d"]

# ── 2. Filter: Technical + Treated ───────────────────────────
print("[2] Filtering Technical treated...")
tech_treated = (y_treat == 1) & (q1 == 1) & (q2 == 0) & (q3 == 0) & (q4 == 0)
idx = np.where(tech_treated)[0]
print(f"  N = {len(idx)}")

cids_tt = cids[idx]
emb_tt = emb_full[idx]
X_cov_tt = X_cov[idx]
X_E_tt = X_E[idx]  # mentee editing history (34 features) for stage 2
y_primary_tt = oc_primary[idx]
y_retention_tt = oc_retention[idx]
y_active14_tt = oc_active14[idx]
y_active30_tt = oc_active30[idx]

# ── 3. Load reply features ───────────────────────────────────
print("[3] Loading reply features...")
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
cids_f = cids_tt[valid]
emb_f = emb_tt[valid]
X_cov_f = X_cov_tt[valid]
X_E_f = X_E_tt[valid]
y_pri_f = y_primary_tt[valid]
y_ret_f = y_retention_tt[valid]
y_a14_f = y_active14_tt[valid]
y_a30_f = y_active30_tt[valid]
R_f = rf_aligned.loc[valid, REPLY_COLS].values.astype(float)
print(f"  Final: {len(cids_f)}")

# ── 4. Load question/reply text ──────────────────────────────
print("[4] Loading text...")
text_map = {}
with open(TURNS) as f:
    for line in f:
        d = json.loads(line)
        text_map[d["conversation_id"]] = {
            "question": d.get("question_clean", ""),
            "reply": d.get("reply_clean", ""),
        }

# ══════════════════════════════════════════════════════════════
# STAGE 1: Question Embedding Clustering
# ══════════════════════════════════════════════════════════════
print("\n[Stage 1] Question embedding clustering (cosine on full 1024d)...")
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

# UMAP to 50d (preserve more semantics than 15d, avoid curse of dimensionality)
reducer_cluster = umap.UMAP(n_components=50, n_neighbors=15, min_dist=0.0,
                            metric="cosine", random_state=42, verbose=False)
emb_cluster = reducer_cluster.fit_transform(emb_f)

clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5,
                             metric="euclidean", cluster_selection_method="eom")
q_labels = clusterer.fit_predict(emb_cluster)

# 2D UMAP for visualization only
reducer_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                       metric="cosine", random_state=42, verbose=False)
emb_2d = reducer_2d.fit_transform(emb_f)

# ── Post-hoc merge: combine clusters with centroid cosine > 0.95 ──
from collections import Counter
unique_labels = sorted(set(q_labels) - {-1})
centroids = np.array([emb_f[q_labels == cl].mean(axis=0) for cl in unique_labels])
# Normalize centroids
centroids_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
cos_matrix = centroids_n @ centroids_n.T

# Union-Find merge
parent = {cl: cl for cl in unique_labels}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

merge_count = 0
for i in range(len(unique_labels)):
    for j in range(i + 1, len(unique_labels)):
        if cos_matrix[i, j] > 0.90:
            ri, rj = find(unique_labels[i]), find(unique_labels[j])
            if ri != rj:
                parent[rj] = ri
                merge_count += 1

# Relabel
label_map = {}
new_id = 0
for cl in unique_labels:
    root = find(cl)
    if root not in label_map:
        label_map[root] = new_id
        new_id += 1
    label_map[cl] = label_map[root]

q_labels_merged = np.array([label_map.get(find(cl), -1) if cl >= 0 else -1 for cl in q_labels])

# Post-merge quality check: dissolve clusters with low internal cosine similarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
emb_norm = emb_f / (np.linalg.norm(emb_f, axis=1, keepdims=True) + 1e-12)
MIN_INTERNAL_COS = 0.35
dissolved = 0
for cl in set(q_labels_merged):
    if cl == -1:
        continue
    mask_cl = q_labels_merged == cl
    if mask_cl.sum() < 5:
        continue
    cl_emb = emb_norm[mask_cl]
    cos_mat = cl_emb @ cl_emb.T
    upper = cos_mat[np.triu_indices_from(cos_mat, k=1)]
    if upper.mean() < MIN_INTERNAL_COS:
        q_labels_merged[mask_cl] = -1
        dissolved += 1
print(f"  Dissolved {dissolved} low-coherence clusters (internal cos < {MIN_INTERNAL_COS})")
q_labels = q_labels_merged

n_clusters_before = len(unique_labels)
n_clusters = len(set(q_labels)) - (1 if -1 in q_labels else 0)
n_noise = (q_labels == -1).sum()
print(f"  Initial clusters: {n_clusters_before}, merged {merge_count} pairs -> {n_clusters} clusters")
print(f"  Noise: {n_noise} ({n_noise/len(q_labels)*100:.1f}%)")

sizes = Counter(q_labels)
size_vals = [v for k, v in sizes.items() if k >= 0]
print(f"  Cluster sizes: min={min(size_vals)}, median={sorted(size_vals)[len(size_vals)//2]}, max={max(size_vals)}")

# Stage 1 summary
print("  Building stage 1 summary...")
s1_rows = []
for cl in sorted(sizes.keys()):
    if cl == -1:
        label = "noise"
    else:
        label = f"qc_{cl}"
    mask = q_labels == cl
    n = mask.sum()
    ret = y_pri_f[mask].mean()
    ret2 = y_ret_f[mask].mean()

    # Sample questions
    cl_cids = cids_f[mask]
    samples = np.random.RandomState(42).choice(len(cl_cids), min(5, len(cl_cids)), replace=False)
    sample_qs = " ||| ".join([text_map.get(int(cl_cids[i]), {}).get("question", "") for i in samples])

    s1_rows.append({
        "question_cluster": cl, "label": label, "n": n,
        "retention_14d": round(ret, 4), "retention_2d": round(ret2, 4),
        "sample_questions": sample_qs,
    })

s1_df = pd.DataFrame(s1_rows)
s1_df.to_csv(OUT / "stage1_question_clusters.csv", index=False)
print(f"  Saved stage1_question_clusters.csv ({len(s1_df)} clusters)")

# ══════════════════════════════════════════════════════════════
# STAGE 2: Within each question cluster, sub-cluster by mentee
# ══════════════════════════════════════════════════════════════
print("\n[Stage 2] Within-cluster mentee sub-grouping...")

# Use mentee editing history features (34 dims) for stage 2
E_cols = list(D["E_cols"])
scaler_e = StandardScaler()
# ══════════════════════════════════════════════════════════════
# Stage 2: Within each question cluster, sub-cluster by mentee,
#           then sort by active_days_30d (desc) within each cell
# ══════════════════════════════════════════════════════════════
print("\n[Stage 2] Mentee sub-clustering + sorting by active_days_30d...")
from sklearn.cluster import KMeans

E_cols = list(D["E_cols"])
scaler_e = StandardScaler()
X_E_scaled = scaler_e.fit_transform(X_E_f)

MIN_FOR_SUBCLUSTER = 40

detail_rows = []
for cl in sorted(sizes.keys()):
    mask = q_labels == cl
    n_cl = mask.sum()
    cl_label = "noise" if cl == -1 else f"qc_{cl}"

    cl_cids = cids_f[mask]
    cl_E = X_E_scaled[mask]
    cl_y_pri = y_pri_f[mask]
    cl_y_ret = y_ret_f[mask]
    cl_y_a14 = y_a14_f[mask]
    cl_y_a30 = y_a30_f[mask]
    cl_R = R_f[mask]
    cl_emb2d = emb_2d[mask]

    # Mentee sub-clustering
    if n_cl >= MIN_FOR_SUBCLUSTER:
        n_sub = min(max(2, n_cl // 20), 5)
        km = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
        m_labels = km.fit_predict(cl_E)
    else:
        m_labels = np.zeros(n_cl, dtype=int)

    # Sort by (mentee_subgroup asc, active_days_14d desc)
    order = np.lexsort((-cl_y_a14, m_labels))

    for i in order:
        cid = int(cl_cids[i])
        txt = text_map.get(cid, {"question": "", "reply": ""})
        row = {
            "conversation_id": cid,
            "question_cluster": cl,
            "qc_label": cl_label,
            "mentee_subgroup": int(m_labels[i]),
            "cell": f"{cl_label}_m{m_labels[i]}",
            "active_days_14d": float(cl_y_a14[i]),
            "active_days_30d": float(cl_y_a30[i]),
            "retention_14d": int(cl_y_pri[i]),
            "retention_2d": int(cl_y_ret[i]),
            "question_text": txt["question"],
            "reply_text": txt["reply"],
            "umap_1": round(float(cl_emb2d[i, 0]), 4),
            "umap_2": round(float(cl_emb2d[i, 1]), 4),
        }
        for fi, fname in enumerate(REPLY_COLS):
            row[fname] = round(float(cl_R[i, fi]), 4)
        detail_rows.append(row)

detail_df = pd.DataFrame(detail_rows)
detail_df.to_csv(OUT / "cluster_detail.csv", index=False)
print(f"  Saved cluster_detail.csv ({len(detail_df)} rows)")

# ── Summary stats ─────────────────────────────────────────────
print(f"\n  Question clusters (excl noise): {n_clusters}")
print(f"  Observations in noise: {n_noise}")

# Show top 10 largest question clusters
print("\n  Top 15 question clusters by size:")
top = s1_df[s1_df["question_cluster"] >= 0].nlargest(15, "n")
for _, r in top.iterrows():
    print(f"    {r['label']:>8s}: n={r['n']:>5d}, ret={r['retention_14d']:.3f}  {r['sample_questions'][:80]}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"  s16 COMPLETE ({elapsed:.0f}s)")
print(f"  Output: {OUT}")
print(f"{'=' * 60}")
