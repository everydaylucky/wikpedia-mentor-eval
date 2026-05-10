#!/usr/bin/env python3
"""
s10_3_agreement.py — Human vs LLM (v2) agreement analysis.

Computes Cohen's κ and confusion matrices for:
  1. Yubo vs Rota (human-human, Iteration 2)
  2. Yubo vs DS-v2 (human-LLM)
  3. Rota vs DS-v2 (human-LLM)

Reports separately for:
  - Full set (n=241)
  - Test set (last 121, unseen during prompt tuning) — primary metric
  - Validation set (first 120) — supplementary

Also compares old DS (v1, from golden standard file) vs new DS (v2) to show
annotation quality change.

Input:
  irr_question_annotation_rota4.csv   — Rota Iteration 2 labels
  irr_question_annotation_yubo3.xlsx  — Yubo Iteration 2 labels
  irr_sample_200.csv                  — conversation_id mapping
  irr_golden_standard_llm_difference.xlsx — old DS-v1 labels
  corpus_annotations_v2.jsonl         — new DS-v2 labels

Output:
  Printed tables (matching note.md format)
  data/s10/agreement_v2.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

IRR_DIR = Path("/Users/Shared/baiduyun/00 Code/0Wiki/2026-4/2026-4-24")
BASE = Path(__file__).resolve().parent
V2_FILE = BASE / "data" / "s10" / "corpus_annotations_v2.jsonl"
OUT_DIR = BASE / "data" / "s10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["Q0", "Q2", "Q3", "Q4", "Q5"]
DIM_NAMES = {"Q0": "Q1 Substantive", "Q2": "Q2 Referent", "Q3": "Q3 Appraisal",
             "Q4": "Q4 Normative", "Q5": "Q5 Own Work"}

SEP = "=" * 70


def cohens_kappa(a, b):
    """Cohen's κ for two binary arrays (Y/N strings)."""
    n = len(a)
    assert n == len(b)
    both_y = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "Y")
    both_n = sum(1 for i in range(n) if a[i] == "N" and b[i] == "N")
    a_y_b_n = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "N")
    a_n_b_y = sum(1 for i in range(n) if a[i] == "N" and b[i] == "Y")

    po = (both_y + both_n) / n
    pa = ((both_y + a_y_b_n) * (both_y + a_n_b_y) +
          (a_y_b_n + both_n) * (a_n_b_y + both_n)) / (n * n)
    if pa == 1.0:
        return 1.0
    kappa = (po - pa) / (1 - pa)
    return kappa


def kappa_level(k):
    if k >= 0.81:
        return "Almost perfect"
    if k >= 0.61:
        return "Substantial"
    if k >= 0.41:
        return "Moderate"
    if k >= 0.21:
        return "Fair"
    return "Slight"


def confusion(a, b):
    """Return (both_y, both_n, a_y_b_n, a_n_b_y)."""
    n = len(a)
    both_y = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "Y")
    both_n = sum(1 for i in range(n) if a[i] == "N" and b[i] == "N")
    a_y_b_n = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "N")
    a_n_b_y = sum(1 for i in range(n) if a[i] == "N" and b[i] == "Y")
    return both_y, both_n, a_y_b_n, a_n_b_y


def print_agreement_table(title, pairs, n):
    """Print a formatted agreement table.
    pairs: dict of dim -> (a_labels, b_labels, a_name, b_name)
    """
    print(f"\n  {title} (n={n})")
    print(f"  {'Dim':<6s} {'κ':>7s} {'Level':<16s} {'Both Y':>7s} {'Both N':>7s} "
          f"{'A=Y/B=N':>8s} {'A=N/B=Y':>8s} {'Agree%':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*16} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for d in DIMS:
        a, b, a_name, b_name = pairs[d]
        k = cohens_kappa(a, b)
        by, bn, abn, anb = confusion(a, b)
        agree = (by + bn) / len(a) * 100
        print(f"  {d:<6s} {k:>7.4f} {kappa_level(k):<16s} {by:>7d} {bn:>7d} "
              f"{abn:>8d} {anb:>8d} {agree:>6.1f}%")


def main():
    print(f"\n{SEP}")
    print("  s10_3: HUMAN vs LLM (v2) AGREEMENT ANALYSIS")
    print(SEP)

    # ── Load data ─────────────────────────────────────────────────────
    rota = pd.read_csv(IRR_DIR / "irr_question_annotation_rota4.csv")
    yubo = pd.read_excel(IRR_DIR / "irr_question_annotation_yubo3.xlsx")
    sample = pd.read_csv(IRR_DIR / "irr_sample_200.csv")
    gs = pd.read_excel(IRR_DIR / "irr_golden_standard_llm_difference.xlsx")

    # Load v2 LLM annotations
    v2 = {}
    with open(V2_FILE) as f:
        for line in f:
            d = json.loads(line)
            v2[d["cid"]] = d

    n = len(rota)
    assert n == 241 and len(yubo) == 241 and len(sample) == 241 and len(gs) == 241
    print(f"  Loaded {n} IRR questions")

    # ── Build label arrays ────────────────────────────────────────────
    # Yubo and Rota use Q0 in their files (= Q1 substantive in the paper)
    yubo_labels = {d: [] for d in DIMS}
    rota_labels = {d: [] for d in DIMS}
    ds_v2_labels = {d: [] for d in DIMS}
    ds_v1_labels = {d: [] for d in DIMS}

    for i in range(n):
        cid = sample.iloc[i]["conversation_id"]

        for d in DIMS:
            # Yubo
            val_y = str(yubo.iloc[i][d]).strip().upper()
            yubo_labels[d].append(val_y if val_y in ("Y", "N") else "?")

            # Rota
            val_r = str(rota.iloc[i][d]).strip().upper()
            rota_labels[d].append(val_r if val_r in ("Y", "N") else "?")

            # DS v2
            v2_entry = v2[cid]
            ds_v2_labels[d].append(v2_entry[d])

            # DS v1 (old)
            val_old = str(gs.iloc[i][f"deepseek-v4-flash_{d}"]).strip().upper()
            ds_v1_labels[d].append(val_old if val_old in ("Y", "N") else "?")

    # Check for missing values
    for d in DIMS:
        n_missing_y = yubo_labels[d].count("?")
        n_missing_r = rota_labels[d].count("?")
        if n_missing_y > 0 or n_missing_r > 0:
            print(f"  WARNING: {d} has {n_missing_y} missing Yubo, {n_missing_r} missing Rota")

    # ── Split indices ─────────────────────────────────────────────────
    val_idx = list(range(0, 120))
    test_idx = list(range(120, 241))
    all_idx = list(range(0, 241))
    print(f"  Validation set: {len(val_idx)}, Test set: {len(test_idx)}, Full: {len(all_idx)}")

    def subset(labels, indices):
        return [labels[i] for i in indices]

    # ── Positive counts ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  POSITIVE (Y) COUNTS")
    print(SEP)
    print(f"\n  {'Dim':<6s} {'Annotator':<12s} {'Val(120)':>9s} {'Test(121)':>10s} {'Total':>7s}")
    print(f"  {'-'*6} {'-'*12} {'-'*9} {'-'*10} {'-'*7}")
    for d in DIMS:
        for name, labels in [("Yubo", yubo_labels[d]), ("Rota", rota_labels[d]), ("DS-v2", ds_v2_labels[d])]:
            val_y = sum(1 for i in val_idx if labels[i] == "Y")
            test_y = sum(1 for i in test_idx if labels[i] == "Y")
            total_y = val_y + test_y
            print(f"  {d:<6s} {name:<12s} {val_y:>9d} {test_y:>10d} {total_y:>7d}")

    # ══════════════════════════════════════════════════════════════════
    # HUMAN-HUMAN AGREEMENT (Yubo vs Rota)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  HUMAN-HUMAN AGREEMENT (Yubo vs Rota, Iteration 2)")
    print(SEP)

    for set_name, idx_set in [("Full set", all_idx), ("Test set", test_idx), ("Validation set", val_idx)]:
        pairs = {}
        for d in DIMS:
            a = subset(yubo_labels[d], idx_set)
            b = subset(rota_labels[d], idx_set)
            pairs[d] = (a, b, "Yubo", "Rota")
        print_agreement_table(f"Yubo vs Rota — {set_name}", pairs, len(idx_set))

    # ══════════════════════════════════════════════════════════════════
    # HUMAN-LLM AGREEMENT (v2)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  HUMAN vs LLM (DS-v2) AGREEMENT")
    print(SEP)

    for set_name, idx_set in [("Test set (PRIMARY)", test_idx), ("Full set", all_idx), ("Validation set", val_idx)]:
        # Yubo vs DS-v2
        pairs_yd = {}
        for d in DIMS:
            a = subset(yubo_labels[d], idx_set)
            b = subset(ds_v2_labels[d], idx_set)
            pairs_yd[d] = (a, b, "Yubo", "DS-v2")
        print_agreement_table(f"Yubo vs DS-v2 — {set_name}", pairs_yd, len(idx_set))

        # Rota vs DS-v2
        pairs_rd = {}
        for d in DIMS:
            a = subset(rota_labels[d], idx_set)
            b = subset(ds_v2_labels[d], idx_set)
            pairs_rd[d] = (a, b, "Rota", "DS-v2")
        print_agreement_table(f"Rota vs DS-v2 — {set_name}", pairs_rd, len(idx_set))

    # ══════════════════════════════════════════════════════════════════
    # v1 vs v2 COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  DS-v1 vs DS-v2 COMPARISON (label changes on 241 IRR questions)")
    print(SEP)

    print(f"\n  {'Dim':<6s} {'Same':>6s} {'v1=Y→v2=N':>10s} {'v1=N→v2=Y':>10s} {'Net ΔY':>8s}")
    print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    for d in DIMS:
        same = sum(1 for i in range(n) if ds_v1_labels[d][i] == ds_v2_labels[d][i])
        y_to_n = sum(1 for i in range(n) if ds_v1_labels[d][i] == "Y" and ds_v2_labels[d][i] == "N")
        n_to_y = sum(1 for i in range(n) if ds_v1_labels[d][i] == "N" and ds_v2_labels[d][i] == "Y")
        net = n_to_y - y_to_n
        print(f"  {d:<6s} {same:>6d} {y_to_n:>10d} {n_to_y:>10d} {net:>+8d}")

    # Compare v1 vs v2 agreement with humans
    print(f"\n  κ comparison (test set, n=121):")
    print(f"  {'Dim':<6s} {'Yubo-v1':>9s} {'Yubo-v2':>9s} {'Δ':>7s} {'Rota-v1':>9s} {'Rota-v2':>9s} {'Δ':>7s}")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*9} {'-'*7}")
    for d in DIMS:
        y_sub = subset(yubo_labels[d], test_idx)
        r_sub = subset(rota_labels[d], test_idx)
        v1_sub = subset(ds_v1_labels[d], test_idx)
        v2_sub = subset(ds_v2_labels[d], test_idx)

        k_yv1 = cohens_kappa(y_sub, v1_sub)
        k_yv2 = cohens_kappa(y_sub, v2_sub)
        k_rv1 = cohens_kappa(r_sub, v1_sub)
        k_rv2 = cohens_kappa(r_sub, v2_sub)
        print(f"  {d:<6s} {k_yv1:>9.4f} {k_yv2:>9.4f} {k_yv2 - k_yv1:>+7.4f} "
              f"{k_rv1:>9.4f} {k_rv2:>9.4f} {k_rv2 - k_rv1:>+7.4f}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE for paper
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  SUMMARY TABLE (test set, n=121) — for paper")
    print(SEP)

    rows = []
    print(f"\n  {'Dim':<6s} {'Yubo-Rota':>10s} {'Yubo-DSv2':>10s} {'Rota-DSv2':>10s}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for d in DIMS:
        y = subset(yubo_labels[d], test_idx)
        r = subset(rota_labels[d], test_idx)
        v = subset(ds_v2_labels[d], test_idx)
        k_yr = cohens_kappa(y, r)
        k_yv = cohens_kappa(y, v)
        k_rv = cohens_kappa(r, v)
        print(f"  {d:<6s} {k_yr:>10.4f} {k_yv:>10.4f} {k_rv:>10.4f}")
        rows.append({
            "Dimension": d, "Name": DIM_NAMES[d],
            "Yubo_Rota": round(k_yr, 4),
            "Yubo_DSv2": round(k_yv, 4),
            "Rota_DSv2": round(k_rv, 4),
            "Set": "test",
        })

    # Also save full set
    for d in DIMS:
        y = yubo_labels[d]
        r = rota_labels[d]
        v = ds_v2_labels[d]
        rows.append({
            "Dimension": d, "Name": DIM_NAMES[d],
            "Yubo_Rota": round(cohens_kappa(y, r), 4),
            "Yubo_DSv2": round(cohens_kappa(y, v), 4),
            "Rota_DSv2": round(cohens_kappa(r, v), 4),
            "Set": "full",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "agreement_v2.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'agreement_v2.csv'}")
    print(SEP)


if __name__ == "__main__":
    main()
