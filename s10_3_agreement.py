#!/usr/bin/env python3
"""
s10_3_agreement.py -- Human-human and human-LLM agreement analysis.

Computes Cohen's kappa, confusion matrices, and prevalence for:
  1. Annotator 1 (A1) vs Annotator 2 (A2) -- human-human, Iteration 2
  2. A1 vs LLM (DeepSeek-v4-Flash, non-thinking, with few-shot)
  3. A2 vs LLM

The LLM was run with the codebook prompt + 120 few-shot examples drawn from
A1's validation set labels (idx 1-120). A1 was the primary designer of the
codebook and prompt, which may explain the higher A1-LLM agreement compared
to A2-LLM. The test set (idx 121-241, last 121 questions) was held out during
prompt tuning and few-shot example selection.

Reports separately for:
  - Test set (last 121, held out) -- primary metric
  - Full set (n=241) -- supplementary / appendix

Input (all under data/s10/):
  irr_annotation_a2.csv              -- Annotator 2 labels
  irr_annotation_a1.xlsx             -- Annotator 1 labels
  llm_fewshot_dsv4flash_nt.jsonl     -- LLM few-shot annotations (241 rows)

Output:
  Printed tables
  data/s10/agreement_fewshot.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "s10"

DIMS = ["Q0", "Q2", "Q3", "Q4", "Q5"]
DIM_NAMES = {
    "Q0": "Q1 Substantive", "Q2": "Q2 Referent", "Q3": "Q3 Appraisal",
    "Q4": "Q4 Normative", "Q5": "Q5 Own Work",
}

SEP = "=" * 70


def to_yn(val):
    s = str(val).strip().upper()
    return s if s in ("Y", "N") else "N"


def cohens_kappa(a, b):
    n = len(a)
    assert n == len(b)
    by = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "Y")
    bn = sum(1 for i in range(n) if a[i] == "N" and b[i] == "N")
    abn = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "N")
    anb = sum(1 for i in range(n) if a[i] == "N" and b[i] == "Y")
    po = (by + bn) / n
    pe = ((by + abn) * (by + anb) + (abn + bn) * (anb + bn)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def kappa_level(k):
    if k >= 0.81: return "Almost perfect"
    if k >= 0.61: return "Substantial"
    if k >= 0.41: return "Moderate"
    if k >= 0.21: return "Fair"
    return "Slight"


def confusion(a, b):
    n = len(a)
    by = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "Y")
    bn = sum(1 for i in range(n) if a[i] == "N" and b[i] == "N")
    abn = sum(1 for i in range(n) if a[i] == "Y" and b[i] == "N")
    anb = sum(1 for i in range(n) if a[i] == "N" and b[i] == "Y")
    return by, bn, abn, anb


def print_table(title, pairs, n):
    print(f"\n  {title} (n={n})")
    print(f"  {'Dim':<6s} {'k':>7s} {'Level':<16s} {'Both Y':>7s} {'Both N':>7s} "
          f"{'A=Y/B=N':>8s} {'A=N/B=Y':>8s} {'Agree%':>7s}")
    print(f"  {'-'*6} {'-'*7} {'-'*16} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for d in DIMS:
        a, b = pairs[d]
        k = cohens_kappa(a, b)
        by, bn, abn, anb = confusion(a, b)
        agree = (by + bn) / len(a) * 100
        print(f"  {d:<6s} {k:>7.3f} {kappa_level(k):<16s} {by:>7d} {bn:>7d} "
              f"{abn:>8d} {anb:>8d} {agree:>6.1f}%")


def print_prevalence(title, label_sets, n):
    """Print prevalence (Y-rate) per annotator per dimension."""
    names = [name for name, _ in label_sets]
    print(f"\n  {title} (n={n})")
    header = f"  {'Dim':<6s}" + "".join(f" {name:>10s}" for name in names)
    print(header)
    print(f"  {'-'*6}" + "".join(f" {'-'*10}" for _ in names))
    for d in DIMS:
        row = f"  {d:<6s}"
        for name, labels in label_sets:
            y_count = sum(1 for v in labels[d] if v == "Y")
            pct = y_count / n * 100
            row += f" {y_count:>4d} ({pct:>4.1f}%)"
        print(row)


def print_bias(title, pairs_list, n):
    """Print disagreement direction (bias) for each pair."""
    print(f"\n  {title} (n={n})")
    print(f"  {'Dim':<6s} {'Pair':<12s} {'A>B (A=Y,B=N)':>14s} {'B>A (A=N,B=Y)':>14s} {'Net bias':>10s}")
    print(f"  {'-'*6} {'-'*12} {'-'*14} {'-'*14} {'-'*10}")
    for d in DIMS:
        for pair_name, a, b in pairs_list:
            _, _, abn, anb = confusion(a[d], b[d])
            net = abn - anb
            direction = ""
            if net > 0:
                direction = f"A +{net}"
            elif net < 0:
                direction = f"B +{-net}"
            else:
                direction = "even"
            print(f"  {d:<6s} {pair_name:<12s} {abn:>14d} {anb:>14d} {direction:>10s}")


def main():
    print(f"\n{SEP}")
    print("  s10_3: AGREEMENT ANALYSIS (A1, A2, LLM=DS-v4-Flash-NT few-shot)")
    print(SEP)

    # -- Load data (all from data/s10/) --
    a1_df = pd.read_excel(DATA_DIR / "irr_annotation_a1.xlsx")
    a2_df = pd.read_csv(DATA_DIR / "irr_annotation_a2.csv")
    llm = {}
    with open(DATA_DIR / "llm_fewshot_dsv4flash_nt.jsonl") as f:
        for line in f:
            d = json.loads(line)
            llm[d["idx"]] = d

    n = len(a1_df)
    assert n == 241 and len(a2_df) == 241 and len(llm) == 241
    print(f"  Loaded {n} IRR questions")

    # -- Build label arrays --
    a1_labels = {d: [] for d in DIMS}
    a2_labels = {d: [] for d in DIMS}
    llm_labels = {d: [] for d in DIMS}

    for i in range(n):
        idx = i + 1
        for d in DIMS:
            a1_labels[d].append(to_yn(a1_df.iloc[i][d]))
            a2_labels[d].append(to_yn(a2_df.iloc[i][d]))
            llm_labels[d].append(llm[idx][d])

    # -- Split --
    # First 20 are calibration (co-labeled, not independent) -- exclude from kappa
    cal_idx = list(range(0, 20))
    val_idx = list(range(0, 120))
    test_idx = list(range(120, 241))
    indep_idx = list(range(20, 241))  # n=221, excludes 20 calibration
    all_idx = list(range(0, 241))
    print(f"  Calibration: {len(cal_idx)}, Test: {len(test_idx)}, Independent: {len(indep_idx)}, Full: {len(all_idx)}")

    def sub(labels, indices):
        return {d: [labels[d][i] for i in indices] for d in DIMS}

    # ══════════════════════════════════════════════════════════════════
    # PREVALENCE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  PREVALENCE (Y-rate per annotator)")
    print(SEP)

    for set_name, idx in [("Test set", test_idx), ("Independent (excl cal)", indep_idx)]:
        a1_sub = sub(a1_labels, idx)
        a2_sub = sub(a2_labels, idx)
        llm_sub = sub(llm_labels, idx)
        print_prevalence(
            f"Prevalence -- {set_name}",
            [("A1", a1_sub), ("A2", a2_sub), ("LLM", llm_sub)],
            len(idx),
        )

    # ══════════════════════════════════════════════════════════════════
    # HUMAN-HUMAN AGREEMENT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  HUMAN-HUMAN AGREEMENT (A1 vs A2, excl 20 calibration)")
    print(SEP)

    for set_name, idx in [("Independent (excl cal)", indep_idx), ("Test set", test_idx)]:
        s = sub(a1_labels, idx)
        t = sub(a2_labels, idx)
        pairs = {d: (s[d], t[d]) for d in DIMS}
        print_table(f"A1 vs A2 -- {set_name}", pairs, len(idx))

    # ══════════════════════════════════════════════════════════════════
    # HUMAN-LLM AGREEMENT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  HUMAN vs LLM AGREEMENT (DS-v4-Flash-NT, few-shot from A1 val set)")
    print(SEP)

    for set_name, idx in [("Test set (PRIMARY)", test_idx), ("Independent (excl cal)", indep_idx)]:
        a1_sub = sub(a1_labels, idx)
        a2_sub = sub(a2_labels, idx)
        llm_sub = sub(llm_labels, idx)

        pairs_a1 = {d: (a1_sub[d], llm_sub[d]) for d in DIMS}
        print_table(f"A1 vs LLM -- {set_name}", pairs_a1, len(idx))
        pairs_a2 = {d: (a2_sub[d], llm_sub[d]) for d in DIMS}
        print_table(f"A2 vs LLM -- {set_name}", pairs_a2, len(idx))

    # ══════════════════════════════════════════════════════════════════
    # DISAGREEMENT DIRECTION (BIAS)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  DISAGREEMENT DIRECTION (who labels more Y)")
    print(SEP)

    for set_name, idx in [("Test set", test_idx), ("Independent (excl cal)", indep_idx)]:
        a1_sub = sub(a1_labels, idx)
        a2_sub = sub(a2_labels, idx)
        llm_sub = sub(llm_labels, idx)
        print_bias(
            f"Bias -- {set_name}",
            [("A1 vs A2", a1_sub, a2_sub),
             ("A1 vs LLM", a1_sub, llm_sub),
             ("A2 vs LLM", a2_sub, llm_sub)],
            len(idx),
        )

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE for paper
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  SUMMARY TABLE (for paper)")
    print(SEP)

    rows = []
    for set_name, idx in [("test", test_idx), ("independent", indep_idx)]:
        a1_sub = sub(a1_labels, idx)
        a2_sub = sub(a2_labels, idx)
        llm_sub = sub(llm_labels, idx)

        print(f"\n  {set_name.upper()} SET (n={len(idx)})")
        print(f"  {'Dim':<6s} {'A1-A2':>8s} {'A1-LLM':>8s} {'A2-LLM':>8s}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
        for d in DIMS:
            k_12 = cohens_kappa(a1_sub[d], a2_sub[d])
            k_1l = cohens_kappa(a1_sub[d], llm_sub[d])
            k_2l = cohens_kappa(a2_sub[d], llm_sub[d])
            print(f"  {d:<6s} {k_12:>8.3f} {k_1l:>8.3f} {k_2l:>8.3f}")
            rows.append({
                "Dimension": d, "Name": DIM_NAMES[d], "Set": set_name,
                "A1_A2": round(k_12, 3), "A1_LLM": round(k_1l, 3),
                "A2_LLM": round(k_2l, 3),
            })

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "agreement_fewshot.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(SEP)


if __name__ == "__main__":
    main()
