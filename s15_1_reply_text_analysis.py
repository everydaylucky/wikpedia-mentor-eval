#!/usr/bin/env python3
"""
s15_1_reply_text_analysis.py — Mentor reply text feature extraction & subgroup comparison.

Extracts interpretable text features from mentor replies (for treated group only),
then compares feature distributions across Morrison question-type subgroups.

Feature categories:
  1. Surface (from s11): word count, sentences, paragraphs, avg word length, etc.
  2. Wiki structure (from s11): wikilinks, policy/help links, list, topic mentions
  3. Perspective API — Wikipedia-trained core attributes only (from s11)
  4. Pronouns (new): you/your, I/me/my, we/our counts
  5. Sentiment (new): VADER, TextBlob polarity/subjectivity
  6. Politeness (new): 21 regex-based strategies (same as question-side in s12)
  7. Readability (new): Flesch-Kincaid grade, avg syllables per word
  8. Imperative sentences (new): SpaCy POS tagging
  9. Step-by-step structure (new): regex for numbered/ordered lists
  10. Responsiveness (new): reply lag hours, reply/question word ratio

Input:
  data/s7/s7_conversations_cleaned.jsonl  — reply text + timestamps
  data/s11/s11_features.jsonl             — existing reply features
  data/s10/corpus_annotations_v2.jsonl    — question type labels
  data/s12/psm_data/psm_dataset.npz      — sample filter (who's in analytic sample)

Output:
  data/s15/reply_features.csv
  data/s15/subgroup_comparison.csv
  data/s15/figures/feature_by_subgroup.pdf
"""
import json, os, re, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
S7_FILE = BASE / "data" / "s7" / "s7_conversations_cleaned.jsonl"
S11_FILE = BASE / "data" / "s11" / "s11_features.jsonl"
V2_FILE = BASE / "data" / "s10" / "corpus_annotations_v2.jsonl"
PSM_FILE = BASE / "data" / "s12" / "psm_data" / "psm_dataset.npz"
OUT_DIR = BASE / "data" / "s15"
OUT_FIG = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEP = "=" * 70

# ── Dependencies ─────────────────────────────────────────────────────────────

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("WARNING: vaderSentiment not installed → VADER features = 0")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("WARNING: textblob not installed → TextBlob features = 0")

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    HAS_SPACY = True
except (ImportError, OSError):
    HAS_SPACY = False
    print("WARNING: spacy/en_core_web_sm not available → imperative detection = 0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction functions
# ══════════════════════════════════════════════════════════════════════════════

def count_pronouns(text):
    """Count pronoun categories in text."""
    if not text:
        return 0, 0, 0
    lo = text.lower()
    words = re.findall(r"\b[a-z']+\b", lo)
    you_set = {"you", "your", "yours", "yourself", "yourselves"}
    i_set = {"i", "me", "my", "mine", "myself"}
    we_set = {"we", "us", "our", "ours", "ourselves"}
    n_you = sum(1 for w in words if w in you_set)
    n_i = sum(1 for w in words if w in i_set)
    n_we = sum(1 for w in words if w in we_set)
    return n_you, n_i, n_we


def extract_vader_reply(text):
    if not HAS_VADER or not text:
        return 0.0, 0.0, 0.0, 0.0
    vs = _vader.polarity_scores(text)
    return vs["neg"], vs["neu"], vs["pos"], vs["compound"]


def extract_textblob_reply(text):
    if not HAS_TEXTBLOB or not text:
        return 0.0, 0.0
    tb = TextBlob(text)
    return tb.sentiment.polarity, tb.sentiment.subjectivity


def extract_politeness_reply(text):
    """21 regex-based politeness strategies (same as s12 question-side)."""
    if not text:
        return {f"r_polite_{i}": 0 for i in range(21)}
    lo = text.lower().strip()
    names = [
        "please", "please_start", "hedge_word", "indirect_btw", "hedges_think",
        "factuality", "deference", "gratitude", "apologizing",
        "1st_person_pl", "1st_person", "1st_person_start",
        "2nd_person", "2nd_person_start", "indirect_greeting",
        "direct_question", "direct_start",
        "has_positive", "has_negative", "subjunctive", "indicative",
    ]
    vals = [
        int("please" in lo),
        int(lo.startswith("please")),
        int(bool(re.search(r"\b(kind of|sort of|maybe|perhaps|possibly|probably|somewhat|rather)\b", lo))),
        int(bool(re.search(r"\b(by the way|btw|incidentally)\b", lo))),
        int(bool(re.search(r"\b(i think|i believe|i suppose|i guess|i mean|it seems)\b", lo))),
        int(bool(re.search(r"\b(actually|in fact|just|really)\b", lo))),
        int(bool(re.search(r"\b(great|nice|good|excellent|interesting|wonderful)\b", lo))),
        int(bool(re.search(r"\b(thank|thanks|grateful|appreciate)\b", lo))),
        int(bool(re.search(r"\b(sorry|apologi[sz]e|excuse me|pardon)\b", lo))),
        int(bool(re.search(r"\b(we|us|our|ours|ourselves)\b", lo))),
        int(bool(re.search(r"\b(i|my|me|mine|myself)\b", lo))),
        int(bool(re.match(r"(i|my)\b", lo))),
        int(bool(re.search(r"\b(you|your|yours|yourself)\b", lo))),
        int(bool(re.match(r"(you|your)\b", lo))),
        int(bool(re.match(r"(hi|hello|hey|dear|greetings)\b", lo))),
        int("?" in text),
        int(bool(re.match(r"(can|could|would|will|do|does|did|is|are|was|were|have|has|had|should)\b", lo))),
        int(bool(re.search(r"\b(good|great|nice|love|wonderful|excellent|amazing|awesome|happy|glad)\b", lo))),
        int(bool(re.search(r"\b(bad|terrible|awful|horrible|hate|worst|ugly|stupid|annoying|frustrated|angry)\b", lo))),
        int(bool(re.search(r"\b(could|would|should|might|may)\b", lo))),
        int(bool(re.search(r"\b(will|going to|have to|need to|must|shall)\b", lo))),
    ]
    return {f"r_polite_{n}": v for n, v in zip(names, vals)}


def count_syllables(word):
    """Rough syllable count for English words."""
    word = word.lower().strip()
    if len(word) <= 2:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_features(text):
    """Flesch-Kincaid grade level and avg syllables per word."""
    if not text:
        return 0.0, 0.0
    words = re.findall(r"[a-zA-Z]+", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    n_words = len(words)
    n_sent = max(len(sentences), 1)
    if n_words == 0:
        return 0.0, 0.0
    total_syl = sum(count_syllables(w) for w in words)
    avg_syl = total_syl / n_words
    fk_grade = 0.39 * (n_words / n_sent) + 11.8 * avg_syl - 15.59
    return fk_grade, avg_syl


def count_imperatives_spacy(text):
    """Count imperative sentences using SpaCy POS tagging."""
    if not HAS_SPACY or not text:
        return 0
    doc = _nlp(text)
    count = 0
    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space]
        if not tokens:
            continue
        first = tokens[0]
        if first.pos_ == "VERB" and first.tag_ == "VB":
            count += 1
    return count


def count_step_by_step(text):
    """Count numbered steps (1. 2. 3. or Step 1, Step 2, etc.)."""
    if not text:
        return 0, False
    numbered = re.findall(r"(?:^|\n)\s*(\d+)[.\)]\s", text)
    step_kw = re.findall(r"(?i)\bstep\s+\d+", text)
    n_steps = max(len(numbered), len(step_kw))
    has_steps = n_steps >= 2
    return n_steps, has_steps


def compute_reply_lag_hours(q_ts_str, r_ts_str):
    """Compute reply lag in hours from timestamp strings."""
    if not q_ts_str or not r_ts_str:
        return np.nan
    try:
        fmt1 = "%Y-%m-%dT%H:%M:%S"
        fmt2 = "%Y-%m-%dT%H:%M:%SZ"
        for fmt in [fmt1, fmt2]:
            try:
                q_ts = datetime.strptime(q_ts_str[:19], fmt1)
                r_ts = datetime.strptime(r_ts_str[:19], fmt1)
                delta = (r_ts - q_ts).total_seconds() / 3600
                return max(0, delta)
            except ValueError:
                continue
        return np.nan
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"\n{SEP}")
    print("  s15: MENTOR REPLY TEXT FEATURE EXTRACTION & SUBGROUP ANALYSIS")
    print(SEP)

    # ── Load s7 (reply text + timestamps) ────────────────────────────────
    print("\n  Loading s7 conversations...")
    s7 = {}
    with open(S7_FILE) as f:
        for line in f:
            d = json.loads(line)
            s7[d["conversation_id"]] = d
    print(f"  s7: {len(s7):,} conversations")

    # ── Load s11 (existing features) ─────────────────────────────────────
    print("  Loading s11 features...")
    s11 = {}
    with open(S11_FILE) as f:
        for line in f:
            d = json.loads(line)
            s11[d["conversation_id"]] = d
    print(f"  s11: {len(s11):,} records")

    # ── Load question type annotations ───────────────────────────────────
    print("  Loading v2 annotations...")
    v2 = {}
    with open(V2_FILE) as f:
        for line in f:
            d = json.loads(line)
            v2[d["cid"]] = d
    print(f"  v2: {len(v2):,} annotations")

    # ── Load PSM dataset to get analytic sample ──────────────────────────
    print("  Loading PSM dataset for sample filter...")
    psm = np.load(PSM_FILE, allow_pickle=True)
    y_treat = psm["y_treat"]
    # s11 and PSM share same order — s11 cids ARE the sample
    s11_cids = list(s11.keys())
    treated_cids = set(s11_cids[i] for i in range(len(s11_cids)) if y_treat[i] == 1)
    print(f"  PSM sample: {len(s11_cids):,} total, {len(treated_cids):,} treated")

    # ── Extract features for treated group ───────────────────────────────
    print(f"\n  Extracting new reply features for {len(treated_cids):,} treated conversations...")

    # Perspective API core attributes (Wikipedia-trained only)
    PERSP_CORE = [
        "persp_r_toxicity", "persp_r_severe_toxicity", "persp_r_identity_attack",
        "persp_r_insult", "persp_r_profanity", "persp_r_threat",
    ]

    # Existing s11 reply features to carry forward
    S11_REPLY = [
        "r_words", "r_chars", "r_sentences", "r_avg_word_len", "r_type_token_ratio",
        "r_n_question_marks", "r_n_exclamation", "r_has_question_mark",
        "r_has_greeting", "r_has_thanks", "r_has_apology", "r_has_frustration",
        "r_has_self_intro", "r_has_urgency",
        "r_n_policy", "r_n_help", "r_n_wikilink", "r_n_link", "r_n_draft",
        "r_mentions_deletion", "r_mentions_revert", "r_mentions_notability",
        "r_mentions_copyright", "r_mentions_draft", "r_mentions_protection",
        "r_mentions_conflict",
        "r_n_paragraphs", "r_avg_sentence_len", "r_is_single_sentence", "r_has_list",
    ]

    rows = []
    n_done = 0
    for cid in treated_cids:
        if cid not in s7 or cid not in s11:
            continue
        rec7 = s7[cid]
        rec11 = s11[cid]

        reply_text = rec7.get("reply_clean", "") or ""
        if not reply_text:
            continue

        question_text = rec7.get("question_clean", "") or ""
        q_words = rec7.get("question_words", 0) or len(question_text.split())
        r_words = rec7.get("reply_words", 0) or len(reply_text.split())

        row = {"conversation_id": cid}

        # (1) Existing s11 reply features
        for feat in S11_REPLY:
            row[feat] = rec11.get(feat, 0)

        # (2) Perspective API — core only (Wikipedia-trained)
        for feat in PERSP_CORE:
            row[feat] = rec11.get(feat)

        # (3) Pronouns
        n_you, n_i, n_we = count_pronouns(reply_text)
        row["r_pronoun_you"] = n_you
        row["r_pronoun_i"] = n_i
        row["r_pronoun_we"] = n_we
        row["r_pronoun_you_rate"] = n_you / max(r_words, 1)
        row["r_pronoun_i_rate"] = n_i / max(r_words, 1)
        row["r_pronoun_we_rate"] = n_we / max(r_words, 1)

        # (4) Sentiment — VADER
        neg, neu, pos, compound = extract_vader_reply(reply_text)
        row["r_vader_neg"] = neg
        row["r_vader_neu"] = neu
        row["r_vader_pos"] = pos
        row["r_vader_compound"] = compound

        # (5) Sentiment — TextBlob
        polarity, subjectivity = extract_textblob_reply(reply_text)
        row["r_tb_polarity"] = polarity
        row["r_tb_subjectivity"] = subjectivity

        # (6) Politeness strategies (21 features)
        polite = extract_politeness_reply(reply_text)
        row.update(polite)

        # (7) Readability
        fk_grade, avg_syl = readability_features(reply_text)
        row["r_flesch_kincaid"] = fk_grade
        row["r_avg_syllables"] = avg_syl

        # (8) Imperative sentences
        row["r_n_imperatives"] = count_imperatives_spacy(reply_text)

        # (9) Step-by-step
        n_steps, has_steps = count_step_by_step(reply_text)
        row["r_n_steps"] = n_steps
        row["r_has_steps"] = int(has_steps)

        # (10) Responsiveness
        reply_lag = compute_reply_lag_hours(
            rec7.get("timestamp", ""), rec7.get("reply_timestamp", ""))
        row["r_reply_lag_hours"] = reply_lag
        row["r_reply_q_word_ratio"] = r_words / max(q_words, 1)

        # (11) Actionability composite
        row["r_n_resources"] = (
            row.get("r_n_wikilink", 0) +
            row.get("r_n_policy", 0) +
            row.get("r_n_help", 0) +
            row.get("r_n_link", 0)
        )

        # (12) Question type labels
        ann = v2.get(cid, {})
        row["Q0"] = ann.get("Q0", "N")
        row["Q2"] = ann.get("Q2", "N")
        row["Q3"] = ann.get("Q3", "N")
        row["Q4"] = ann.get("Q4", "N")
        row["Q5"] = ann.get("Q5", "N")

        rows.append(row)
        n_done += 1
        if n_done % 5000 == 0:
            print(f"    {n_done:,} done...")

    df = pd.DataFrame(rows)
    print(f"\n  Extracted features for {len(df):,} treated conversations")
    df.to_csv(OUT_DIR / "reply_features.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'reply_features.csv'}")

    # ══════════════════════════════════════════════════════════════════════
    # SUBGROUP COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  SUBGROUP COMPARISON — Mentor Reply Features by Question Type")
    print(SEP)

    # Define subgroups
    q_sub = df["Q0"] == "Y"
    q_ref = df["Q2"] == "Y"
    q_apr = df["Q3"] == "Y"
    q_nor = df["Q4"] == "Y"
    q_own = df["Q5"] == "Y"

    subgroups = [
        ("Full sample",     pd.Series(True, index=df.index)),
        ("Technical",       q_sub & ~q_ref & ~q_apr & ~q_nor),
        ("Referent (Q2)",   q_ref),
        ("Appraisal (Q3)",  q_apr),
        ("Normative (Q4)",  q_nor),
        ("Non-substantive", df["Q0"] == "N"),
        ("Own work (Q5=Y)", q_own),
        ("No own work (Q5=N)", ~q_own),
    ]

    # Key features to compare
    compare_features = [
        # Surface
        ("r_words", "Reply word count"),
        ("r_sentences", "Reply sentence count"),
        ("r_n_paragraphs", "Reply paragraph count"),
        ("r_avg_word_len", "Avg word length"),
        ("r_type_token_ratio", "Type-token ratio"),
        # Pronouns
        ("r_pronoun_you", "\"You\" count"),
        ("r_pronoun_i", "\"I\" count"),
        ("r_pronoun_we", "\"We\" count"),
        ("r_pronoun_you_rate", "\"You\" rate (per word)"),
        ("r_pronoun_i_rate", "\"I\" rate (per word)"),
        # Sentiment
        ("r_vader_compound", "VADER compound"),
        ("r_tb_polarity", "TextBlob polarity"),
        ("r_tb_subjectivity", "TextBlob subjectivity"),
        # Readability
        ("r_flesch_kincaid", "Flesch-Kincaid grade"),
        ("r_avg_syllables", "Avg syllables/word"),
        # Structure
        ("r_n_imperatives", "Imperative sentences"),
        ("r_n_steps", "Numbered steps"),
        ("r_has_steps", "Has step-by-step (%)"),
        ("r_has_list", "Has list (%)"),
        # Wiki resources
        ("r_n_resources", "Resource links (total)"),
        ("r_n_wikilink", "Wikilinks"),
        ("r_n_policy", "Policy references"),
        ("r_n_help", "Help page references"),
        # Responsiveness
        ("r_reply_lag_hours", "Reply lag (hours)"),
        ("r_reply_q_word_ratio", "Reply/question word ratio"),
        # Politeness
        ("r_polite_gratitude", "Gratitude"),
        ("r_polite_deference", "Deference"),
        ("r_polite_has_positive", "Positive language"),
        ("r_polite_indirect_greeting", "Greeting"),
        # Perspective (core, Wikipedia-trained)
        ("persp_r_toxicity", "Toxicity"),
        ("persp_r_insult", "Insult"),
    ]

    # Build comparison table
    comp_rows = []
    print(f"\n  {'Feature':<30s}", end="")
    for sg_name, _ in subgroups:
        print(f" {sg_name:>14s}", end="")
    print()
    print("  " + "-" * (30 + 14 * len(subgroups)))

    for feat_col, feat_name in compare_features:
        if feat_col not in df.columns:
            continue
        print(f"  {feat_name:<30s}", end="")
        row_data = {"Feature": feat_name, "Feature_col": feat_col}
        for sg_name, sg_mask in subgroups:
            vals = df.loc[sg_mask, feat_col].dropna()
            m = vals.mean()
            row_data[sg_name] = round(m, 4)
            print(f" {m:>14.4f}", end="")
        print()
        comp_rows.append(row_data)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "subgroup_comparison.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'subgroup_comparison.csv'}")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  FIGURES")
    print(SEP)

    sg_names_plot = [s[0] for s in subgroups if s[0] != "Full sample"]
    colors = {
        "Technical": "#1f77b4", "Referent (Q2)": "#ff7f0e",
        "Appraisal (Q3)": "#2ca02c", "Normative (Q4)": "#d62728",
        "Non-substantive": "#9467bd",
        "Own work (Q5=Y)": "#8c564b", "No own work (Q5=N)": "#e377c2",
    }

    # Figure 1: Key features bar chart (4 panels)
    panel_features = [
        ("r_words", "Reply Word Count"),
        ("r_pronoun_you_rate", "\"You\" Rate (per word)"),
        ("r_n_resources", "Resource Links"),
        ("r_reply_lag_hours", "Reply Lag (hours)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax_i, (feat_col, feat_title) in enumerate(panel_features):
        ax = axes[ax_i // 2, ax_i % 2]
        means = []
        sems = []
        for sg_name, sg_mask in subgroups:
            if sg_name == "Full sample":
                continue
            vals = df.loc[sg_mask, feat_col].dropna()
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)))

        x = np.arange(len(sg_names_plot))
        bars = ax.bar(x, means, 0.6,
                      color=[colors.get(s, "#333") for s in sg_names_plot],
                      alpha=0.85)
        ax.errorbar(x, means, yerr=sems, fmt="none", color="black", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sg_names_plot, fontsize=7, rotation=20, ha="right")
        ax.set_title(feat_title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Mentor Reply Features by Question Type (Treated Group Only)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "feature_by_subgroup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved feature_by_subgroup.pdf")

    # Figure 2: Pronoun comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, (feat, title) in enumerate([
        ("r_pronoun_you_rate", "\"You\" (2nd person)"),
        ("r_pronoun_i_rate", "\"I\" (1st person singular)"),
        ("r_pronoun_we_rate", "\"We\" (1st person plural)"),
    ]):
        ax = axes[ax_i]
        means = [df.loc[m, feat].dropna().mean() for _, m in subgroups if _ != "Full sample"]
        x = np.arange(len(sg_names_plot))
        ax.bar(x, means, 0.6,
               color=[colors.get(s, "#333") for s in sg_names_plot], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sg_names_plot, fontsize=7, rotation=20, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Rate (per word)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Pronoun Usage in Mentor Replies by Question Type",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "pronouns_by_subgroup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved pronouns_by_subgroup.pdf")

    # Figure 3: Sentiment comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, (feat, title) in enumerate([
        ("r_vader_compound", "VADER Compound"),
        ("r_tb_polarity", "TextBlob Polarity"),
        ("r_tb_subjectivity", "TextBlob Subjectivity"),
    ]):
        ax = axes[ax_i]
        means = [df.loc[m, feat].dropna().mean() for _, m in subgroups if _ != "Full sample"]
        x = np.arange(len(sg_names_plot))
        ax.bar(x, means, 0.6,
               color=[colors.get(s, "#333") for s in sg_names_plot], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sg_names_plot, fontsize=7, rotation=20, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Sentiment of Mentor Replies by Question Type",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sentiment_by_subgroup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved sentiment_by_subgroup.pdf")

    # Figure 4: Readability + actionability
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_i, (feat, title) in enumerate([
        ("r_flesch_kincaid", "Flesch-Kincaid Grade"),
        ("r_n_imperatives", "Imperative Sentences"),
        ("r_n_resources", "Resource Links"),
    ]):
        ax = axes[ax_i]
        means = [df.loc[m, feat].dropna().mean() for _, m in subgroups if _ != "Full sample"]
        x = np.arange(len(sg_names_plot))
        ax.bar(x, means, 0.6,
               color=[colors.get(s, "#333") for s in sg_names_plot], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sg_names_plot, fontsize=7, rotation=20, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Readability & Actionability of Mentor Replies by Question Type",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "readability_by_subgroup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved readability_by_subgroup.pdf")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY STATS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  SUMMARY STATISTICS (Full sample, treated only)")
    print(SEP)

    num_cols = [c for c in df.columns if c not in
                ["conversation_id", "Q0", "Q2", "Q3", "Q4", "Q5"]]
    desc = df[num_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.columns = ["N", "Mean", "SD", "Min", "Median", "Max"]
    print(desc.to_string())

    elapsed = time.time() - t0
    print(f"\n{SEP}")
    print(f"  s15 COMPLETE ({elapsed:.0f}s)")
    print(f"  Output: {OUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
