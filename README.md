# Wikipedia Mentor Research — Data Collection & Analysis Pipeline

Research pipeline for collecting and analyzing Wikipedia's [Growth Team Mentorship Program](https://www.mediawiki.org/wiki/Growth/Mentor_dashboard) data.

This repository contains the code and data for the paper:

> **Who Benefits from Wikipedia's Growth Mentorship? The Role of Newcomer Information Needs.**

---

## Overview

The program pairs newly registered Wikipedia editors ("mentees") with experienced editors ("mentors"). This pipeline collects the full historical roster of mentors, their conversations with mentees, validates coverage using Wikipedia edit tags, recovers missing conversations, cleans wikitext into analysis-ready text, exports a ConvoKit corpus, and runs the full causal analysis (propensity score stratification, heterogeneous effects by question type, and reply text feature analysis).

**Pipeline:**

```
s1_collect_mentors.py       → data/s1/  (mentor roster + weight history)
s2_collect_conversations.py → data/s2/  (talk page conversations, merged)
s3_validate_tags.py         → data/s3/  (edit tag validation + coverage report)
s4_recover_missing.py       → data/s4/  (recover deleted/missing conversations)
s5_merge_dataset.py         → data/s5/  (final merged dataset: s2 + s4)
s6_collect_users.py         → data/s6/  (user profiles, contributions, logs, abuse filter)
s6_describe.py              → stdout    (descriptive statistics for s1-s5)
s7_clean_conversations.py   → data/s7/  (wikitext → cleaned text + semantic tokens)
s8_extract_first_turns.py   → data/s8/  (first question + reply per mentee)
s9_export_corpus.py         → wiki-mentor-corpus/  (ConvoKit format)
s10_1_perspective_api.py       → data/s10/ (Google Perspective API toxicity scores)
s10_2_llm_annotation.py        → data/s10/ (LLM annotation of question types)
s10_3_agreement.py             → data/s10/ (inter-annotator & human-LLM agreement)
s11_build_features.py          → data/s11/ (analysis-ready feature matrix)
s12_build_psm_dataset.py       → data/s12/ (PSM dataset: covariates + outcomes)
s13_psm_analysis.py            → data/s13/ (main PSM results + robustness checks)
s14_heterogeneous_effects.py   → data/s14/ (within-subgroup PSM by question type)
s15_1_reply_text_analysis.py   → data/s15/ (reply text feature extraction)
s15_2_reply_association.py     → data/s15/ (reply feature–retention association)
```

All scripts are **resume-safe** with checkpoint files. Restart at any point and completed work is skipped.

---

## Directory Structure

```
├── s1_collect_mentors.py
├── s2_collect_conversations.py
├── s3_validate_tags.py
├── s4_recover_missing.py
├── s5_merge_dataset.py
├── s6_collect_users.py
├── s6_describe.py
├── s7_clean_conversations.py
├── s8_extract_first_turns.py
├── s9_export_corpus.py
├── s10_1_perspective_api.py
├── s10_2_llm_annotation.py
├── s10_3_agreement.py
├── s11_build_features.py
├── s12_build_psm_dataset.py
├── s13_psm_analysis.py
├── s14_heterogeneous_effects.py
├── s15_1_reply_text_analysis.py
├── s15_2_reply_association.py
├── data/
│   ├── s1/    (mentor list, change log, checkpoints)
│   ├── s2/    (conversations, merged output, checkpoints)
│   ├── s3/    (tag revision cache, coverage results)
│   ├── s4/    (recovered conversations, caches)
│   ├── s5/    (final merged dataset, report)
│   ├── s6/    (user profiles, contributions, logs, abuse filter)
│   ├── s7/    (cleaned conversations)
│   ├── s8/    (first turns extraction)
│   ├── s10/   (Perspective scores, LLM annotations, agreement)
│   ├── s11/   (feature matrix)
│   ├── s12/   (PSM dataset: covariates + outcomes)
│   ├── s13/   (PSM results, figures, tables)
│   ├── s14/   (subgroup PSM results, figures, tables)
│   ├── s15/   (reply features, association results)
│   └── legacy/  (old Perspective data for import)
├── wiki-mentor-corpus/  (ConvoKit output)
└── README.md
```

---

## Scripts

### `s1_collect_mentors.py`
Collects the complete mentor roster (2021-05 to present) from two API sources:
- `Wikipedia:Growth_Team_features/Mentor_list` — wikitext revisions (2021-05 to 2022-10)
- `MediaWiki:GrowthMentors.json` — JSON revisions (2022-10 to present)

Builds a full timeline per mentor including pool status changes, weight history, join/leave events.

| Output | Description |
|--------|-------------|
| `data/s1/s1_mentor_list.jsonl` | One record per mentor with full timeline |
| `data/s1/s1_mentor_change.jsonl` | Every state-change event across all mentors |

---

### `s2_collect_conversations.py`
Two-phase script:
1. **Phase 1 — Collection:** Fetches User talk pages + archives for each mentor. Runs search API for unmatched pages.
2. **Phase 2 — Fix & Merge:** Discovers ALL subpages (not just archives), reclassifies misplaced unmatched records, fetches newly-found pages, and merges everything into a single deduplicated output.

| Output | Description |
|--------|-------------|
| `data/s2/s2_mentor_conversation_merged.jsonl` | All conversations, deduplicated and merged |
| `data/s2/s2_mentor_conversation_unmatched_clean.jsonl` | Non-s1 mentor pages |

---

### `s3_validate_tags.py`
Two-phase validation using Wikipedia edit tags as ground truth:
1. **Phase 1 — Download:** Fetches all revisions tagged `mentorship panel question` / `mentorship module question`.
2. **Phase 2 — Match:** Compares tag revisions against s2 regex-extracted questions.

| Output | Description |
|--------|-------------|
| `data/s3/s3_tag_revisions_cache.jsonl` | Raw tag revision metadata |
| `data/s3/s3_tag_match_results.jsonl` | Per-mentor coverage results |
| `data/s3/s3_tag_match_report.txt` | Summary coverage table |

---

### `s4_recover_missing.py`
Recovers conversations identified as missing by s3 tag validation using batch revision history diffs.

| Output | Description |
|--------|-------------|
| `data/s4/s4_recovered_conversations.jsonl` | Recovered conversations |

---

### `s5_merge_dataset.py`
Merges s2 wikitext-parsed questions with s4 recovered conversations. Deduplicates by (mentor, mentee, timestamp).

| Output | Description |
|--------|-------------|
| `data/s5/s5_all_conversations.jsonl` | Complete dataset (~41,000 conversations) |

Each record: `mentor`, `mentee`, `revid`, `timestamp`, `article`, `question_text`, `mentor_reply`, `source`, `page`.

---

### `s6_collect_users.py`
Collects user data for all mentors & mentees from Wikipedia API:
- Phase 1: User profiles (registration, editcount, groups, gender, block status)
- Phase 2: User contributions (full edit history since 2020-01-01)
- Phase 3: Log events (blocks, rights changes, account creation)
- Phase 4: Abuse filter log

Resume-safe with per-user checkpoints. Concurrent workers with rate limiting.

| Output | Description |
|--------|-------------|
| `data/s6/s6_user_profiles.jsonl` | User profiles (~37K users) |
| `data/s6/s6_user_contribs.jsonl` | Full edit history (~3.3 GB) |
| `data/s6/s6_user_logevents.jsonl` | Log events |
| `data/s6/s6_user_abuselog.jsonl` | Abuse filter hits |

---

### `s6_describe.py`
Prints descriptive statistics for s1 (mentors), s2 (conversations), and s5 (final dataset).

---

### `s7_clean_conversations.py`
Two-level wikitext cleaning:
1. `clean_wikitext()` — Remove wiki markup, signatures, templates → plain text with semantic tokens (`[POLICY]`, `[HELP_PAGE]`, `[DRAFT]`, `[WIKILINK]`, `[LINK]`, etc.)
2. `make_emb_text()` — Remove usernames, residual signatures → embedding-ready text

Also: reply truncation (keep only mentor's first signed block), reply timestamp extraction.

| Output | Description |
|--------|-------------|
| `data/s7/s7_conversations_cleaned.jsonl` | Cleaned conversations (~75 MB) |

---

### `s8_extract_first_turns.py`
Extracts each mentee's first question + mentor's first reply. Adds:
- `is_first_conversation`: whether this is the mentee's earliest conversation
- `is_english`: ASCII letter ratio ≥ 0.8
- `reply_signer`: who actually signed the reply (assigned_mentor / other_only / mentor_and_other / unknown)
- `actual_responders`: list of signer names

| Output | Description |
|--------|-------------|
| `data/s8/s8_first_turns.jsonl` | First-turn dataset (~51 MB) |

---

### `s9_export_corpus.py`
Exports to [ConvoKit](https://convokit.cornell.edu/) corpus format with additional data cleaning:
- Drops empty mentee/question records
- Fixes negative response time → treat as no reply
- Fixes empty reply_emb → treat as no reply
- Adds `mentor_type` (auto/manual) from s1 weight_history at conversation time

| Output | Description |
|--------|-------------|
| `wiki-mentor-corpus/utterances.jsonl` | One JSON per question/reply |
| `wiki-mentor-corpus/speakers.json` | Speaker metadata |
| `wiki-mentor-corpus/conversations.json` | Conversation metadata |
| `wiki-mentor-corpus/corpus.json` | Corpus-level metadata |
| `wiki-mentor-corpus/index.json` | Field type index |

---

### `s10_1_perspective_api.py` (optional)
Fetches Google Perspective API scores (23 attributes: toxicity, insult, curiosity, etc.) for questions and replies. Imports old scores from `data/legacy/`. Resume-safe.

| Output | Description |
|--------|-------------|
| `data/s10/s10_perspective_mentee.csv` | Mentee question scores |
| `data/s10/s10_perspective_mentor.csv` | Mentor reply scores |

---

### `s10_2_llm_annotation.py`
Annotates all 41,339 first-turn questions with Morrison (1993) information-seeking dimensions using DeepSeek-v4-Flash. Five binary dimensions: Q1 (Substantive), Q2 (Referent), Q3 (Appraisal), Q4 (Normative), Q5 (Own Work). Resume-safe with JSONL checkpoint.

| Output | Description |
|--------|-------------|
| `data/s10/corpus_annotations_v2.jsonl` | One record per question with Q1–Q5 labels |

---

### `s10_3_agreement.py`
Computes inter-annotator reliability (Cohen's κ) for human-human (Iteration 2) and human-LLM agreement. Reports separately for validation set (first 120), test set (last 121), and full set (n=241).

| Output | Description |
|--------|-------------|
| `data/s10/agreement_v2.csv` | κ values by dimension and annotator pair |

---

### `s11_build_features.py`
Builds the analysis-ready feature matrix from s1/s5/s6/s7/s8/s10. Computes edit history features, text embeddings (OpenAI), linguistic features, Perspective API scores, and temporal variables.

| Output | Description |
|--------|-------------|
| `data/s11/s11_features.jsonl` | One record per conversation with all features |

---

### `s12_build_psm_dataset.py`
Assembles the final PSM dataset from s8/s10/s11. Applies sample restrictions (English, first conversation, auto-assigned mentor). Constructs 164 covariates across 6 feature blocks, treatment indicators, and 10 outcome variables. Exports as NumPy archive.

| Output | Description |
|--------|-------------|
| `data/s12/psm_data/psm_dataset.npz` | Arrays: X covariates, treatment, outcomes, IDs |

---

### `s13_psm_analysis.py`
Full PSM analysis: propensity score estimation (logistic regression with 5-fold CV), stratification, covariate balance diagnostics, ATT estimation with cluster bootstrap CIs, feature ablation (15 specifications), strata sensitivity, PS trimming, outcome window sensitivity, Rosenbaum bounds, and E-values.

| Output | Description |
|--------|-------------|
| `data/s13/tables/` | Main results, robustness checks, covariate balance |
| `data/s13/figures/` | Forest plot, PS distribution, balance plots, robustness |

---

### `s14_heterogeneous_effects.py`
Within-subgroup PSM analysis by Morrison question type (Q1–Q5). Re-estimates propensity scores within each subgroup and computes subgroup-specific ATTs with bootstrap CIs.

| Output | Description |
|--------|-------------|
| `data/s14/tables/` | Subgroup ATTs, cross-DV heatmap data |
| `data/s14/figures/` | Subgroup forest plots, heatmap |

---

### `s15_1_reply_text_analysis.py`
Extracts interpretable text features from mentor replies (treated group only): reply lag, word count, question marks, "I"/"we" rates, gratitude, greeting, subjectivity, etc. Compares feature distributions across Morrison question-type subgroups.

| Output | Description |
|--------|-------------|
| `data/s15/reply_features.csv` | Per-reply feature matrix |
| `data/s15/subgroup_comparison.csv` | Feature means by question type |

---

### `s15_2_reply_association.py`
OLS regression of reply text features on newcomer retention outcomes, controlling for 164 pre-treatment covariates. Reports standardized coefficients across multiple DVs and subgroups.

| Output | Description |
|--------|-------------|
| `data/s15/association_results.csv` | Full-sample regression coefficients |
| `data/s15/association_by_subgroup.csv` | Subgroup-level coefficients |

---

## Setup & Usage

**Dependencies:** Python 3 standard library only (s1-s9). See `requirements.txt` for additional packages needed by s10–s15 (numpy, pandas, scikit-learn, matplotlib, openai, etc.).

**Environment:** Copy `.env.example` to `.env` and fill in your keys:
```
WIKI_BOT_PASSWORD=your_bot_password_here       # s6: authenticated Wikipedia API access
PERSPECTIVE_API_KEY=your_perspective_api_key    # s10_1: Google Perspective API
DEEPSEEK_API_KEY=your_deepseek_api_key         # s10_2: LLM annotation
```

**Run in order:**

```bash
# Data collection (s1–s9)
python3 s1_collect_mentors.py              # ~2h
python3 s2_collect_conversations.py        # ~4-6h
python3 s3_validate_tags.py                # ~1h
python3 s4_recover_missing.py              # ~2-4h
python3 s5_merge_dataset.py                # <1min
python3 s6_collect_users.py                # ~6-12h
python3 s6_describe.py                     # <1min
python3 s7_clean_conversations.py          # ~2min
python3 s8_extract_first_turns.py          # ~1min
python3 s9_export_corpus.py                # <1min

# Annotation & features (s10–s11)
python3 s10_1_perspective_api.py           # ~10h (optional, requires API key)
python3 s10_2_llm_annotation.py            # ~2h (requires API key)
python3 s10_3_agreement.py                 # <1min (requires human annotation files)
python3 s11_build_features.py              # ~30min

# Analysis (s12–s15)
python3 s12_build_psm_dataset.py           # ~2min
python3 s13_psm_analysis.py                # ~20min
python3 s14_heterogeneous_effects.py       # ~30min
python3 s15_1_reply_text_analysis.py       # ~2min
python3 s15_2_reply_association.py         # ~5min
```

All scripts are resume-safe. Restart at any point and completed work is skipped.

---

## Data Summary

*As of 2026-04-24.*

| Stage | Key Metric |
|-------|-----------|
| S1 | 440 mentors (288 active, 152 exited) |
| S2 | ~42,200 questions from talk pages |
| S3 | 81.4% strict tag match, 15.6% missing |
| S4 | Recovered ~5,174 missing conversations |
| S5 | **41,339 total conversations** (403 mentors × 36,431 mentees) |
| S6 | 36,822 user profiles + full edit histories (3.3 GB) |
| S7 | 41,339 cleaned conversations |
| S8 | 41,339 first-turn records (35,920 English first conversations) |
| S9 | **35,480 conversations** in ConvoKit corpus (after cleanup) |
| S10 | LLM annotations for 41,339 questions|
| S11 | Feature matrix: edit history, text, embeddings, Perspective scores |
| S12 | **35,381 conversations** in PSM dataset (164 covariates, 10 outcomes) |
| S13 | ATT = +0.0165 (primary), stable across 15 specifications |
| S14 | Subgroup ATTs: Technical +0.024\*, Referent ≈ 0, Own Work +0.033\* |
| S15 | Reply lag strongest correlate of retention (β = −0.035) |

---

## Notes

- Large data files are excluded from version control via `.gitignore`.
- `wiki-mentor-corpus/` is the final ConvoKit corpus, loadable with `from convokit import Corpus; c = Corpus('wiki-mentor-corpus')`.
- 90.9% of mentees have only 1 conversation; 7.0% have 2; 1.3% have 3+.
- S4 recovery skips reverted edits (identified via `mw-reverted` tag).
