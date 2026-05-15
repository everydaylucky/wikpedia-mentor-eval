#!/usr/bin/env python3
"""
s12_build_psm_dataset.py — Build complete PSM analysis dataset.

Replicates the original 45a_build_analysis_dataset.py feature design using the
unified pipeline data sources (s6/s7/s8/s10/s11).

Feature groups:
  X_E        34  Mentee edit history (pre-Q)
  X_Qtext    37  Question text features (wiki format + VADER + TextBlob + politeness)
  X_Qpersp   15  Perspective API scores (15 reliable attributes)
  X_emb20    20  PCA-20 of Qwen text-embedding-v4
  X_temporal  N  Year-month dummies + article context flag
  X_Qtype     5  LLM question type (Morrison socialization: substantive/referent/appraisal/normative/own_work)
  ─────────────
  Total:    111 + N_temporal

  Vs original 171 = 34+44+15+20+58:  dropped detoxify(6)+emotion(1), added Qtype(5).

Output:
  data/s12/psm_data/psm_dataset.npz

Dependencies:
  pip install numpy pandas scikit-learn vaderSentiment textblob
"""
import argparse, json, math, re, sys, warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("WARNING: vaderSentiment not installed → VADER features = 0")
    print("  pip install vaderSentiment")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("WARNING: textblob not installed → TextBlob features = 0")
    print("  pip install textblob")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = DATA / "s12" / "psm_data"

S7_FILE  = DATA / "s7" / "s7_conversations_cleaned.jsonl"
S8_FILE  = DATA / "s8" / "s8_first_turns.jsonl"
S11_FILE = DATA / "s11" / "s11_features.jsonl"
ANNOTATIONS_FILE = BASE / "data" / "s10" / "corpus_annotations_v2.jsonl"

DEFAULT_S6_DIR = Path("/Users/Shared/baiduyun/00 Code/0Wiki/0wiki-mentor-github")
EMB_Q_NPZ  = DEFAULT_S6_DIR / "s8_embeddings_q.npz"
EMB_Q_META = DEFAULT_S6_DIR / "s8_embeddings_q_meta.jsonl"

SEP = "=" * 72

# ── Column names (matching original 45a) ────────────────────────────────────
E_COLS = [
    "n_edits", "n_edits_7d", "n_edits_1d",
    "ns0_mainspace", "ns2_userpage", "ns3_usertalk", "ns4_wp", "ns118_draft",
    "n_unique_ns", "mainspace_ratio", "draft_ratio",
    "avg_sizediff", "std_sizediff", "max_sizediff", "neg_sizediff_ratio",
    "hours_since_last_edit", "active_span_hours", "account_age_hours",
    "n_reverts", "n_ai_reverts", "revert_rate_pre",
    "tag_visualeditor", "tag_mobile", "tag_newcomer_task",
    "tag_editcheck_newref", "tag_mw_reverted", "tag_discussion",
    "log_create", "log_thanks", "n_abuse", "n_abuse_warn",
    "q_weekday", "q_hour_utc", "q_is_weekend",
]

QTEXT_COLS = [
    # Wiki format (10)
    "q_has_sig", "q_has_unsigned", "q_link_count", "q_template_count",
    "q_has_url", "q_mentions_wp", "q_mentions_help", "q_is_indented",
    "q_body_char_len", "q_body_word_count",
    # VADER (4)
    "q_vader_neg", "q_vader_neu", "q_vader_pos", "q_vader_compound",
    # TextBlob (2)
    "q_tb_polarity", "q_tb_subjectivity",
    # Politeness strategies — regex approximation of ConvoKit (21)
    "q_poly_feature_politeness_==Please==",
    "q_poly_feature_politeness_==Please_start==",
    "q_poly_feature_politeness_==HASHEDGE==",
    "q_poly_feature_politeness_==Indirect_(btw)==",
    "q_poly_feature_politeness_==Hedges==",
    "q_poly_feature_politeness_==Factuality==",
    "q_poly_feature_politeness_==Deference==",
    "q_poly_feature_politeness_==Gratitude==",
    "q_poly_feature_politeness_==Apologizing==",
    "q_poly_feature_politeness_==1st_person_pl.==",
    "q_poly_feature_politeness_==1st_person==",
    "q_poly_feature_politeness_==1st_person_start==",
    "q_poly_feature_politeness_==2nd_person==",
    "q_poly_feature_politeness_==2nd_person_start==",
    "q_poly_feature_politeness_==Indirect_(greeting)==",
    "q_poly_feature_politeness_==Direct_question==",
    "q_poly_feature_politeness_==Direct_start==",
    "q_poly_feature_politeness_==HASPOSITIVE==",
    "q_poly_feature_politeness_==HASNEGATIVE==",
    "q_poly_feature_politeness_==SUBJUNCTIVE==",
    "q_poly_feature_politeness_==INDICATIVE==",
]

QPERSP_COLS = [
    "q_persp_toxicity", "q_persp_severe_toxicity", "q_persp_identity_attack",
    "q_persp_insult", "q_persp_profanity", "q_persp_threat",
    "q_persp_sexually_explicit", "q_persp_flirtation",
]

# Perspective column mapping: s11 name → our name
PERSP_S11_MAP = {
    f"persp_q_{a.replace('q_persp_', '')}": a
    for a in QPERSP_COLS
}  # persp_q_toxicity → q_persp_toxicity, etc.

# LLM question annotation (Morrison socialization framework, 5 binary dims)
# Q0=Substantive, Q2=Referent, Q3=Appraisal, Q4=Normative, Q5=OwnWork
# File field names: Q0,Q2,Q3,Q4,Q5 (Q1 in the note = Q0 in the file)
QTYPE_COLS = [
    "q_substantive",   # Q0/Q1: substantive question about Wikipedia editing
    "q_referent",      # Q2: lacks direction / doesn't know what to do
    "q_appraisal",     # Q3: requests feedback on existing work
    "q_normative",     # Q4: seeks info about rules/policies/norms
    "q_own_work",      # Q5: has prior observable editing activity
]
QTYPE_FILE_MAP = {"Q0": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4}


# ── Helpers ─────────────────────────────────────────────────────────────────

def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def safe_div(a, b, default=0):
    return a / b if b > 0 else default


# ── Wiki format features (from raw wikitext) ───────────────────────────────

_TS_RE = re.compile(r'\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)')
_UNSIGNED_RE = re.compile(r'\{\{unsigned|Preceding unsigned comment added by', re.I)


def extract_wiki_features(raw_text):
    if not raw_text:
        return [0] * 10

    has_sig = int(bool(_TS_RE.search(raw_text)))
    has_unsigned = int(bool(_UNSIGNED_RE.search(raw_text)))

    sig_match = _TS_RE.search(raw_text)
    body = raw_text[:sig_match.start()].strip() if sig_match else raw_text.strip()
    body = re.sub(r'[—–~\-]{2,}\s*$', '', body).strip()

    return [
        has_sig,
        has_unsigned,
        len(re.findall(r'\[\[.+?\]\]', body)),      # link_count
        len(re.findall(r'\{\{.+?\}\}', body)),       # template_count
        int("http" in body.lower()),                  # has_url
        len(re.findall(r'WP:', body, re.I)),          # mentions_wp
        len(re.findall(r'Help:', body, re.I)),        # mentions_help
        int(body.startswith(':') or body.startswith('*')),  # is_indented
        len(body),                                    # body_char_len
        len(body.split()),                            # body_word_count
    ]


# ── VADER features ──────────────────────────────────────────────────────────

def extract_vader(text):
    if not HAS_VADER or not text:
        return [0.0, 0.0, 0.0, 0.0]
    vs = _vader.polarity_scores(text)
    return [vs["neg"], vs["neu"], vs["pos"], vs["compound"]]


# ── TextBlob features ──────────────────────────────────────────────────────

def extract_textblob(text):
    if not HAS_TEXTBLOB or not text:
        return [0.0, 0.0]
    tb = TextBlob(text)
    return [tb.sentiment.polarity, tb.sentiment.subjectivity]


# ── Politeness features (regex approximation of ConvoKit) ───────────────────

def extract_politeness(text):
    if not text:
        return [0] * 21

    lo = text.lower().strip()
    return [
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


# ── Edit history features (from s6 contribs, matching original 45a) ────────

def build_E_features(edits, Q, pre_logs, pre_abuse, account_age_hours):
    pre = [e for e in edits if e["ts"] < Q]
    n = len(pre)

    ns_counts = Counter(e["ns"] for e in pre)
    tag_counts = Counter()
    sizediffs = []
    for e in pre:
        sizediffs.append(e["sizediff"])
        for t in e.get("tags", []):
            tag_counts[t] += 1

    avg_sd = np.mean(sizediffs) if sizediffs else 0
    std_sd = np.std(sizediffs) if len(sizediffs) > 1 else 0
    max_sd = max([abs(s) for s in sizediffs], default=0)
    neg_r = safe_div(sum(1 for s in sizediffs if s < 0), n)
    hrs_last = (Q - pre[-1]["ts"]).total_seconds() / 3600 if pre else -1
    span = (pre[-1]["ts"] - pre[0]["ts"]).total_seconds() / 3600 if len(pre) > 1 else 0

    n_7d = sum(1 for e in pre if (Q - e["ts"]).days <= 7)
    n_1d = sum(1 for e in pre if (Q - e["ts"]).days <= 1)

    log_ta = Counter()
    for l in pre_logs:
        log_ta[l["type"] + "/" + l["action"]] += 1

    n_reverted = tag_counts.get("mw-reverted", 0)

    return [
        n,                                    # n_edits
        n_7d,                                 # n_edits_7d
        n_1d,                                 # n_edits_1d
        ns_counts.get(0, 0),                  # ns0_mainspace
        ns_counts.get(2, 0),                  # ns2_userpage
        ns_counts.get(3, 0),                  # ns3_usertalk
        ns_counts.get(4, 0),                  # ns4_wp
        ns_counts.get(118, 0),                # ns118_draft
        len(ns_counts),                       # n_unique_ns
        safe_div(ns_counts.get(0, 0), n),     # mainspace_ratio
        safe_div(ns_counts.get(118, 0), n),   # draft_ratio
        avg_sd,                               # avg_sizediff
        std_sd,                               # std_sizediff
        max_sd,                               # max_sizediff
        neg_r,                                # neg_sizediff_ratio
        hrs_last,                             # hours_since_last_edit
        span,                                 # active_span_hours
        account_age_hours,                    # account_age_hours
        n_reverted,                           # n_reverts (count of mw-reverted tags)
        0,                                    # n_ai_reverts (not available in s6)
        safe_div(n_reverted, n),              # revert_rate_pre
        tag_counts.get("visualeditor", 0),
        tag_counts.get("mobile edit", 0),
        tag_counts.get("newcomer task", 0),
        tag_counts.get("editcheck-newreference", 0),
        tag_counts.get("mw-reverted", 0),
        tag_counts.get("discussiontools-added-comment", 0),
        log_ta.get("create/create", 0) + log_ta.get("newusers/create", 0),
        log_ta.get("thanks/thank", 0),
        len(pre_abuse),
        sum(1 for a in pre_abuse if "warn" in a.get("result", "").lower()),
        Q.weekday(),                          # q_weekday
        Q.hour,                               # q_hour_utc
        int(Q.weekday() >= 5),                # q_is_weekend
    ]


# ── Outcome variables (anchor-based, matching original 45a) ─────────────────

def build_outcomes(edits, Q, anchor, replied):
    post_anchor_main = sorted(
        [e for e in edits if e["ts"] > anchor and e["ns"] == 0],
        key=lambda x: x["ts"]
    )
    post_anchor_all = [e for e in edits if e["ts"] > anchor and e["ns"] % 2 == 0]

    w14 = anchor + timedelta(days=14)
    w28 = anchor + timedelta(days=28)
    w30 = anchor + timedelta(days=30)
    w60 = anchor + timedelta(days=60)

    # Constructive = not reverted
    post_main_c = [e for e in post_anchor_main if "mw-reverted" not in e.get("tags", [])]
    post_all_c = [e for e in post_anchor_all if "mw-reverted" not in e.get("tags", [])]

    primary = int(bool(post_anchor_main) and post_anchor_main[0]["ts"] <= w14)
    primary_c = int(bool(post_main_c) and post_main_c[0]["ts"] <= w14)

    dates_14d = set(e["ts"].date() for e in post_main_c if e["ts"] <= w14)
    cross_day_c = int(len(dates_14d) >= 2)

    dates_14d_any = set(e["ts"].date() for e in post_anchor_main if e["ts"] <= w14)
    cross_day_any = int(len(dates_14d_any) >= 2)

    sec2 = int(any(e["ts"] > w14 and e["ts"] <= w28 and e["ns"] == 0 for e in post_anchor_main))
    c_15_60 = int(any(e["ts"] > w14 and e["ts"] <= w60 for e in post_main_c))

    n_mainspace_14d = len([e for e in post_anchor_main if e["ts"] <= w14])

    w14_edits = [e for e in post_anchor_all if e["ts"] <= w14]
    reverted_any = int(sum(1 for e in w14_edits if "mw-reverted" in e.get("tags", [])) >= 1)
    unique_ns = len(set(e["ns"] for e in w14_edits))

    active_14d = len(set(e["ts"].date() for e in post_anchor_all if e["ts"] <= w14))
    active_30d = len(set(e["ts"].date() for e in post_anchor_all if e["ts"] <= w30))
    constr_30d = len(set(e["ts"].date() for e in post_all_c if e["ts"] <= w30))

    # 15-30d window outcomes (persistence beyond initial 14d)
    mainspace_15_30d = int(any(e["ts"] > w14 and e["ts"] <= w30 for e in post_anchor_main))
    n_mainspace_15_30d = len([e for e in post_anchor_main if e["ts"] > w14 and e["ts"] <= w30])
    active_days_15_30d = len(set(e["ts"].date() for e in post_anchor_all if e["ts"] > w14 and e["ts"] <= w30))

    oc = {
        "primary": primary,
        "n_mainspace_edits_14d": n_mainspace_14d,
        "primary_constructive": primary_c,
        "sec2": sec2,
        "constructive_edit_15_60d": c_15_60,
        "reverted_any": reverted_any,
        "active_days_14d": active_14d,
        "active_days_30d": active_30d,
        "constructive_days_30d": constr_30d,
        "unique_ns": unique_ns,
        "cross_day_constructive_14d": cross_day_c,
        "cross_day_any_14d": cross_day_any,
        "mainspace_15_30d": mainspace_15_30d,
        "n_mainspace_15_30d": n_mainspace_15_30d,
        "active_days_15_30d": active_days_15_30d,
    }

    # Window sensitivity outcomes (any mainspace edit, regardless of revert)
    for wd in [7, 14, 21, 28, 30, 60, 180]:
        ww = anchor + timedelta(days=wd)
        oc[f"mainspace_{wd}d"] = int(any(e["ts"] <= ww for e in post_anchor_main))

    # Teahouse-style windowed outcomes: 3-4wk (15-28d), 1-2mo (29-60d), 2-6mo (61-180d)
    w180 = anchor + timedelta(days=180)
    th_windows = [("15_28d", w14, w28), ("29_60d", w28, w60), ("61_180d", w60, w180)]
    for wname, wlo, whi in th_windows:
        edits_in_w = [e for e in post_anchor_main if e["ts"] > wlo and e["ts"] <= whi]
        n_in_w = len(edits_in_w)
        oc[f"th_1plus_{wname}"] = int(n_in_w >= 1)
        oc[f"th_5plus_{wname}"] = int(n_in_w >= 5)

    return oc


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s6-dir", type=str, default=str(DEFAULT_S6_DIR))
    args = ap.parse_args()
    s6_dir = Path(args.s6_dir)

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{SEP}")
    print("  s12: BUILD PSM DATASET")
    print(SEP)

    # ── 1. Load s8 first turns (master record list) ─────────────────────────
    print("\n[1] Loading s8 first turns (master records)...")
    s8_data = {}
    with open(S8_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("is_first_conversation"):
                continue
            if not r.get("is_english"):
                continue
            s8_data[r["conversation_id"]] = r
    print(f"  {len(s8_data):,} English first-conversation records")

    # ── 2. Load s11 features (for perspective + mentor features) ────────────
    print("[2] Loading s11 features...")
    s11_data = {}
    with open(S11_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s11_data[r["conversation_id"]] = r
    print(f"  {len(s11_data):,} records")

    # Use s11 as the starting set, then filter to corpus-only
    cid_list = sorted(s11_data.keys())
    print(f"  s11 records: {len(cid_list):,}")

    # ── 3. Load s7 cleaned (for raw wikitext) ──────────────────────────────
    print("[3] Loading s7 conversations (raw wikitext)...")
    s7_raw = {}
    with open(S7_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cid = r["conversation_id"]
            if cid in s11_data:
                s7_raw[cid] = r.get("question_raw", "")
    print(f"  {len(s7_raw):,} matched")

    # ── 4. Load s6 user data ────────────────────────────────────────────────
    needed_users = set()
    user_to_cids = defaultdict(list)
    for cid in cid_list:
        s8r = s8_data.get(cid)
        if s8r:
            mentee = s8r["mentee"]
            needed_users.add(mentee)
            user_to_cids[mentee].append(cid)
    print(f"\n[4] Loading s6 user data for {len(needed_users):,} mentees...")

    # 4a. Profiles
    print("  [4a] Profiles...")
    user_profiles = {}
    prof_path = s6_dir / "s6_user_profiles.jsonl"
    if prof_path.exists():
        with open(prof_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["username"] in needed_users:
                    user_profiles[r["username"]] = r
    print(f"    {len(user_profiles):,} profiles loaded")

    # 4b. Contributions
    print("  [4b] Contributions (this may take a minute)...")
    user_edits = {}
    contribs_path = s6_dir / "s6_user_contribs.jsonl"
    if contribs_path.exists():
        loaded = 0
        with open(contribs_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                uname = r["username"]
                if uname not in needed_users:
                    continue
                edits = []
                for e in (r.get("edits") or []):
                    ts = parse_ts(e.get("timestamp"))
                    if ts:
                        edits.append({
                            "ts": ts, "ns": e.get("ns", 0),
                            "sizediff": e.get("sizediff", 0),
                            "tags": set(e.get("tags", [])),
                        })
                edits.sort(key=lambda x: x["ts"])
                user_edits[uname] = edits
                loaded += 1
                if loaded % 5000 == 0:
                    print(f"      {loaded:,}...", flush=True)
        print(f"    {len(user_edits):,} users with contributions")
    else:
        print(f"    WARNING: {contribs_path} not found!")

    # 4c. Logevents
    print("  [4c] Logevents...")
    user_logs = {}
    logs_path = s6_dir / "s6_user_logevents.jsonl"
    if logs_path.exists():
        with open(logs_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                uname = r["username"]
                if uname not in needed_users:
                    continue
                events = []
                for e in (r.get("events") or []):
                    ts = parse_ts(e.get("timestamp"))
                    if ts:
                        events.append({"ts": ts, "type": e.get("type", ""),
                                       "action": e.get("action", "")})
                user_logs[uname] = events
    print(f"    {len(user_logs):,} users with logevents")

    # 4d. Abuselog
    print("  [4d] Abuselog...")
    user_abuse = {}
    abuse_path = s6_dir / "s6_user_abuselog.jsonl"
    if abuse_path.exists():
        with open(abuse_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                uname = r["username"]
                if uname not in needed_users:
                    continue
                entries = []
                for e in (r.get("entries") or []):
                    ts = parse_ts(e.get("timestamp"))
                    if ts:
                        entries.append({"ts": ts, "result": e.get("result", "")})
                user_abuse[uname] = entries
    print(f"    {len(user_abuse):,} users with abuselog")

    # ── 5. Load embeddings ──────────────────────────────────────────────────
    print("\n[5] Loading embeddings...")
    emb_all = np.load(EMB_Q_NPZ)["embeddings"]
    emb_meta = []
    with open(EMB_Q_META, encoding="utf-8") as f:
        for line in f:
            emb_meta.append(json.loads(line))
    emb_cid_to_idx = {m["conversation_id"]: m["idx"] for m in emb_meta}
    print(f"  {emb_all.shape[0]:,} embeddings ({emb_all.shape[1]}d)")

    # ── 5b. Load LLM question annotations ─────────────────────────────────
    print("[5b] Loading LLM question annotations...")
    ann_data = {}
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                # Support both old format (id="q_123") and new format (cid=123)
                if "cid" in r:
                    cid = r["cid"]
                elif "id" in r:
                    cid = int(r["id"].replace("q_", ""))
                else:
                    continue
                if "Q0" in r:
                    ann_data[cid] = [int(r[k] == "Y") for k in ["Q0", "Q2", "Q3", "Q4", "Q5"]]
        print(f"  {len(ann_data):,} annotations loaded")
    else:
        print(f"  WARNING: {ANNOTATIONS_FILE} not found!")

    # ── 5c. Filter to corpus-only (exclude records without annotations) ────
    n_before = len(cid_list)
    cid_list = [cid for cid in cid_list if cid in ann_data]
    N = len(cid_list)
    print(f"\n  Corpus filter: {n_before:,} → {N:,} (excluded {n_before - N} non-corpus records)")

    # ── 6. Compute reply lag for anchor ─────────────────────────────────────
    print("\n[6] Computing reply lag for anchor...")
    all_reply_lags = []
    reply_ts_map = {}
    for cid in cid_list:
        s8r = s8_data.get(cid)
        if not s8r or not s8r.get("has_reply"):
            continue
        rts = parse_ts(s8r.get("reply_timestamp"))
        qts = parse_ts(s8r.get("timestamp"))
        if rts and qts:
            lag = (rts - qts).total_seconds() / 86400
            if 0 <= lag < 365:
                reply_ts_map[cid] = rts
                all_reply_lags.append(lag)

    lag_array = np.array(all_reply_lags) if all_reply_lags else np.array([1.0])
    median_lag = float(np.median(lag_array))
    lag_p25 = float(np.percentile(lag_array, 25))
    lag_p75 = float(np.percentile(lag_array, 75))
    lag_rng = np.random.RandomState(42)
    multi_rng = np.random.RandomState(123)
    N_MULTI_DRAWS = 20
    PSEUDO_STRATS = ["median", "p25", "p75", "zero"]
    print(f"  Median reply lag: {median_lag:.2f} days (mean={lag_array.mean():.2f}, std={lag_array.std():.2f})")
    print(f"  p25={lag_p25:.2f}d, p75={lag_p75:.2f}d")
    print(f"  Conversations with reply timestamp: {len(reply_ts_map):,}")
    print(f"  Control anchor: random sample from {len(lag_array):,} observed reply lags")
    print(f"  Pseudo-time robustness: +4 fixed strategies + multi-draw (K={N_MULTI_DRAWS})")

    # ── 7. Build all features ───────────────────────────────────────────────
    print(f"\n[7] Building features for {N:,} conversations...")
    E_rows = []
    Qtext_rows = []
    Qpersp_rows = []
    Qtype_rows = []
    emb_indices = []

    y_treat_list = []
    y_treat_48h_list = []
    cid_order = []
    mentee_ids = []
    mentor_ids = []
    q_year_month_list = []
    q_has_article_list = []
    mentor_reply_rate_list = []

    OC_KEYS = ["primary", "n_mainspace_edits_14d", "primary_constructive", "sec2", "constructive_edit_15_60d",
                "reverted_any", "active_days_14d", "active_days_30d", "constructive_days_30d",
                "unique_ns", "cross_day_constructive_14d", "cross_day_any_14d",
                "mainspace_15_30d", "n_mainspace_15_30d", "active_days_15_30d",
                "th_1plus_15_28d", "th_5plus_15_28d",
                "th_1plus_29_60d", "th_5plus_29_60d",
                "th_1plus_61_180d", "th_5plus_61_180d"]
    OC_WINDOW_KEYS = [f"mainspace_{w}d" for w in [7, 14, 21, 28, 30, 60, 180]]
    ALL_OC_KEYS = OC_KEYS + OC_WINDOW_KEYS
    OC = {k: [] for k in ALL_OC_KEYS}
    ALT_OC = {}
    for ps_name in PSEUDO_STRATS + ["multi"]:
        ALT_OC[ps_name] = {k: [] for k in ALL_OC_KEYS}

    n_missing_edits = 0
    n_missing_emb = 0

    for i, cid in enumerate(cid_list):
        s8r = s8_data.get(cid, {})
        s11r = s11_data[cid]
        mentee = s8r.get("mentee", s11r.get("mentee", ""))
        Q = parse_ts(s8r.get("timestamp", s11r.get("timestamp")))
        if not Q:
            continue

        # ── Treatment ──
        replied = int(s8r.get("has_reply", s11r.get("has_reply", 0)))
        rts = reply_ts_map.get(cid)
        replied_48h = int(rts is not None and (rts - Q).total_seconds() / 3600 <= 48) if replied else 0

        y_treat_list.append(replied)
        y_treat_48h_list.append(replied_48h)
        cid_order.append(cid)
        mentee_ids.append(mentee)
        mentor_ids.append(s8r.get("mentor", s11r.get("mentor", "")))
        mentor_reply_rate_list.append(s11r.get("m_prior_reply_rate", 0) or 0)

        # Year-month and article context
        q_ym = f"{Q.year}-{Q.month:02d}"
        q_year_month_list.append(q_ym)
        q_has_article_list.append(int(bool(s8r.get("article"))))

        # ── Anchor ──
        if replied and rts:
            anchor = rts
        else:
            sampled_lag = float(lag_rng.choice(lag_array))
            anchor = Q + timedelta(days=sampled_lag)

        # ── E features ──
        edits = user_edits.get(mentee, [])
        if not edits:
            n_missing_edits += 1

        prof = user_profiles.get(mentee, {})
        reg_dt = parse_ts(prof.get("registration"))
        acct_age = (Q - reg_dt).total_seconds() / 3600 if reg_dt else -1

        pre_logs = [l for l in user_logs.get(mentee, []) if l["ts"] < Q]
        pre_abuse = [a for a in user_abuse.get(mentee, []) if a["ts"] < Q]
        E_rows.append(build_E_features(edits, Q, pre_logs, pre_abuse, acct_age))

        # ── Qtext features ──
        raw_text = s7_raw.get(cid, "")
        clean_text = s8r.get("question_clean", "")
        wiki_feats = extract_wiki_features(raw_text)
        vader_feats = extract_vader(clean_text)
        tb_feats = extract_textblob(clean_text)
        poly_feats = extract_politeness(clean_text)
        Qtext_rows.append(wiki_feats + vader_feats + tb_feats + poly_feats)

        # ── Qpersp features ──
        persp_row = []
        for col in QPERSP_COLS:
            s11_key = f"persp_q_{col.replace('q_persp_', '')}"
            persp_row.append(s11r.get(s11_key, 0) or 0)
        Qpersp_rows.append(persp_row)

        # ── Question type annotations (LLM) ──
        Qtype_rows.append(ann_data.get(cid, [0, 0, 0, 0, 0]))

        # ── Embedding index ──
        if cid in emb_cid_to_idx:
            emb_indices.append(emb_cid_to_idx[cid])
        else:
            emb_indices.append(-1)
            n_missing_emb += 1

        # ── Outcomes ──
        oc = build_outcomes(edits, Q, anchor, replied)
        for k in ALL_OC_KEYS:
            OC[k].append(oc.get(k, 0))

        # ── Pseudo-time robustness outcomes ──
        if replied and rts:
            for ps_name in PSEUDO_STRATS + ["multi"]:
                for k in ALL_OC_KEYS:
                    ALT_OC[ps_name][k].append(oc.get(k, 0))
        else:
            strat_lags = {"median": median_lag, "p25": lag_p25, "p75": lag_p75, "zero": 0.0}
            for ps_name, lag_val in strat_lags.items():
                anchor_alt = Q + timedelta(days=lag_val)
                oc_alt = build_outcomes(edits, Q, anchor_alt, replied)
                for k in ALL_OC_KEYS:
                    ALT_OC[ps_name][k].append(oc_alt.get(k, 0))
            multi_ocs = {k: [] for k in ALL_OC_KEYS}
            for _ in range(N_MULTI_DRAWS):
                lag_draw = float(multi_rng.choice(lag_array))
                anchor_alt = Q + timedelta(days=lag_draw)
                oc_draw = build_outcomes(edits, Q, anchor_alt, replied)
                for k in ALL_OC_KEYS:
                    multi_ocs[k].append(oc_draw.get(k, 0))
            for k in ALL_OC_KEYS:
                ALT_OC["multi"][k].append(float(np.mean(multi_ocs[k])))

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1:,}/{N:,}]", flush=True)

    print(f"\n  Built {len(cid_order):,} rows")
    print(f"  Missing edit data: {n_missing_edits:,}")
    print(f"  Missing embeddings: {n_missing_emb:,}")

    # ── 8. Process embeddings → PCA-20 ──────────────────────────────────────
    print("\n[8] Building PCA-20 embeddings...")
    valid_emb_mask = np.array(emb_indices) >= 0
    n_valid = valid_emb_mask.sum()
    print(f"  {n_valid:,}/{len(emb_indices):,} have embeddings")

    X_emb_full = np.zeros((len(emb_indices), emb_all.shape[1]), dtype=np.float32)
    for i, idx in enumerate(emb_indices):
        if idx >= 0:
            X_emb_full[i] = emb_all[idx]

    pca = PCA(n_components=20, random_state=42)
    X_emb20 = pca.fit_transform(X_emb_full)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA-20 variance explained: {var_explained:.3f}")

    # ── 9. Build temporal dummies ───────────────────────────────────────────
    print("[9] Building temporal controls...")
    ym_series = pd.Series(q_year_month_list)
    ym_dummies = pd.get_dummies(ym_series, prefix="ym", drop_first=True).values.astype(float)
    q_has_article_arr = np.array(q_has_article_list, dtype=float).reshape(-1, 1)
    X_temporal = np.hstack([ym_dummies, q_has_article_arr])
    ym_cols = [f"ym_{ym}" for ym in sorted(set(q_year_month_list))[1:]]
    temporal_cols = ym_cols + ["q_has_article_context"]
    print(f"  {ym_dummies.shape[1]} year-month dummies + 1 article_context = {X_temporal.shape[1]} cols")

    # ── 10. Build mentor features (for robustness only) ─────────────────────
    print("[10] Building mentor features (robustness)...")
    M_COLS = ["m_prior_mentee_count", "m_prior_reply_rate", "m_prior_avg_response_hours",
              "m_recent30d_mentee_count", "m_recent30d_reply_rate", "m_days_active"]
    M_rows = []
    for cid in cid_order:
        s11r = s11_data[cid]
        M_rows.append([s11r.get(c, 0) or 0 for c in M_COLS])
    X_M = np.array(M_rows, dtype=float)
    print(f"  {len(M_COLS)} cols")

    # ── 11. Convert to arrays ───────────────────────────────────────────────
    print("\n[11] Converting to arrays...")
    X_E = np.array(E_rows, dtype=float)
    X_Qtext = np.array(Qtext_rows, dtype=float)
    X_Qpersp = np.array(Qpersp_rows, dtype=float)
    X_Qtype = np.array(Qtype_rows, dtype=float)
    y_treat = np.array(y_treat_list, dtype=float)
    y_treat_48h = np.array(y_treat_48h_list, dtype=float)

    n_ann_available = int((X_Qtype.sum(axis=1) != 0).sum() + (X_Qtype.max(axis=1) == 0).sum())
    print(f"  X_Qtype: {X_Qtype.shape[1]} cols, {int(X_Qtype[:, 0].sum()):,} substantive (Q1=Y)")

    total_features = X_E.shape[1] + X_Qtext.shape[1] + X_Qpersp.shape[1] + X_Qtype.shape[1] + X_emb20.shape[1] + X_temporal.shape[1]

    # ── 12. Save ────────────────────────────────────────────────────────────
    print(f"\n[12] Saving to {OUT}...")
    out_path = OUT / "psm_dataset.npz"

    save_dict = {
        # Identifiers
        "cid_order": np.array(cid_order),
        "mentee_ids": np.array(mentee_ids),
        "mentor_ids": np.array(mentor_ids),
        "q_year_month": np.array(q_year_month_list),
        "q_has_article_context": np.array(q_has_article_list, dtype=float),
        "mentor_reply_rate": np.array(mentor_reply_rate_list, dtype=float),
        # Treatment
        "y_treat": y_treat,
        "y_treat_48h": y_treat_48h,
        # Feature matrices
        "X_E": X_E,
        "X_Qtext": X_Qtext,
        "X_Qpersp": X_Qpersp,
        "X_Qtype": X_Qtype,
        "X_emb20": X_emb20.astype(float),
        "X_emb_full": X_emb_full,
        "X_temporal": X_temporal,
        "X_M": X_M,
        # Column names
        "E_cols": np.array(E_COLS),
        "Qtext_cols": np.array(QTEXT_COLS),
        "Qpersp_cols": np.array(QPERSP_COLS),
        "Qtype_cols": np.array(QTYPE_COLS),
        "M_cols": np.array(M_COLS),
        "temporal_cols": np.array(temporal_cols),
    }

    # Outcomes
    for k in ALL_OC_KEYS:
        save_dict[f"oc_{k}"] = np.array(OC[k], dtype=float)

    # Pseudo-time robustness outcomes
    for ps_name in PSEUDO_STRATS + ["multi"]:
        for k in ALL_OC_KEYS:
            save_dict[f"oc_{k}__{ps_name}"] = np.array(ALT_OC[ps_name][k], dtype=float)
    save_dict["pseudo_lag_info"] = np.array([median_lag, lag_p25, lag_p75, 0.0, N_MULTI_DRAWS])

    np.savez_compressed(out_path, **save_dict)
    sz = out_path.stat().st_size / 1024 / 1024

    # ── 13. Summary ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SAVED: {out_path} ({sz:.1f} MB)")
    print(f"  N = {len(cid_order):,}")
    print(f"  Treated (ever replied):  {int(y_treat.sum()):,}")
    print(f"  Treated (48h):           {int(y_treat_48h.sum()):,}")
    print(f"  Control:                 {int(len(y_treat) - y_treat.sum()):,}")
    print(f"\n  Feature blocks:")
    print(f"    X_E (edit history):    {X_E.shape[1]:>3d} cols")
    print(f"    X_Qtext (text):        {X_Qtext.shape[1]:>3d} cols")
    print(f"    X_Qpersp (perspective):{X_Qpersp.shape[1]:>3d} cols")
    print(f"    X_Qtype (LLM annot.):  {X_Qtype.shape[1]:>3d} cols")
    print(f"    X_emb20 (PCA):         {X_emb20.shape[1]:>3d} cols")
    print(f"    X_temporal:            {X_temporal.shape[1]:>3d} cols")
    print(f"    ────────────────────────────")
    print(f"    TOTAL:                 {total_features:>3d} cols")
    print(f"    X_M (mentor, robust.): {X_M.shape[1]:>3d} cols")
    print(f"    X_emb_full:            {X_emb_full.shape[1]:>3d} cols")
    print(f"\n  Outcomes: {OC_KEYS}")
    print(f"  Window outcomes: {OC_WINDOW_KEYS}")

    for k in ["primary_constructive", "cross_day_constructive_14d", "reverted_any"]:
        arr = np.array(OC[k])
        print(f"    {k}: {arr.sum():.0f}/{len(arr)} = {arr.mean()*100:.1f}%")

    print(SEP)
    print("Done.\n")


if __name__ == "__main__":
    main()
