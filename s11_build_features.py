#!/usr/bin/env python3
"""
s11_build_features.py — Build analysis-ready feature matrix for PSM.

Reads: s1 (mentors), s5 (conversations), s7 (cleaned text), s8 (first turns),
       s6 (profiles, contribs, logevents, abuselog), s10 (perspective API).

Outputs:
  data/s11/s11_features.jsonl  — one row per conversation (first-turn, English, auto-only)

Usage:
  python s11_build_features.py                         # default: use 0wiki-mentor-github s6 data
  python s11_build_features.py --s6-dir data/s6        # use local s6 data
  python s11_build_features.py --sample 1000           # quick test with 1000 rows
"""
import argparse, csv, datetime, json, math, os, re, sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
OUT_DIR = DATA / "s11"

S1_FILE = DATA / "s1" / "s1_mentor_list.jsonl"
S5_FILE = DATA / "s5" / "s5_all_conversations.jsonl"
S7_FILE = DATA / "s7" / "s7_conversations_cleaned.jsonl"
S8_FILE = DATA / "s8" / "s8_first_turns.jsonl"
S10_MENTEE = DATA / "s10" / "s10_perspective_mentee.csv"
S10_MENTOR = DATA / "s10" / "s10_perspective_mentor.csv"

# Default external s6 data (already collected)
DEFAULT_S6_DIR = Path("/Users/Shared/baiduyun/00 Code/0Wiki/0wiki-mentor-github")

DT = datetime.datetime
TD = datetime.timedelta

PERSPECTIVE_ATTRS = [
    "TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT",
    "SEXUALLY_EXPLICIT", "FLIRTATION",
    "AFFINITY_EXPERIMENTAL", "COMPASSION_EXPERIMENTAL", "CURIOSITY_EXPERIMENTAL",
    "NUANCE_EXPERIMENTAL", "PERSONAL_STORY_EXPERIMENTAL", "REASONING_EXPERIMENTAL",
    "RESPECT_EXPERIMENTAL",
    "ATTACK_ON_AUTHOR", "ATTACK_ON_COMMENTER", "INCOHERENT", "INFLAMMATORY",
    "LIKELY_TO_REJECT", "OBSCENE", "SPAM", "UNSUBSTANTIAL",
]

WINDOWS = [1, 7, 14, 30, 60]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_ts(s):
    if not s:
        return None
    try:
        return DT.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def safe_div(a, b, default=0):
    return a / b if b > 0 else default


def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def sentence_count(text):
    if not text:
        return 0
    return max(1, len(re.split(r'[.!?]+', text.strip())) - 1) if text.strip() else 0


# ── Text Feature Extraction ─────────────────────────────────────────────────

def extract_text_features(text, prefix="q"):
    """Extract features from question_clean or reply_clean."""
    f = {}
    if not text:
        for k in [f"{prefix}_words", f"{prefix}_chars", f"{prefix}_sentences",
                  f"{prefix}_avg_word_len", f"{prefix}_type_token_ratio",
                  f"{prefix}_n_question_marks", f"{prefix}_n_exclamation",
                  f"{prefix}_has_question_mark", f"{prefix}_has_greeting",
                  f"{prefix}_has_thanks", f"{prefix}_has_apology",
                  f"{prefix}_has_frustration", f"{prefix}_has_self_intro",
                  f"{prefix}_has_urgency", f"{prefix}_n_policy", f"{prefix}_n_help",
                  f"{prefix}_n_wikilink", f"{prefix}_n_link", f"{prefix}_n_draft",
                  f"{prefix}_mentions_deletion", f"{prefix}_mentions_revert",
                  f"{prefix}_mentions_notability", f"{prefix}_mentions_copyright",
                  f"{prefix}_mentions_draft", f"{prefix}_mentions_protection",
                  f"{prefix}_mentions_conflict", f"{prefix}_n_paragraphs",
                  f"{prefix}_avg_sentence_len", f"{prefix}_is_single_sentence",
                  f"{prefix}_has_list"]:
            f[k] = 0
        return f

    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    n_sents = sentence_count(text)
    unique_words = set(w.lower() for w in words)
    lower = text.lower()

    f[f"{prefix}_words"] = n_words
    f[f"{prefix}_chars"] = n_chars
    f[f"{prefix}_sentences"] = n_sents
    f[f"{prefix}_avg_word_len"] = safe_div(sum(len(w) for w in words), n_words, 0)
    f[f"{prefix}_type_token_ratio"] = safe_div(len(unique_words), n_words, 0)
    f[f"{prefix}_avg_sentence_len"] = safe_div(n_words, max(n_sents, 1), 0)
    f[f"{prefix}_is_single_sentence"] = int(n_sents <= 1)
    f[f"{prefix}_n_paragraphs"] = max(1, text.count("\n\n") + 1)

    f[f"{prefix}_n_question_marks"] = text.count("?")
    f[f"{prefix}_n_exclamation"] = text.count("!")
    f[f"{prefix}_has_question_mark"] = int("?" in text)

    f[f"{prefix}_has_greeting"] = int(bool(re.match(
        r"^(hi|hello|hey|dear|good\s+(morning|afternoon|evening)|greetings)\b", lower)))
    f[f"{prefix}_has_thanks"] = int(bool(re.search(
        r"\b(thank|thanks|grateful|appreciate)\b", lower)))
    f[f"{prefix}_has_apology"] = int(bool(re.search(
        r"\b(sorry|apologi[sz]e|excuse me|pardon)\b", lower)))
    f[f"{prefix}_has_frustration"] = int(bool(re.search(
        r"\b(confused|frustrat|don'?t understand|why was my|what happened to)\b", lower)))
    f[f"{prefix}_has_self_intro"] = int(bool(re.search(
        r"\b(i'?m new|new to|first time|beginner|newbie|newcomer|just started)\b", lower)))
    f[f"{prefix}_has_urgency"] = int(bool(re.search(
        r"\b(urgent|asap|please help|need help|help me)\b", lower)))
    f[f"{prefix}_has_list"] = int(bool(re.search(r"^\s*[\*\-\#\d]", text, re.M)))

    # Semantic token counts (from s7 cleaning)
    f[f"{prefix}_n_policy"] = text.count("[POLICY]")
    f[f"{prefix}_n_help"] = text.count("[HELP_PAGE]")
    f[f"{prefix}_n_wikilink"] = text.count("[WIKILINK]")
    f[f"{prefix}_n_link"] = text.count("[LINK]")
    f[f"{prefix}_n_draft"] = text.count("[DRAFT]")

    # Topic signals
    f[f"{prefix}_mentions_deletion"] = int(bool(re.search(
        r"\b(delet|speedy|[ac]fd|csd|proposed deletion|prod)\b", lower)))
    f[f"{prefix}_mentions_revert"] = int(bool(re.search(
        r"\b(revert|undone|rolled? back|undo)\b", lower)))
    f[f"{prefix}_mentions_notability"] = int(bool(re.search(
        r"\b(notab|GNG|significant coverage|reliable sources)\b", lower)))
    f[f"{prefix}_mentions_copyright"] = int(bool(re.search(
        r"\b(copyright|copyvio|plagiari|fair use)\b", lower)))
    f[f"{prefix}_mentions_draft"] = int(bool(re.search(
        r"\b(draft|sandbox|userspace)\b", lower)))
    f[f"{prefix}_mentions_protection"] = int(bool(re.search(
        r"\b(protect|semi.?protect|autoconfirm|edit.?request)\b", lower)))
    f[f"{prefix}_mentions_conflict"] = int(bool(re.search(
        r"\b(dispute|edit.?war|consensus|3rr|revert.?war)\b", lower)))

    return f


# ── Windowed Edit Features ───────────────────────────────────────────────────

def build_edit_features(edits, Q, windows=WINDOWS):
    """Build pre-Q windowed features from a user's edit list."""
    pre = [e for e in edits if e["ts"] < Q]
    f = {}

    for w in windows:
        cutoff = Q - TD(days=w)
        we = [e for e in pre if e["ts"] >= cutoff]
        suffix = f"_{w}d"
        n = len(we)
        f[f"n_edits{suffix}"] = n
        ns_counts = Counter(e["ns"] for e in we)
        f[f"n_mainspace{suffix}"] = ns_counts.get(0, 0)
        f[f"n_usertalk{suffix}"] = ns_counts.get(3, 0)
        f[f"n_draft{suffix}"] = ns_counts.get(118, 0)
        f[f"n_unique_ns{suffix}"] = len(ns_counts)
        f[f"mainspace_ratio{suffix}"] = safe_div(ns_counts.get(0, 0), n)
        f[f"draft_ratio{suffix}"] = safe_div(ns_counts.get(118, 0), n)
        sds = [e["sizediff"] for e in we]
        f[f"avg_sizediff{suffix}"] = safe_div(sum(sds), n) if sds else 0
        f[f"neg_sizediff_ratio{suffix}"] = safe_div(sum(1 for s in sds if s < 0), n)
        f[f"n_unique_articles{suffix}"] = len(set(e.get("pageid", e.get("title", "")) for e in we))

    # All-time pre-Q
    n = len(pre)
    f["n_edits_all"] = n
    ns_all = Counter(e["ns"] for e in pre)
    f["n_unique_ns_all"] = len(ns_all)
    f["mainspace_ratio_all"] = safe_div(ns_all.get(0, 0), n)
    f["draft_ratio_all"] = safe_div(ns_all.get(118, 0), n)

    sds_all = [e["sizediff"] for e in pre]
    f["avg_sizediff_all"] = safe_div(sum(sds_all), n) if sds_all else 0
    f["std_sizediff_all"] = (sum((s - f["avg_sizediff_all"])**2 for s in sds_all) / n)**0.5 if n > 1 else 0
    f["max_abs_sizediff_all"] = max((abs(s) for s in sds_all), default=0)
    f["neg_sizediff_ratio_all"] = safe_div(sum(1 for s in sds_all if s < 0), n)

    if pre:
        f["hours_since_last_edit"] = (Q - pre[-1]["ts"]).total_seconds() / 3600
        f["active_span_hours"] = (pre[-1]["ts"] - pre[0]["ts"]).total_seconds() / 3600 if n > 1 else 0
    else:
        f["hours_since_last_edit"] = -1
        f["active_span_hours"] = 0

    # Time pattern features
    f["edit_rate_7d"] = f["n_edits_7d"] / 7.0
    f["edit_rate_30d"] = f["n_edits_30d"] / 30.0

    if pre:
        hours = [e["ts"].hour for e in pre]
        weekdays = [e["ts"].weekday() for e in pre]
        f["weekend_ratio"] = safe_div(sum(1 for d in weekdays if d >= 5), n)
        f["night_ratio"] = safe_div(sum(1 for h in hours if h < 6), n)
        hour_counts = Counter(hours)
        f["hour_entropy"] = entropy(hour_counts)
        edit_days = set(e["ts"].date() for e in pre)
        f["burstiness"] = safe_div(n, len(edit_days)) if edit_days else 0
    else:
        f["weekend_ratio"] = 0
        f["night_ratio"] = 0
        f["hour_entropy"] = 0
        f["burstiness"] = 0

    # Tag features (all pre-Q)
    tag_counts = Counter()
    for e in pre:
        for t in e.get("tags", []):
            tag_counts[t] += 1

    tag_map = {
        "tag_visualeditor": "visualeditor",
        "tag_wikieditor": "wikieditor",
        "tag_mobile": "mobile edit",
        "tag_mobile_web": "mobile web edit",
        "tag_mobile_app": "mobile app edit",
        "tag_newcomer_task": "newcomer task",
        "tag_newcomer_copyedit": "newcomer task copyedit",
        "tag_newcomer_addlink": "newcomer task add link",
        "tag_newcomer_references": "newcomer task references",
        "tag_newcomer_expand": "newcomer task expand",
        "tag_newcomer_update": "newcomer task update",
        "tag_newcomer_links": "newcomer task links",
        "tag_newcomer_revisetone": "newcomer task revise tone",
        "tag_editcheck_newref": "editcheck-newreference",
        "tag_editcheck_newcontent": "editcheck-newcontent",
        "tag_editcheck_references": "editcheck-references",
        "tag_editcheck_tone": "editcheck-tone",
        "tag_mw_reverted": "mw-reverted",
        "tag_discussion": "discussiontools-added-comment",
        "tag_app_suggestededit": "app-suggestededit",
    }
    for feat_name, tag_name in tag_map.items():
        f[feat_name] = tag_counts.get(tag_name, 0)

    # Revert features
    n_reverted = sum(1 for e in pre if "mw-reverted" in e.get("tags", []))
    f["n_reverts_pre"] = n_reverted
    f["revert_rate_pre"] = safe_div(n_reverted, n)

    return f


# ── Y Variables ──────────────────────────────────────────────────────────────

def build_outcomes(edits, Q, reg_dt=None):
    """Build all outcome variables from post-Q edits."""
    y = {}
    all_edits = edits
    post_q = [e for e in all_edits if e["ts"] > Q]
    post_q_ms = [e for e in post_q if e["ns"] % 2 == 0]  # content namespaces

    # ── Retention (windowed) ──
    for w_start, w_end, name in [
        (7, 14, "ret_7d"), (7, 21, "ret_14d"), (7, 37, "ret_30d"),
        (7, 67, "ret_60d"), (21, 35, "ret_14_28d"),
    ]:
        ws = Q + TD(days=w_start)
        we = Q + TD(days=w_end)
        window_edits = [e for e in post_q_ms if ws <= e["ts"] < we]
        y[name] = int(len(window_edits) >= 1)
        y[f"{name}_count"] = len(window_edits)
        y[f"{name}_5plus"] = int(len(window_edits) >= 5)
        active_days = len(set(e["ts"].date() for e in window_edits))
        y[f"{name}_active_days"] = active_days
        y[f"{name}_crossday"] = int(active_days >= 2)

    # ── WMF retention ──
    if reg_dt:
        wmf_edits = [e for e in all_edits if reg_dt + TD(days=30) <= e["ts"] < reg_dt + TD(days=60)]
        y["ret_wmf"] = int(len(wmf_edits) >= 1)
        y["ret_wmf_valid"] = int((Q - reg_dt).total_seconds() / 86400 < 30)
    else:
        y["ret_wmf"] = 0
        y["ret_wmf_valid"] = 0

    # ── Continuous retention ──
    y["log1p_edits_7_37"] = math.log1p(y["ret_30d_count"])

    if post_q_ms:
        y["days_to_first_post_edit"] = min(60, (post_q_ms[0]["ts"] - Q).total_seconds() / 86400)
    else:
        y["days_to_first_post_edit"] = 60  # censored

    if len(post_q_ms) >= 5:
        sorted_ms = sorted(post_q_ms, key=lambda e: e["ts"])
        y["days_to_5th"] = min(37, (sorted_ms[4]["ts"] - Q).total_seconds() / 86400)
    else:
        y["days_to_5th"] = 37  # censored

    if post_q_ms:
        y["last_edit_days_post_q"] = (post_q_ms[-1]["ts"] - Q).total_seconds() / 86400
    else:
        y["last_edit_days_post_q"] = 0

    y["any_edit_post"] = int(len(post_q) >= 1)

    # Session count (distinct weeks in 60d)
    if post_q_ms:
        weeks = set((e["ts"] - Q).days // 7 for e in post_q_ms if (e["ts"] - Q).days < 60)
        y["session_weeks_60d"] = len(weeks)
    else:
        y["session_weeks_60d"] = 0

    # Streak: max consecutive active days
    if post_q_ms:
        active_dates = sorted(set(e["ts"].date() for e in post_q_ms
                                   if (e["ts"] - Q).days < 60))
        max_streak = 1
        cur_streak = 1
        for i in range(1, len(active_dates)):
            if (active_dates[i] - active_dates[i-1]).days == 1:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 1
        y["streak_max_60d"] = max_streak
    else:
        y["streak_max_60d"] = 0

    # Returned after gap
    if post_q_ms:
        active_dates = sorted(set(e["ts"].date() for e in post_q_ms))
        has_gap_return = False
        for i in range(1, len(active_dates)):
            if (active_dates[i] - active_dates[i-1]).days >= 7:
                has_gap_return = True
                break
        y["returned_after_gap"] = int(has_gap_return)
    else:
        y["returned_after_gap"] = 0

    # ── Quality ──
    w_start_q = Q + TD(days=7)
    w_end_q = Q + TD(days=37)
    window_ms = [e for e in post_q_ms if w_start_q <= e["ts"] < w_end_q]
    n_reverted = sum(1 for e in window_ms if "mw-reverted" in e.get("tags", []))
    y["reverted_any"] = int(n_reverted >= 1)
    y["reverted_count"] = n_reverted
    y["revert_rate_post"] = safe_div(n_reverted, len(window_ms))
    y["productive_edits"] = len(window_ms) - n_reverted

    # ── Diversity ──
    if window_ms:
        ns_counts = Counter(e["ns"] for e in window_ms)
        y["unique_ns"] = len(ns_counts)
        y["ns_entropy"] = entropy(ns_counts)
        y["unique_articles"] = len(set(e.get("pageid", e.get("title", "")) for e in window_ms))
        y["mainspace_ratio_post"] = safe_div(ns_counts.get(0, 0), len(window_ms))
    else:
        y["unique_ns"] = 0
        y["ns_entropy"] = 0
        y["unique_articles"] = 0
        y["mainspace_ratio_post"] = 0

    return y


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s6-dir", type=str, default=str(DEFAULT_S6_DIR))
    ap.add_argument("--sample", type=int, default=0, help="Process only N rows for testing")
    args = ap.parse_args()
    s6_dir = Path(args.s6_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "s11_features.jsonl"

    print("=" * 70)
    print("  s11_build_features.py — Feature Construction")
    print("=" * 70)

    # ── 1. Load s1 mentor timelines ──
    print("\n[1] Loading s1 mentor timelines...")
    mentor_timelines = {}
    with open(S1_FILE, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            periods = []
            for entry in (m.get("weight_history") or []):
                if entry["weight"] is None:
                    continue
                ts = parse_ts(entry["timestamp"])
                if ts:
                    periods.append((ts, entry["weight"]))
            periods.sort(key=lambda x: x[0])
            mentor_timelines[m["username"]] = periods
    print(f"  {len(mentor_timelines)} mentors")

    def get_mentor_type(mentor, timestamp):
        tl = mentor_timelines.get(mentor, [])
        if not tl:
            return "unknown"
        conv_dt = parse_ts(timestamp) if isinstance(timestamp, str) else timestamp
        if not conv_dt:
            return "unknown"
        dates = [p[0] for p in tl]
        idx = bisect_right(dates, conv_dt) - 1
        if idx < 0:
            return "unknown"
        return "auto" if tl[idx][1] >= 1 else "manual"

    # ── 2. Load s8 first turns ──
    print("[2] Loading s8 first turns...")
    records = []
    with open(S8_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("is_first_conversation"):
                continue
            if not r.get("is_english"):
                continue
            mt = get_mentor_type(r["mentor"], r["timestamp"])
            if mt != "auto":
                continue
            r["_mentor_type"] = mt
            records.append(r)
    print(f"  {len(records):,} English first-turn auto conversations")

    if args.sample:
        records = records[:args.sample]
        print(f"  Sampled to {len(records):,}")

    # ── 3. Load s5 all conversations (for mentor prior features) ──
    print("[3] Loading s5 all conversations (for mentor features)...")
    mentor_convs = defaultdict(list)
    with open(S5_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            ts = parse_ts(c.get("timestamp"))
            if ts:
                mentor_convs[c["mentor"]].append({
                    "ts": ts,
                    "mentee": c["mentee"],
                    "has_reply": bool(c.get("mentor_reply")),
                })
    for m in mentor_convs:
        mentor_convs[m].sort(key=lambda x: x["ts"])
    print(f"  {len(mentor_convs)} mentors, {sum(len(v) for v in mentor_convs.values()):,} conversations")

    # ── 4. Load s7 cleaned (for reply timestamps) ──
    print("[4] Loading s7 cleaned (for reply lag)...")
    s7_data = {}
    with open(S7_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s7_data[r["conversation_id"]] = {
                "reply_timestamp": r.get("reply_timestamp"),
            }
    print(f"  {len(s7_data):,} records")

    # ── 5. Load s6 user data ──
    print("[5] Loading s6 user profiles...")
    user_profiles = {}
    prof_path = s6_dir / "s6_user_profiles.jsonl"
    if prof_path.exists():
        with open(prof_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                user_profiles[r["username"]] = r
    print(f"  {len(user_profiles):,} profiles")

    print("[5b] Loading s6 user contributions (this may take a minute)...")
    user_edits = {}
    contribs_path = s6_dir / "s6_user_contribs.jsonl"
    # Collect which users we actually need
    needed_users = set()
    for r in records:
        needed_users.add(r["mentee"])
    print(f"  Need contributions for {len(needed_users):,} mentees")

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
                            "ts": ts,
                            "ns": e.get("ns", 0),
                            "sizediff": e.get("sizediff", 0),
                            "tags": e.get("tags", []),
                            "pageid": e.get("revid", ""),
                            "title": e.get("title", ""),
                        })
                edits.sort(key=lambda x: x["ts"])
                user_edits[uname] = edits
                loaded += 1
                if loaded % 5000 == 0:
                    print(f"    loaded {loaded:,}...", flush=True)
        print(f"  {len(user_edits):,} users with contributions")
    else:
        print(f"  WARNING: {contribs_path} not found!")

    print("[5c] Loading s6 logevents...")
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
                        events.append({
                            "ts": ts,
                            "type": e.get("type", ""),
                            "action": e.get("action", ""),
                        })
                user_logs[uname] = events
    print(f"  {len(user_logs):,} users with logevents")

    print("[5d] Loading s6 abuselog...")
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
                        entries.append({
                            "ts": ts,
                            "result": e.get("result", ""),
                        })
                user_abuse[uname] = entries
    print(f"  {len(user_abuse):,} users with abuselog")

    # ── 6. Load s10 perspective scores ──
    print("[6] Loading s10 perspective scores...")
    persp_mentee = {}
    if S10_MENTEE.exists():
        with open(S10_MENTEE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = int(row["conversation_id"])
                scores = {}
                for attr in PERSPECTIVE_ATTRS:
                    v = row.get(attr, "")
                    scores[attr] = float(v) if v else None
                persp_mentee[cid] = scores
    print(f"  Mentee perspective: {len(persp_mentee):,}")

    persp_mentor = {}
    if S10_MENTOR.exists():
        with open(S10_MENTOR, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = int(row["conversation_id"])
                scores = {}
                for attr in PERSPECTIVE_ATTRS:
                    v = row.get(attr, "")
                    scores[attr] = float(v) if v else None
                persp_mentor[cid] = scores
    print(f"  Mentor perspective: {len(persp_mentor):,}")

    # ── 7. Compute mentor typical reply hour ──
    print("[7] Computing mentor typical reply hour...")
    mentor_reply_hours = defaultdict(list)
    for cid, s7r in s7_data.items():
        rts = parse_ts(s7r.get("reply_timestamp"))
        if rts:
            # We need to know which mentor — look up from s5
            pass  # will compute inline

    # ── 8. Build features ──
    print(f"\n[8] Building features for {len(records):,} conversations...")
    out_rows = []
    n_missing_edits = 0

    for i, rec in enumerate(records):
        cid = rec["conversation_id"]
        mentee = rec["mentee"]
        mentor = rec["mentor"]
        Q = parse_ts(rec["timestamp"])
        if not Q:
            continue

        row = {
            "conversation_id": cid,
            "mentee": mentee,
            "mentor": mentor,
            "timestamp": rec["timestamp"],
            "mentor_type": rec["_mentor_type"],
            "has_reply": int(rec.get("has_reply", False)),
            "reply_signer": rec.get("reply_signer", ""),
        }

        # ── Reply lag ──
        reply_ts = parse_ts(rec.get("reply_timestamp"))
        if reply_ts and Q:
            reply_lag_hours = (reply_ts - Q).total_seconds() / 3600
            row["reply_lag_hours"] = reply_lag_hours
            row["replied_7d"] = int(reply_lag_hours <= 168 and rec.get("has_reply"))
            row["replied_48h"] = int(reply_lag_hours <= 48 and rec.get("has_reply"))
        else:
            row["reply_lag_hours"] = None
            row["replied_7d"] = 0
            row["replied_48h"] = 0

        # ── Account age ──
        prof = user_profiles.get(mentee, {})
        reg_dt = parse_ts(prof.get("registration"))
        if reg_dt and Q:
            row["account_age_hours"] = (Q - reg_dt).total_seconds() / 3600
        else:
            row["account_age_hours"] = -1

        # ── Mentee edit features (windowed) ──
        edits = user_edits.get(mentee, [])
        if not edits:
            n_missing_edits += 1
        edit_feats = build_edit_features(edits, Q)
        row.update(edit_feats)

        # ── Mentee log features ──
        pre_logs = [l for l in user_logs.get(mentee, []) if l["ts"] < Q]
        log_ta = Counter()
        for l in pre_logs:
            log_ta[l["type"] + "/" + l["action"]] += 1
        row["log_create"] = log_ta.get("newusers/create", 0) + log_ta.get("create/create", 0)
        row["log_delete"] = log_ta.get("delete/delete", 0)
        row["log_block"] = log_ta.get("block/block", 0)
        row["log_move"] = log_ta.get("move/move", 0)
        row["log_thanks"] = log_ta.get("thanks/thank", 0)

        # ── Mentee abuse features ──
        pre_abuse = [a for a in user_abuse.get(mentee, []) if a["ts"] < Q]
        row["n_abuse"] = len(pre_abuse)
        row["n_abuse_warn"] = sum(1 for a in pre_abuse if "warn" in a.get("result", "").lower())
        row["n_abuse_disallow"] = sum(1 for a in pre_abuse if "disallow" in a.get("result", "").lower())

        # ── Mentor features (pre-Q) ──
        m_convs = mentor_convs.get(mentor, [])
        m_pre = [c for c in m_convs if c["ts"] < Q]
        row["m_prior_mentee_count"] = len(set(c["mentee"] for c in m_pre))
        m_replied = sum(1 for c in m_pre if c["has_reply"])
        row["m_prior_reply_rate"] = safe_div(m_replied, len(m_pre))
        row["m_days_active"] = (Q - m_pre[0]["ts"]).days if m_pre else 0

        m_30d = [c for c in m_pre if (Q - c["ts"]).days <= 30]
        row["m_recent30d_mentee_count"] = len(set(c["mentee"] for c in m_30d))
        row["m_recent30d_reply_count"] = sum(1 for c in m_30d if c["has_reply"])
        row["m_recent30d_reply_rate"] = safe_div(row["m_recent30d_reply_count"], len(m_30d))

        # Mentor avg response hours (from s7 reply timestamps of prior convos)
        # Simplified: use pre-computed if available, else skip
        row["m_prior_avg_response_hours"] = 0  # TODO: compute from s7 data

        # ── Time/matching features ──
        row["q_weekday"] = Q.weekday()
        row["q_hour_utc"] = Q.hour
        row["q_is_weekend"] = int(Q.weekday() >= 5)
        row["q_year"] = Q.year
        row["q_quarter"] = f"{Q.year}Q{(Q.month - 1) // 3 + 1}"
        row["q_month"] = Q.month

        # ── Text features (question) ──
        q_text_feats = extract_text_features(rec.get("question_clean", ""), prefix="q")
        row.update(q_text_feats)

        # ── Text features (reply) ──
        r_text_feats = extract_text_features(rec.get("reply_clean", ""), prefix="r")
        row.update(r_text_feats)

        # ── Perspective API features ──
        persp_q = persp_mentee.get(cid, {})
        for attr in PERSPECTIVE_ATTRS:
            row[f"persp_q_{attr.lower()}"] = persp_q.get(attr)

        persp_r = persp_mentor.get(cid, {})
        for attr in PERSPECTIVE_ATTRS:
            row[f"persp_r_{attr.lower()}"] = persp_r.get(attr)

        # ── Y variables ──
        outcomes = build_outcomes(edits, Q, reg_dt)
        row.update(outcomes)

        out_rows.append(row)

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1:,}/{len(records):,}]", flush=True)

    print(f"\n  Built {len(out_rows):,} rows")
    print(f"  Missing edit data: {n_missing_edits:,} mentees")

    # ── 9. Write output ──
    print(f"\n[9] Writing {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, default=str) + "\n")
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"  Wrote {len(out_rows):,} rows ({sz:.1f} MB)")

    # ── 10. Summary stats ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total rows:          {len(out_rows):,}")

    n_treated = sum(1 for r in out_rows if r.get("replied_7d"))
    print(f"  Treated (replied≤7d):{n_treated:,}")
    print(f"  Control:             {len(out_rows) - n_treated:,}")

    n_has_edits = sum(1 for r in out_rows if r.get("n_edits_all", 0) > 0)
    print(f"  Has pre-Q edits:     {n_has_edits:,}")

    n_persp = sum(1 for r in out_rows if r.get("persp_q_toxicity") is not None)
    print(f"  Has perspective (q): {n_persp:,}")

    for y_name in ["ret_14d", "ret_30d", "ret_60d", "ret_wmf", "reverted_any"]:
        vals = [r.get(y_name, 0) for r in out_rows]
        print(f"  {y_name}: {sum(vals):,}/{len(vals):,} = {sum(vals)/len(vals)*100:.1f}%")

    # Feature count
    sample = out_rows[0]
    meta_keys = {"conversation_id", "mentee", "mentor", "timestamp", "mentor_type",
                 "reply_signer", "q_quarter"}
    y_keys = {k for k in sample if k.startswith("ret_") or k.startswith("reverted")
              or k in ("log1p_edits_7_37", "days_to_first_post_edit", "days_to_5th",
                       "last_edit_days_post_q", "any_edit_post", "session_weeks_60d",
                       "streak_max_60d", "returned_after_gap", "revert_rate_post",
                       "productive_edits", "unique_ns", "ns_entropy", "unique_articles",
                       "mainspace_ratio_post")}
    x_keys = set(sample.keys()) - meta_keys - y_keys
    print(f"\n  X features: {len(x_keys)}")
    print(f"  Y outcomes:  {len(y_keys)}")
    print(f"  Meta fields: {len(meta_keys)}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
