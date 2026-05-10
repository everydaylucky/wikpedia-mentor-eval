#!/usr/bin/env python3
"""
s8_extract_first_turns.py — Extract each mentee's first question + mentor's first reply.

For each conversation in data/s7/s7_conversations_cleaned.jsonl, outputs one record with:
  - mentee first question (clean + emb text)
  - mentor first reply (clean + emb text)
  - metadata (mentor, mentee, timestamp, article, is_first_conversation)
  - is_english flag based on ASCII letter ratio of question text
  - reply_signer: who actually signed the reply (assigned_mentor / other_only / etc.)

Output: data/s8/s8_first_turns.jsonl
"""
import json, re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE / "data" / "s7" / "s7_conversations_cleaned.jsonl"
OUTPUT = BASE / "data" / "s8" / "s8_first_turns.jsonl"


def is_english(text, threshold=0.8):
    """Check if text is primarily English by ASCII letter ratio."""
    if not text or len(text.strip()) < 3:
        return True
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    all_alpha = sum(1 for c in text if c.isalpha())
    if all_alpha == 0:
        return True
    return (ascii_alpha / all_alpha) >= threshold


TS_RE = re.compile(r'\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)', re.I)
USER_LINK_RE = re.compile(r'\[\[User(?:[_ ]talk)?:([^\]|#]+)[^\]]*\]\]', re.I)


def extract_signers(raw):
    """Extract the signer (last [[User:]] before each timestamp) from wikitext."""
    signers = set()
    for ts_match in TS_RE.finditer(raw):
        ts_start = ts_match.start()
        window = raw[max(0, ts_start - 500):ts_start]
        user_matches = list(USER_LINK_RE.finditer(window))
        if user_matches:
            signer = user_matches[-1].group(1).strip().replace("_", " ").lower()
            signers.add(signer)
    return signers


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data/s7/s7_conversations_cleaned.jsonl...")
    records = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Total conversations: {len(records):,}")

    # Determine is_first_conversation per mentee (by timestamp)
    mentee_first_ts = {}
    for r in records:
        m = r["mentee"]
        ts = r["timestamp"]
        if m not in mentee_first_ts or (ts and ts < mentee_first_ts[m]):
            mentee_first_ts[m] = ts

    mentee_first_cid = {}
    for r in records:
        m = r["mentee"]
        ts = r["timestamp"]
        if ts == mentee_first_ts.get(m) and m not in mentee_first_cid:
            mentee_first_cid[m] = r["conversation_id"]

    out = []
    for r in records:
        is_first = (r["conversation_id"] == mentee_first_cid.get(r["mentee"]))
        q_eng = is_english(r["question_emb"])

        # Determine reply signer + actual responder names
        reply_signer = "none"
        actual_responders = []
        if r["has_reply"] and r.get("reply_raw"):
            raw = r["reply_raw"]
            mentor_norm = r["mentor"].strip().replace("_", " ").lower()
            mentee_norm = r["mentee"].strip().replace("_", " ").lower()
            signers = extract_signers(raw)
            others = signers - {mentor_norm, mentee_norm}
            has_mentor = mentor_norm in signers
            has_other = bool(others)
            if has_mentor and has_other:
                reply_signer = "mentor_and_other"
            elif has_mentor:
                reply_signer = "assigned_mentor"
            elif has_other:
                reply_signer = "other_only"
            else:
                reply_signer = "unknown"
            actual_responders = sorted(signers - {mentee_norm})

        # reply_signer=unknown means no identifiable mentor/other replied
        # (typically mentee self-reply or parsing artifact) — treat as no reply
        if reply_signer == "unknown":
            r = dict(r, has_reply=False, reply_clean="", reply_emb="",
                     reply_words=0, reply_timestamp=None)
            reply_signer = "none"
            actual_responders = []

        rec = {
            "conversation_id": r["conversation_id"],
            "mentor": r["mentor"],
            "mentee": r["mentee"],
            "revid": r["revid"],
            "timestamp": r["timestamp"],
            "article": r["article"],
            "source": r["source"],
            "page": r["page"],
            "is_first_conversation": is_first,
            "is_english": q_eng,
            "reply_signer": reply_signer,
            "actual_responders": actual_responders,
            "question_clean": r["question_clean"],
            "question_emb": r["question_emb"],
            "question_words": r["question_words"],
            "reply_timestamp": r.get("reply_timestamp"),
            "has_reply": r["has_reply"],
            "reply_clean": r["reply_clean"],
            "reply_emb": r["reply_emb"],
            "reply_words": r["reply_words"],
        }
        out.append(rec)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(out)
    first_only = [r for r in out if r["is_first_conversation"]]
    has_reply = sum(1 for r in first_only if r["has_reply"])
    first_eng = sum(1 for r in first_only if r["is_english"])
    first_eng_reply = sum(1 for r in first_only if r["is_english"] and r["has_reply"])
    empty_q = sum(1 for r in first_only if not r["question_emb"])

    signer_dist = Counter(r["reply_signer"] for r in first_only if r["has_reply"])

    print(f"\n  Results:")
    print(f"    Total conversations:       {total:,}")
    print(f"    First conversations:       {len(first_only):,}")
    print(f"    English first convs:       {first_eng:,}")
    print(f"    Non-English:               {len(first_only) - first_eng:,}")
    print(f"    With mentor reply:         {has_reply:,}")
    print(f"    English + has reply:       {first_eng_reply:,}")
    print(f"    Empty question (emb):      {empty_q:,}")
    print(f"\n  Reply signer (first convs with reply):")
    for k, v in signer_dist.most_common():
        print(f"    {k:25s}: {v:>6,} ({v/has_reply*100:.1f}%)")
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
