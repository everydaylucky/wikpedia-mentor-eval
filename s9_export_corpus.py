#!/usr/bin/env python3
"""
s9_export_corpus.py — Export cleaned first-turn data to ConvoKit corpus format.

Data cleaning before export:
  - Drop empty mentee / empty question records
  - Fix replies with negative response time (>5 min) → treat as no reply
  - Fix empty reply_emb with has_reply=True → treat as no reply

Adds mentor_type (auto/manual) from s1 weight_history at conversation time.

ConvoKit corpus structure:
  wiki-mentor-corpus/
    utterances.jsonl   — one JSON per utterance (question or reply)
    speakers.json      — speaker metadata
    conversations.json — conversation metadata
    corpus.json        — corpus-level metadata
    index.json         — field type index

Usage:
  python s9_export_corpus.py                    # all English first convs
  python s9_export_corpus.py --assigned-only     # only assigned_mentor replies
"""
import argparse, datetime, json, os
from bisect import bisect_right
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE / "data" / "s8" / "s8_first_turns.jsonl"
S1_FILE = BASE / "data" / "s1" / "s1_mentor_list.jsonl"
OUT_DIR = BASE / "wiki-mentor-corpus"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assigned-only", action="store_true",
                    help="Only include replies from assigned mentor")
    ap.add_argument("--output", type=str, default=None,
                    help="Output directory (default: wiki-mentor-corpus)")
    args = ap.parse_args()

    out_dir = Path(args.output) if args.output else OUT_DIR
    out_dir.mkdir(exist_ok=True)

    print("Loading data/s8/s8_first_turns.jsonl...")
    records = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    eng_first = [r for r in records if r["is_first_conversation"] and r["is_english"]]
    print(f"  English first conversations: {len(eng_first):,}")

    # ── Load mentor assignment type (auto/manual) from weight_history ──
    mentor_timelines = {}
    with open(S1_FILE, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            periods = []
            for entry in (m.get("weight_history") or []):
                if entry["weight"] is None:
                    continue
                ts = datetime.datetime.fromisoformat(
                    entry["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
                periods.append((ts, entry["weight"]))
            periods.sort(key=lambda x: x[0])
            mentor_timelines[m["username"]] = periods
    print(f"  Mentor weight timelines: {len(mentor_timelines):,}")

    # ── Data cleaning ──
    cleaned = []
    dropped_empty_mentee = 0
    dropped_empty_question = 0
    fixed_bad_reply = 0
    fixed_empty_reply = 0

    for r in eng_first:
        if not r["mentee"]:
            dropped_empty_mentee += 1
            continue
        if not r["question_emb"]:
            dropped_empty_question += 1
            continue

        if r["has_reply"] and r.get("reply_timestamp") and r.get("timestamp"):
            try:
                q_dt = datetime.datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
                r_dt = datetime.datetime.fromisoformat(r["reply_timestamp"].replace("Z", "+00:00"))
                if (r_dt - q_dt).total_seconds() < -300:
                    r = dict(r, has_reply=False, reply_emb="", reply_clean="",
                             reply_words=0, reply_timestamp=None, reply_signer="none",
                             actual_responders=[])
                    fixed_bad_reply += 1
            except (ValueError, TypeError):
                pass

        if r["has_reply"] and not r["reply_emb"]:
            r = dict(r, has_reply=False, reply_clean="", reply_words=0,
                     reply_timestamp=None, reply_signer="none", actual_responders=[])
            fixed_empty_reply += 1

        cleaned.append(r)

    print(f"  Dropped empty mentee:        {dropped_empty_mentee:,}")
    print(f"  Dropped empty question:      {dropped_empty_question:,}")
    print(f"  Fixed bad reply (neg time):  {fixed_bad_reply:,}")
    print(f"  Fixed empty reply:           {fixed_empty_reply:,}")
    print(f"  After cleanup:               {len(cleaned):,}")

    eng_first = cleaned

    # ── Build utterances, speakers, conversations ──
    utterances = []
    speakers = {}
    conversations = {}

    for r in eng_first:
        cid = r["conversation_id"]
        q_id = f"q_{cid}"
        r_id = f"r_{cid}"
        conv_id = q_id

        mentee = r["mentee"]
        mentor = r["mentor"]

        if mentee not in speakers:
            speakers[mentee] = {"role": "mentee"}
        if mentor not in speakers:
            speakers[mentor] = {"role": "mentor"}

        utterances.append({
            "id": q_id,
            "speaker": mentee,
            "conversation_id": conv_id,
            "reply_to": None,
            "timestamp": r["timestamp"],
            "text": r["question_emb"] or "",
            "meta": {
                "clean_text": r["question_clean"],
                "word_count": r["question_words"],
                "revid": r["revid"],
                "article": r["article"],
            },
        })

        if r["has_reply"]:
            if args.assigned_only and r["reply_signer"] != "assigned_mentor":
                pass
            else:
                responders = r.get("actual_responders", [])
                mentor_lower = mentor.strip().replace("_", " ").lower()
                if r["reply_signer"] == "other_only" and responders:
                    reply_speaker = responders[0]
                    for resp in responders:
                        if resp != mentor_lower:
                            reply_speaker = resp
                            break
                else:
                    reply_speaker = mentor

                if reply_speaker not in speakers:
                    speakers[reply_speaker] = {"role": "other_responder"}

                utterances.append({
                    "id": r_id,
                    "speaker": reply_speaker,
                    "conversation_id": conv_id,
                    "reply_to": q_id,
                    "timestamp": r.get("reply_timestamp"),
                    "text": r["reply_emb"] or "",
                    "meta": {
                        "clean_text": r["reply_clean"],
                        "word_count": r["reply_words"],
                        "reply_signer": r["reply_signer"],
                    },
                })

        # Mentor assignment type at conversation time
        mentor_type = "unknown"
        periods = mentor_timelines.get(mentor, [])
        if periods and r.get("timestamp"):
            try:
                conv_dt = datetime.datetime.fromisoformat(
                    r["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
                starts = [p[0] for p in periods]
                idx = bisect_right(starts, conv_dt) - 1
                w = periods[max(0, idx)][1]
                mentor_type = "auto" if w >= 1 else "manual"
            except (ValueError, TypeError):
                pass

        conversations[conv_id] = {
            "mentor": mentor,
            "mentee": mentee,
            "mentor_type": mentor_type,
            "has_reply": r["has_reply"],
            "reply_signer": r["reply_signer"],
            "actual_responders": r.get("actual_responders", []),
            "page": r["page"],
            "article": r["article"],
        }

    # ── Write files ──

    with open(out_dir / "utterances.jsonl", "w", encoding="utf-8") as f:
        for u in utterances:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    with open(out_dir / "speakers.json", "w", encoding="utf-8") as f:
        json.dump(speakers, f, ensure_ascii=False, indent=1)

    with open(out_dir / "conversations.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=1)

    n_q = sum(1 for u in utterances if u["reply_to"] is None)
    n_r = sum(1 for u in utterances if u["reply_to"] is not None)
    corpus_meta = {
        "name": "wiki-mentor-corpus",
        "description": "Wikipedia Growth mentorship program: mentee first questions and mentor first replies (2021-2026)",
        "num_conversations": len(conversations),
        "num_utterances": len(utterances),
        "num_questions": n_q,
        "num_replies": n_r,
        "num_speakers": len(speakers),
        "num_mentors": sum(1 for s in speakers.values() if s["role"] == "mentor"),
        "num_mentees": sum(1 for s in speakers.values() if s["role"] == "mentee"),
        "time_range": "2021-05 to 2026-04",
        "language": "en",
        "source": "Wikipedia API (talk page wikitext + revision history recovery)",
        "assigned_only": args.assigned_only,
    }
    with open(out_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus_meta, f, ensure_ascii=False, indent=2)

    index = {
        "utterances-index": {
            "clean_text": "<class 'str'>",
            "word_count": "<class 'int'>",
            "revid": "<class 'int'>",
            "article": "<class 'str'>",
            "reply_signer": "<class 'str'>",
        },
        "speakers-index": {
            "role": "<class 'str'>",
        },
        "conversations-index": {
            "mentor": "<class 'str'>",
            "mentee": "<class 'str'>",
            "mentor_type": "<class 'str'>",
            "has_reply": "<class 'bool'>",
            "reply_signer": "<class 'str'>",
            "actual_responders": "<class 'list'>",
            "page": "<class 'str'>",
            "article": "<class 'str'>",
        },
        "overall-index": {
            "name": "<class 'str'>",
            "description": "<class 'str'>",
            "num_conversations": "<class 'int'>",
            "num_utterances": "<class 'int'>",
            "num_questions": "<class 'int'>",
            "num_replies": "<class 'int'>",
            "num_speakers": "<class 'int'>",
            "num_mentors": "<class 'int'>",
            "num_mentees": "<class 'int'>",
            "time_range": "<class 'str'>",
            "language": "<class 'str'>",
            "source": "<class 'str'>",
            "assigned_only": "<class 'bool'>",
        },
        "version": "1",
    }
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n  ConvoKit corpus written to {out_dir}/")
    print(f"    utterances.jsonl:   {len(utterances):>7,} utterances")
    print(f"    speakers.json:     {len(speakers):>7,} speakers")
    print(f"    conversations.json:{len(conversations):>7,} conversations")
    print(f"    corpus.json:       corpus metadata")
    print(f"    index.json:        field type index")
    print(f"\n  Questions: {n_q:,}  Replies: {n_r:,}")

    print(f"\n  To verify: python -c \"from convokit import Corpus; c = Corpus('{out_dir}'); print(c)\"")


if __name__ == "__main__":
    main()
