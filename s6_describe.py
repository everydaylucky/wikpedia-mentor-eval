#!/usr/bin/env python3
"""
s6_describe.py — Descriptive statistics for all pipeline stages.

Reads:  data/s1/s1_mentor_list.jsonl
        data/s2/s2_mentor_conversation_merged.jsonl
        data/s5/s5_all_conversations.jsonl
Output: data/s5/s5_dataset_report.txt (stdout + file)
"""
import json, re, statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
S1_DATA = BASE / "data" / "s1"
S2_DATA = BASE / "data" / "s2"
S5_DATA = BASE / "data" / "s5"

Q_RE = re.compile(
    r'^==\s*Question from \[\[User:(?P<user>[^\]|]+)',
    re.MULTILINE,
)


def describe_s1():
    path = S1_DATA / "s1_mentor_list.jsonl"
    if not path.exists():
        print("s1 data not found, skipping")
        return

    mentors = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            mentors.append(json.loads(line))

    n = len(mentors)
    n_current = sum(1 for m in mentors if m["is_current"])
    n_exited = sum(1 for m in mentors if m["exited"])

    print("=" * 60)
    print("S1: Mentor List")
    print("=" * 60)
    print(f"Total mentors:      {n}")
    print(f"Currently active:   {n_current} ({n_current/n*100:.1f}%)")
    print(f"Exited:             {n_exited} ({n_exited/n*100:.1f}%)")

    pool = Counter(m.get("current_pool_status") for m in mentors)
    print(f"\nCurrent pool status:")
    for status, count in pool.most_common():
        print(f"  {status or 'None (exited)'}: {count}")

    join_counts = Counter(m["join_count"] for m in mentors)
    print(f"\nJoin count distribution:")
    for jc in sorted(join_counts):
        print(f"  {jc} times: {join_counts[jc]} mentors")

    years = Counter()
    for m in mentors:
        fj = m.get("first_joined")
        if fj:
            years[fj[:4]] += 1
    print(f"\nFirst joined by year:")
    for y in sorted(years):
        print(f"  {y}: {years[y]}")
    print()


def describe_s2():
    path = S2_DATA / "s2_mentor_conversation_merged.jsonl"
    if not path.exists():
        print("s2 data not found, skipping")
        return

    pages = []
    total_q = 0
    mentor_q = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pages.append(r)
            qc = r.get("q_count", 0)
            total_q += qc
            mentor_q[r["mentor"]] += qc

    print("=" * 60)
    print("S2: Conversations (merged)")
    print("=" * 60)
    print(f"Page records:       {len(pages)}")
    print(f"Total questions:    {total_q}")
    print(f"Unique mentors:     {len(mentor_q)}")

    q_counts = [p.get("q_count", 0) for p in pages]
    print(f"\nQuestions per page:")
    print(f"  Mean:   {statistics.mean(q_counts):.1f}")
    print(f"  Median: {statistics.median(q_counts):.0f}")
    print(f"  Max:    {max(q_counts)}")

    print(f"\nTop 10 mentors:")
    for m, c in mentor_q.most_common(10):
        print(f"  {m}: {c}")
    print()


def describe_s5():
    path = S5_DATA / "s5_all_conversations.jsonl"
    if not path.exists():
        print("s5 data not found, skipping")
        return

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    print("=" * 60)
    print("S5: Complete Dataset")
    print("=" * 60)
    print(f"Total conversations: {total:,}")

    src = Counter(r["source"] for r in records)
    for s, c in src.most_common():
        print(f"  {s}: {c:,} ({c/total*100:.1f}%)")

    has_q = sum(1 for r in records if r.get("question_text"))
    has_r = sum(1 for r in records if r.get("mentor_reply"))
    print(f"\nHas question_text: {has_q:,} ({has_q/total*100:.1f}%)")
    print(f"Has mentor_reply:  {has_r:,} ({has_r/total*100:.1f}%)")
    print(f"Reply rate:        {has_r/total*100:.1f}%")

    mentors = set(r["mentor"] for r in records)
    mentees = set(r["mentee"] for r in records)
    print(f"\nUnique mentors: {len(mentors):,}")
    print(f"Unique mentees: {len(mentees):,}")

    qs_per_mentor = Counter(r["mentor"] for r in records)
    vals = list(qs_per_mentor.values())
    print(f"\nQuestions per mentor:")
    print(f"  Mean:   {statistics.mean(vals):.1f}")
    print(f"  Median: {statistics.median(vals):.1f}")
    print(f"  Max:    {max(vals)}")

    q_lens = [len(r["question_text"]) for r in records if r.get("question_text")]
    r_lens = [len(r["mentor_reply"]) for r in records if r.get("mentor_reply")]
    for label, lens in [("Question", q_lens), ("Reply", r_lens)]:
        print(f"\n{label} text length (chars):")
        print(f"  Mean:   {statistics.mean(lens):.0f}")
        print(f"  Median: {statistics.median(lens):.0f}")
        print(f"  Max:    {max(lens)}")

    years = Counter()
    for r in records:
        ts = r.get("timestamp")
        if ts:
            try:
                years[ts[:4]] += 1
            except Exception:
                pass
    print(f"\nBy year:")
    for y in sorted(years):
        print(f"  {y}: {years[y]:,}")

    report_path = S5_DATA / "s5_dataset_report.txt"
    lines = []
    lines.append(f"S5 Dataset Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Total: {total:,}")
    lines.append(f"Sources: {dict(src)}")
    lines.append(f"Reply rate: {has_r/total*100:.1f}%")
    lines.append(f"Mentors: {len(mentors):,}, Mentees: {len(mentees):,}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport saved: {report_path}")


def main():
    describe_s1()
    describe_s2()
    describe_s5()


if __name__ == "__main__":
    main()
