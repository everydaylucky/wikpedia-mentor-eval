#!/usr/bin/env python3
"""
s_describe.py
Descriptive statistics for s1 (mentor list) and s2 (conversations) data.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

BASE = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────────────────────

mentors = [json.loads(l) for l in open(BASE / "s1_mentor_list.jsonl")]

matched = [json.loads(l) for l in open(BASE / "s2_mentor_conversation_matched.jsonl")]
unmatched = [json.loads(l) for l in open(BASE / "s2_mentor_conversation_unmatched.jsonl")]
all_convs = matched + unmatched

fetched_pages = set()
fp = BASE / "s2_checkpoint_fetched_pages.txt"
if fp.exists():
    fetched_pages = {l.strip() for l in open(fp) if l.strip()}

Q_RE = re.compile(
    r"^==\s*Question from \[\[User:(?P<user>[^\]|]+)(?:\|[^\]]+)?\]\]"
    r"(?:\s+on \[\[(?P<article>[^\]]+)\]\])?(?:\s*\([^)]+\))?\s*==$",
    re.MULTILINE,
)

def sep(title="", width=65):
    if title:
        print(f"\n{'─'*width}")
        print(f"  {title}")
        print(f"{'─'*width}")
    else:
        print(f"\n{'═'*width}")


def pct(n, total):
    return f"{100*n/total:.1f}%" if total else "N/A"


# ══════════════════════════════════════════════════════════════════════════════
sep()
print("  DATA DESCRIPTION  —  Wikipedia Mentor Research")
sep()

# ── S1: Mentor list ───────────────────────────────────────────────────────────
sep("S1 · Mentor list  (s1_mentor_list.jsonl)")

n = len(mentors)
current    = sum(1 for m in mentors if m["is_current"])
exited     = sum(1 for m in mentors if m["exited"])
ever_auto  = sum(1 for m in mentors if m["ever_auto"])
ever_manual= sum(1 for m in mentors if m["ever_manual"])
both       = sum(1 for m in mentors if m["ever_auto"] and m["ever_manual"])

print(f"  Total mentors:           {n:,}")
print(f"  Currently active:        {current:,}  ({pct(current, n)})")
print(f"  Exited:                  {exited:,}  ({pct(exited, n)})")
print()
print(f"  Ever in auto pool:       {ever_auto:,}  ({pct(ever_auto, n)})")
print(f"  Ever in manual pool:     {ever_manual:,}  ({pct(ever_manual, n)})")
print(f"  Ever in both pools:      {both:,}  ({pct(both, n)})")

pool_counts = Counter(m["current_pool_status"] for m in mentors)
print(f"\n  Current pool status breakdown:")
for status, cnt in sorted(pool_counts.items(), key=lambda x: -x[1]):
    print(f"    {str(status):<10} {cnt:>5}  ({pct(cnt, n)})")

join_counts = Counter(m["join_count"] for m in mentors)
print(f"\n  Join-count distribution (times joined the program):")
for k in sorted(join_counts):
    print(f"    {k} time(s):  {join_counts[k]:>4}")

dates = sorted(
    datetime.fromisoformat(m["first_joined"].replace("Z", "+00:00"))
    for m in mentors if m["first_joined"]
)
print(f"\n  Program enrollment span:")
print(f"    Earliest first_joined: {dates[0].date()}")
print(f"    Latest   first_joined: {dates[-1].date()}")

# Year distribution
year_counts = Counter(d.year for d in dates)
print(f"\n  Mentors first joined by year:")
for yr in sorted(year_counts):
    print(f"    {yr}:  {year_counts[yr]:>4}")

# ── S2: Conversations ─────────────────────────────────────────────────────────
sep("S2 · Talk pages fetched  (checkpoint)")

print(f"  Pages in checkpoint:     {len(fetched_pages):,}")
print(f"  Matched page records:    {len(matched):,}")
print(f"  Unmatched page records:  {len(unmatched):,}")
print(f"  Total page records:      {len(all_convs):,}")

sep("S2 · Conversation questions")

total_q = sum(r["q_count"] for r in all_convs)
total_q_matched   = sum(r["q_count"] for r in matched)
total_q_unmatched = sum(r["q_count"] for r in unmatched)

print(f"  Total Q headers found:   {total_q:,}")
print(f"    from matched pages:    {total_q_matched:,}  ({pct(total_q_matched, total_q)})")
print(f"    from unmatched pages:  {total_q_unmatched:,}  ({pct(total_q_unmatched, total_q)})")

# q_count per page distribution
qc = [r["q_count"] for r in all_convs]
qc_sorted = sorted(qc)
n_pages = len(qc)
print(f"\n  Questions per talk page:")
print(f"    Min:    {min(qc)}")
print(f"    Max:    {max(qc)}")
print(f"    Mean:   {sum(qc)/n_pages:.1f}")
median = qc_sorted[n_pages // 2]
print(f"    Median: {median}")
p75 = qc_sorted[int(n_pages * 0.75)]
p90 = qc_sorted[int(n_pages * 0.90)]
print(f"    P75:    {p75}")
print(f"    P90:    {p90}")

brackets = [(1,1),(2,5),(6,20),(21,50),(51,9999)]
labels   = ["1","2–5","6–20","21–50","51+"]
print(f"\n  Pages by question count range:")
for (lo, hi), lbl in zip(brackets, labels):
    cnt = sum(1 for q in qc if lo <= q <= hi)
    print(f"    {lbl:>6} questions:  {cnt:>5} pages  ({pct(cnt, n_pages)})")

sep("S2 · Page text length (wikitext bytes)")

lens = [r["len"] for r in all_convs]
lens_sorted = sorted(lens)
print(f"  Min:    {min(lens):,}")
print(f"  Max:    {max(lens):,}")
print(f"  Mean:   {int(sum(lens)/len(lens)):,}")
print(f"  Median: {lens_sorted[len(lens)//2]:,}")

sep("S2 · Per-mentor coverage (matched pages only)")

mentor_pages   = Counter(r["mentor"] for r in matched)
mentor_qcounts = defaultdict(int)
for r in matched:
    mentor_qcounts[r["mentor"]] += r["q_count"]

n_mentors_with_data = len(mentor_pages)
n_mentors_total = len(mentors)
n_mentors_no_data = n_mentors_total - n_mentors_with_data

print(f"  Mentors with ≥1 page record:  {n_mentors_with_data:,} / {n_mentors_total:,}")
print(f"  Mentors with no page record:  {n_mentors_no_data:,}  ({pct(n_mentors_no_data, n_mentors_total)})")

pages_per_mentor = sorted(mentor_pages.values())
n_m = len(pages_per_mentor)
print(f"\n  Talk pages per mentor (among those with data):")
print(f"    Min:    {min(pages_per_mentor)}")
print(f"    Max:    {max(pages_per_mentor)}")
print(f"    Mean:   {sum(pages_per_mentor)/n_m:.1f}")
print(f"    Median: {pages_per_mentor[n_m//2]}")

print(f"\n  Top 10 mentors by total questions:")
top10 = sorted(mentor_qcounts.items(), key=lambda x: -x[1])[:10]
for rank, (name, total) in enumerate(top10, 1):
    pages = mentor_pages[name]
    print(f"    {rank:>2}. {name:<30} {total:>5} questions  {pages:>3} pages")

sep("S2 · Unmatched mentor pages")

unmatched_mentors = Counter(r["mentor"] for r in unmatched)
print(f"  Unique 'mentors' in unmatched: {len(unmatched_mentors):,}")
print(f"  Top 10 unmatched by question count:")
unmatched_qc = defaultdict(int)
for r in unmatched:
    unmatched_qc[r["mentor"]] += r["q_count"]
for rank, (name, total) in enumerate(sorted(unmatched_qc.items(), key=lambda x: -x[1])[:10], 1):
    print(f"    {rank:>2}. {name:<30} {total:>4} questions")

sep("S2 · Question timestamp coverage (from wikitext parsing)")

# Parse timestamps from a sample of wikitext to show year distribution
TS_RE = re.compile(r'\d{1,2}:\d{2},\s+\d+\s+\w+\s+(\d{4})\s+\(UTC\)')
year_q: Counter = Counter()
sample_limit = 300  # avoid being slow; parse first N records
for r in all_convs[:sample_limit]:
    for m in TS_RE.finditer(r.get("wikitext", "")):
        try:
            year_q[int(m.group(1))] += 1
        except Exception:
            pass

if year_q:
    print(f"  (Sample: first {sample_limit} page records)")
    print(f"  Questions by year (from timestamp strings in wikitext):")
    for yr in sorted(year_q):
        filled = int(40 * year_q[yr] / max(year_q.values()))
        print(f"    {yr}: {year_q[yr]:>6}  {'█' * filled}")
else:
    print("  (Could not parse timestamps from wikitext sample)")

sep()
print("  Done.")
sep()
