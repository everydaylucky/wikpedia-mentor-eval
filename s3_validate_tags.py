#!/usr/bin/env python3
"""
s3_validate_tags.py — Download edit tags + validate s2 coverage.

Phase 1: Download tag-marked revisions for all mentors (pure fetch, resume-safe).
Phase 2: Local coverage analysis comparing tags vs s2 regex-extracted questions.

Reads:  data/s1/s1_mentor_list.jsonl
        data/s2/s2_mentor_conversation_merged.jsonl
Output: data/s3/s3_tag_revisions_cache.jsonl
        data/s3/s3_tag_match_results.jsonl
        data/s3/s3_tag_match_report.txt
"""
import json, re, ssl, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

BASE = Path(__file__).parent
DATA = BASE / "data" / "s3"
S1_DATA = BASE / "data" / "s1"
S2_DATA = BASE / "data" / "s2"

API = "https://en.wikipedia.org/w/api.php"
UA = "MentorResearch/1.0 (academic research)"
DELAY = 0.5

MENTOR_LIST = S1_DATA / "s1_mentor_list.jsonl"
S2_MERGED   = S2_DATA / "s2_mentor_conversation_merged.jsonl"

OUT_CACHE   = DATA / "s3_tag_revisions_cache.jsonl"
CKPT_DL     = DATA / "s3_tag_download_checkpoint.txt"
DL_REPORT   = DATA / "s3_tag_download_report.txt"

OUT_RESULTS = DATA / "s3_tag_match_results.jsonl"
OUT_REPORT  = DATA / "s3_tag_match_report.txt"

TAGS = ["mentorship panel question", "mentorship module question"]

CTX = ssl.create_default_context()

Q_RE = re.compile(
    r"==\s*Question from \[\[User:(?P<user>[^\]|]+)[^\]]*\]\]"
    r".*?\((?P<ts>\d{1,2}:\d{2},?\s+\d{1,2}\s+\w+\s+\d{4})\)\s*==",
    re.IGNORECASE,
)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def api_get(params):
    params["format"] = "json"
    url = f"{API}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": UA})
    for attempt in range(8):
        try:
            with urlopen(req, context=CTX, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"\n  429 (attempt {attempt+1}), cooling {wait}s...", flush=True)
                time.sleep(wait)
            elif attempt < 7:
                time.sleep(2 ** attempt)
            else:
                raise


def parse_s2_timestamp(raw):
    raw = raw.strip()
    m = re.search(r"(\d{1,2}):(\d{2}),?\s+(\d{1,2})\s+(\w+)\s+(\d{4})", raw)
    if not m:
        return None
    hh, mm, dd, mon, yyyy = m.groups()
    mon_n = MONTHS.get(mon.lower())
    if not mon_n:
        return None
    return f"{yyyy}{mon_n:02d}{int(dd):02d}-{hh.zfill(2)}{mm}"


def parse_api_timestamp(ts_str):
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
        return f"{dt.year}{dt.month:02d}{dt.day:02d}-{dt.hour:02d}{dt.minute:02d}"
    except Exception:
        return None


# === Phase 1: Download tag revisions ===

def fetch_tag_revisions(username):
    page = f"User talk:{username}"
    results = []
    seen_revids = set()

    for tag in TAGS:
        params = {
            "action": "query", "titles": page,
            "prop": "revisions",
            "rvprop": "ids|timestamp|user|tags",
            "rvlimit": "500", "rvtag": tag, "rvdir": "newer",
        }
        while True:
            d = api_get(params)
            pages = d.get("query", {}).get("pages", {})
            for pid, info in pages.items():
                if pid == "-1":
                    break
                for rev in info.get("revisions", []):
                    rid = rev["revid"]
                    if rid not in seen_revids:
                        seen_revids.add(rid)
                        results.append({
                            "revid": rid,
                            "timestamp": rev.get("timestamp", ""),
                            "user": rev.get("user", ""),
                            "tags": rev.get("tags", []),
                        })
            if "continue" in d:
                params.update(d["continue"])
                time.sleep(DELAY)
            else:
                break
        time.sleep(DELAY)

    return results


def phase1_download():
    print(f"{'='*70}")
    print("PHASE 1: Download tag-marked revisions")
    print(f"{'='*70}")

    mentors = []
    with open(MENTOR_LIST, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("username"):
                mentors.append(rec["username"])

    done = set()
    if CKPT_DL.exists():
        with open(CKPT_DL, encoding="utf-8") as f:
            done = set(l.strip() for l in f if l.strip())

    remaining = [m for m in mentors if m not in done]
    print(f"Total mentors: {len(mentors)}, done: {len(done)}, remaining: {len(remaining)}")

    t_start = time.time()
    per_mentor_counts = {}

    for i, mentor in enumerate(remaining, 1):
        try:
            revs = fetch_tag_revisions(mentor)
        except Exception as e:
            print(f"  [{i}/{len(remaining)}] {mentor}  ERROR: {e}")
            if "429" in str(e):
                time.sleep(120)
            continue

        with open(OUT_CACHE, "a", encoding="utf-8") as f:
            for rev in revs:
                rec = {"mentor": mentor, **rev}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with open(CKPT_DL, "a", encoding="utf-8") as f:
            f.write(mentor + "\n")

        per_mentor_counts[mentor] = len(revs)
        print(f"  [{i}/{len(remaining)}] {mentor}  tag_revs={len(revs)}")

    elapsed = time.time() - t_start
    total_revs = 0
    if OUT_CACHE.exists():
        with open(OUT_CACHE, encoding="utf-8") as f:
            total_revs = sum(1 for _ in f)

    with open(DL_REPORT, "w", encoding="utf-8") as f:
        f.write(f"s3 tag download report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total mentors: {len(mentors)}\n")
        f.write(f"Completed: {len(done) + len(per_mentor_counts)}\n")
        f.write(f"This run: {len(per_mentor_counts)}\n")
        f.write(f"Total revisions in cache: {total_revs}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")

    print(f"\nCache: {OUT_CACHE} ({total_revs} revisions)")


# === Phase 2: Local coverage analysis ===

def build_s2_index():
    by_ts = defaultdict(set)
    by_name = defaultdict(set)
    n = 0
    with open(S2_MERGED, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mentor = rec.get("mentor", "").lower()
            wt = rec.get("wikitext", "")
            for m in Q_RE.finditer(wt):
                mentee = m.group("user").strip().lower()
                ts_key = parse_s2_timestamp(m.group("ts").strip())
                by_name[mentor].add(mentee)
                n += 1
                if ts_key:
                    by_ts[mentor].add((mentee, ts_key))
    print(f"  s2 index: {n} questions across {len(by_ts)} mentors")
    return by_ts, by_name


def load_cache():
    by_mentor = defaultdict(list)
    with open(OUT_CACHE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mentor = rec.pop("mentor")
            by_mentor[mentor].append(rec)
    return by_mentor


def phase2_match():
    print(f"\n{'='*70}")
    print("PHASE 2: Local coverage analysis")
    print(f"{'='*70}")

    print("Building s2 index...")
    s2_by_ts, s2_by_name = build_s2_index()

    print("Loading tag cache...")
    cache = load_cache()
    print(f"  cache: {sum(len(v) for v in cache.values())} revisions, {len(cache)} mentors")

    all_results = []
    for mentor in sorted(cache.keys()):
        tag_revs = cache[mentor]
        mentor_lower = mentor.lower()
        ts_set = s2_by_ts.get(mentor_lower, set())
        name_set = s2_by_name.get(mentor_lower, set())

        matched, missing, name_only = [], [], []
        for rev in tag_revs:
            mentee_lower = rev["user"].lower()
            ts_key = parse_api_timestamp(rev["timestamp"])
            reverted = "mw-reverted" in rev.get("tags", [])
            rev_out = {
                "revid": rev["revid"], "user": rev["user"],
                "timestamp": rev["timestamp"], "reverted": reverted,
                "tags": rev.get("tags", []),
            }
            if ts_key and (mentee_lower, ts_key) in ts_set:
                matched.append(rev_out)
            elif mentee_lower in name_set:
                name_only.append(rev_out)
            else:
                missing.append(rev_out)

        n_reverted = sum(1 for r in missing if r["reverted"])
        n_not_reverted = sum(1 for r in missing if not r["reverted"])
        total = len(matched) + len(missing) + len(name_only)

        all_results.append({
            "mentor": mentor, "total_tags": total,
            "matched": len(matched), "name_only": len(name_only),
            "still_missing": len(missing),
            "missing_reverted": n_reverted,
            "missing_not_reverted": n_not_reverted,
            "missing_details": missing,
        })

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_tags = sum(r["total_tags"] for r in all_results)
    total_matched = sum(r["matched"] for r in all_results)
    total_name = sum(r["name_only"] for r in all_results)
    total_missing = sum(r["still_missing"] for r in all_results)
    total_reverted = sum(r["missing_reverted"] for r in all_results)
    total_not_reverted = sum(r["missing_not_reverted"] for r in all_results)

    def pct(n, d):
        return 100 * n / d if d else 0.0

    lines = [
        "=" * 70,
        f"  s3_validate_tags — coverage analysis",
        f"  Mentors: {len(all_results)}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70, "",
        f"  Total tag revisions:        {total_tags}",
        f"  Strict match (user+ts):     {total_matched}  ({pct(total_matched,total_tags):.1f}%)",
        f"  Name-only (ts mismatch):    {total_name}  ({pct(total_name,total_tags):.1f}%)",
        f"  Still missing:              {total_missing}  ({pct(total_missing,total_tags):.1f}%)",
        f"  Loose coverage:             {pct(total_matched+total_name,total_tags):.1f}%", "",
        f"  Missing breakdown:",
        f"    Reverted:     {total_reverted}  ({pct(total_reverted,max(total_missing,1)):.1f}% of missing)",
        f"    NOT reverted: {total_not_reverted}  ({pct(total_not_reverted,max(total_missing,1)):.1f}% of missing)", "",
    ]

    lines.append(f"  {'Mentor':<32s} {'tags':>5s} {'match':>5s} {'name':>5s} "
                 f"{'miss':>5s} {'revert':>6s} {'!rev':>5s} {'match%':>7s}")
    lines.append("  " + "-" * 78)
    for r in sorted(all_results, key=lambda r: r["matched"] / max(r["total_tags"], 1)):
        if r["total_tags"] == 0:
            continue
        p = pct(r["matched"], r["total_tags"])
        lines.append(
            f"  {r['mentor']:<32s} {r['total_tags']:>5d} {r['matched']:>5d} "
            f"{r['name_only']:>5d} {r['still_missing']:>5d} "
            f"{r['missing_reverted']:>6d} {r['missing_not_reverted']:>5d} "
            f"{p:>6.1f}%"
        )

    report = "\n".join(lines) + "\n"
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    phase1_download()
    phase2_match()
    print(f"\nResults: {OUT_RESULTS}")
    print(f"Report:  {OUT_REPORT}")


if __name__ == "__main__":
    main()
