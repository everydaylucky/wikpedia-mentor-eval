#!/usr/bin/env python3
"""
s5_merge_dataset.py — Merge s2 + s4 into complete dataset.

Steps:
  1. Parse s2 wikitext into per-question records, join with tag cache for revid
  2. Load s4 recovered records
  3. Deduplicate and merge
  4. Output final dataset

Reads:  data/s2/s2_mentor_conversation_merged.jsonl
        data/s3/s3_tag_revisions_cache.jsonl
        data/s4/s4_recovered_conversations.jsonl
Output: data/s5/s5_all_conversations.jsonl
"""
import json, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data" / "s5"
S2_DATA = BASE / "data" / "s2"
S3_DATA = BASE / "data" / "s3"
S4_DATA = BASE / "data" / "s4"

S2_MERGED = S2_DATA / "s2_mentor_conversation_merged.jsonl"
TAG_CACHE = S3_DATA / "s3_tag_revisions_cache.jsonl"
S4_FILE   = S4_DATA / "s4_recovered_conversations.jsonl"
OUT_FILE  = DATA / "s5_all_conversations.jsonl"

Q_RE = re.compile(
    r"==\s*Question from \[\[User:(?P<user>[^\]|]+)[^\]]*\]\]"
    r"(?P<mid>.*?)\((?P<ts>\d{1,2}:\d{2},?\s+\d{1,2}\s+\w+\s+\d{4})\)\s*==",
    re.IGNORECASE | re.DOTALL,
)

SECTION_RE = re.compile(r"^==\s+[^=]", re.MULTILINE)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_wikitext_ts(raw):
    m = re.search(r"(\d{1,2}):(\d{2}),?\s+(\d{1,2})\s+(\w+)\s+(\d{4})", raw)
    if not m:
        return None, None
    hh, mm, dd, mon, yyyy = m.groups()
    mon_n = MONTHS.get(mon.lower())
    if not mon_n:
        return None, None
    key = f"{yyyy}{mon_n:02d}{int(dd):02d}-{hh.zfill(2)}{mm}"
    iso = f"{yyyy}-{mon_n:02d}-{int(dd):02d}T{hh.zfill(2)}:{mm}:00Z"
    return key, iso


def parse_api_ts(ts_str):
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
        return f"{dt.year}{dt.month:02d}{dt.day:02d}-{dt.hour:02d}{dt.minute:02d}"
    except Exception:
        return None


def extract_section_body(wikitext, match_end):
    rest = wikitext[match_end:]
    nxt = SECTION_RE.search(rest)
    body = rest[:nxt.start()] if nxt else rest
    return body.strip()


def extract_article(mid_text):
    m = re.search(r"on \[\[([^\]]+)\]\]", mid_text)
    return m.group(1) if m else None


def parse_s2():
    tag_index = defaultdict(dict)
    if TAG_CACHE.exists():
        with open(TAG_CACHE, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                key = (t["mentor"].lower(), t["user"].lower(), parse_api_ts(t["timestamp"]))
                tag_index[key] = t

    records = []
    with open(S2_MERGED, encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            mentor = page["mentor"]
            wt = page.get("wikitext", "")
            for m in Q_RE.finditer(wt):
                mentee = m.group("user").strip()
                article = extract_article(m.group("mid"))
                ts_key, ts_iso = parse_wikitext_ts(m.group("ts"))
                body = extract_section_body(wt, m.end())

                tag_rec = tag_index.get((mentor.lower(), mentee.lower(), ts_key))
                revid = tag_rec["revid"] if tag_rec else None

                lines = body.split("\n")
                # Skip "mentor is away" forwarding header — it starts with : but is not a reply
                import re as _re
                while lines and _re.match(
                        r"^\s*:?\s*'*\s*(Note:\s*)?\[\[User[ _]talk:",
                        lines[0]) and "is away" in lines[0]:
                    lines.pop(0)
                q_lines, r_lines = [], []
                found_reply = False
                for ln in lines:
                    stripped = ln.strip()
                    if not found_reply and (stripped.startswith(":") or stripped.startswith("*:")):
                        found_reply = True
                    if found_reply:
                        r_lines.append(ln)
                    else:
                        q_lines.append(ln)

                question_text = "\n".join(q_lines).strip()
                mentor_reply = "\n".join(r_lines).strip() if r_lines else None

                records.append({
                    "mentor": mentor, "mentee": mentee, "revid": revid,
                    "timestamp": ts_iso, "article": article,
                    "question_text": question_text,
                    "mentor_reply": mentor_reply if mentor_reply else None,
                    "source": "s2_wikitext", "page": page["page"],
                })

    print(f"[Step 1] Parsed {len(records)} questions from s2")
    return records


def load_s4():
    if not S4_FILE.exists():
        print("[Step 2] No s4 file found, skipping")
        return []
    out = []
    with open(S4_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("recovery_status") != "ok":
                continue
            out.append({
                "mentor": r["mentor"],
                "mentee": r.get("mentee_at_edit_time") or r["mentee"],
                "revid": r["revid"], "timestamp": r["timestamp"],
                "article": r.get("article"),
                "question_text": r.get("question_text"),
                "mentor_reply": r.get("mentor_reply"),
                "source": "s4_recovered", "page": None,
            })
    print(f"[Step 2] Loaded {len(out)} ok records from s4")
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("S5: Merge s2 + s4 -> Complete Dataset")
    print("=" * 60)

    s2_recs = parse_s2()
    s4_recs = load_s4()

    seen = set()
    merged = []
    for r in s2_recs:
        key = (r["mentor"].lower(), r["mentee"].lower(), r["timestamp"])
        if key not in seen:
            seen.add(key)
            merged.append(r)

    added, dup = 0, 0
    for r in s4_recs:
        key = (r["mentor"].lower(), r["mentee"].lower(), r["timestamp"])
        if key not in seen:
            seen.add(key)
            merged.append(r)
            added += 1
        else:
            dup += 1

    print(f"[Step 3] Merged: {len(s2_recs)} s2 + {added} s4 new ({dup} dup) = {len(merged)} total")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(merged):,} records -> {OUT_FILE}")


if __name__ == "__main__":
    main()
