#!/usr/bin/env python3
"""
s4_recover_missing.py — Recover missing conversations + fix mentor reply field.

Uses tag match results to identify missing conversations, then recovers them
via revision history. Fixes mentor_reply field (strips duplicated question text).

Reads:  data/s3/s3_tag_match_results.jsonl
Output: data/s4/s4_recovered_conversations.jsonl  (final, fixed)
        data/s4/s4_cache_history.jsonl
        data/s4/s4_cache_wikitext_q.jsonl
        data/s4/s4_cache_wikitext_reply.jsonl
        data/s4/s4_recovery_report.txt
"""
import json, re, ssl, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = Path(__file__).parent
DATA = BASE / "data" / "s4"
S3_DATA = BASE / "data" / "s3"

API = "https://en.wikipedia.org/w/api.php"
UA = "MentorResearch/1.0 (academic research)"
DELAY = 1.0

V3_RESULTS = S3_DATA / "s3_tag_match_results.jsonl"
OUT_RAW    = DATA / "s4_recovered_raw.jsonl"
OUT_FILE   = DATA / "s4_recovered_conversations.jsonl"
CKPT_FILE  = DATA / "s4_mentor_checkpoint.txt"

CACHE_HISTORY = DATA / "s4_cache_history.jsonl"
CACHE_Q_WT    = DATA / "s4_cache_wikitext_q.jsonl"
CACHE_R_WT    = DATA / "s4_cache_wikitext_reply.jsonl"
REPORT_FILE   = DATA / "s4_recovery_report.txt"

CTX = ssl.create_default_context()

Q_RE = re.compile(
    r'^==\s*Question from \[\[User:(?P<user>[^\]|]+)(?:\|[^\]]+)?\]\]'
    r'(?:\s+on \[\[(?P<article>[^\]]+)\]\])?\s*'
    r'\((?P<ts>\d{1,2}:\d{2},?\s+\d{1,2}\s+\w+\s+\d{4})\)\s*==$',
    re.MULTILINE,
)
H2_RE = re.compile(r'^==\s[^=]', re.MULTILINE)


def api_get(params):
    params["format"] = "json"
    url = f"{API}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": UA})
    last_exc = None
    for attempt in range(8):
        try:
            with urlopen(req, context=CTX, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            last_exc = e
            if "429" in str(e):
                wait = 60 * (attempt + 1)
                print(f"\n  429 (attempt {attempt+1}), cooling {wait}s...", flush=True)
                time.sleep(wait)
            elif attempt < 7:
                time.sleep(2 ** attempt)
            else:
                raise
    raise last_exc


def fetch_all_revisions(page_title):
    revisions = []
    params = {
        "action": "query", "titles": page_title,
        "prop": "revisions",
        "rvprop": "ids|timestamp|user|comment|tags|size",
        "rvlimit": "500", "rvdir": "newer",
    }
    while True:
        d = api_get(params)
        pages = d.get("query", {}).get("pages", {})
        for pid, info in pages.items():
            if pid == "-1":
                return []
            revisions.extend(info.get("revisions", []))
        if "continue" in d:
            params.update(d["continue"])
            time.sleep(DELAY)
        else:
            break
    return revisions


def batch_fetch_wikitext(revids):
    params = {
        "action": "query",
        "revids": "|".join(str(r) for r in revids),
        "prop": "revisions",
        "rvprop": "content|ids|timestamp|user",
        "rvslots": "main",
    }
    d = api_get(params)
    result = {}
    for pid, info in d.get("query", {}).get("pages", {}).items():
        for rev in info.get("revisions", []):
            slot = rev.get("slots", {}).get("main", {})
            text = slot.get("*", "") or rev.get("*", "")
            result[rev["revid"]] = {
                "wikitext": text,
                "timestamp": rev.get("timestamp", ""),
                "user": rev.get("user", ""),
            }
    return result


def extract_section(wikitext, start_pos):
    eol = wikitext.find("\n", start_pos)
    if eol == -1:
        return ""
    body_start = eol + 1
    next_h2 = H2_RE.search(wikitext, body_start)
    body_end = next_h2.start() if next_h2 else len(wikitext)
    return wikitext[body_start:body_end].strip()


def find_question_section(wikitext, mentee_user):
    mentee_lower = mentee_user.lower()
    for m in Q_RE.finditer(wikitext):
        if m.group("user").strip().lower() == mentee_lower:
            return m, extract_section(wikitext, m.start())
    all_matches = list(Q_RE.finditer(wikitext))
    if all_matches:
        m = all_matches[-1]
        return m, extract_section(wikitext, m.start())
    return None, None


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_reply_only(question_text, mentor_reply):
    q_lines = question_text.split("\n")
    r_lines = mentor_reply.split("\n")
    i = 0
    while i < len(q_lines) and i < len(r_lines):
        if q_lines[i].strip() == r_lines[i].strip():
            i += 1
        else:
            break
    remainder = "\n".join(r_lines[i:]).strip()
    return remainder if remainder else None


def process_mentor(mentor, missing_revs):
    page_title = f"User talk:{mentor}"
    mentor_lower = mentor.lower()

    print(f"    [1/4] fetching history...", end=" ", flush=True)
    time.sleep(DELAY)
    history = fetch_all_revisions(page_title)
    print(f"{len(history)} revisions", flush=True)

    append_jsonl(CACHE_HISTORY, {
        "mentor": mentor, "page": page_title,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "revision_count": len(history), "revisions": history,
    })

    history_sorted = sorted(history, key=lambda r: r["revid"])

    question_revids = [r["revid"] for r in missing_revs]
    print(f"    [2/4] fetching {len(question_revids)} question wikitexts...", end=" ", flush=True)
    q_wikitext = {}
    for i in range(0, len(question_revids), 50):
        batch = question_revids[i:i+50]
        time.sleep(DELAY)
        data = batch_fetch_wikitext(batch)
        q_wikitext.update(data)
        for revid, d in data.items():
            append_jsonl(CACHE_Q_WT, {
                "mentor": mentor, "revid": revid,
                "timestamp": d["timestamp"], "user": d["user"],
                "wikitext": d["wikitext"],
            })
    print(f"got {len(q_wikitext)}", flush=True)

    print(f"    [3/4] identifying reply revids...", end=" ", flush=True)
    question_to_reply_revid = {}
    reply_revids_needed = set()
    for rev in missing_revs:
        qrevid = rev["revid"]
        idx = next((i for i, r in enumerate(history_sorted) if r["revid"] == qrevid), None)
        if idx is None:
            continue
        for future_rev in history_sorted[idx+1 : idx+21]:
            if future_rev.get("user", "").lower() == mentor_lower:
                rrevid = future_rev["revid"]
                reply_revids_needed.add(rrevid)
                question_to_reply_revid[qrevid] = rrevid
                break
    print(f"{len(reply_revids_needed)} reply revids", flush=True)

    print(f"    [4/4] fetching reply wikitexts...", end=" ", flush=True)
    reply_wikitext = {}
    reply_list = sorted(reply_revids_needed)
    for i in range(0, len(reply_list), 50):
        batch = reply_list[i:i+50]
        time.sleep(DELAY)
        data = batch_fetch_wikitext(batch)
        reply_wikitext.update(data)
        for revid, d in data.items():
            append_jsonl(CACHE_R_WT, {
                "mentor": mentor, "revid": revid,
                "timestamp": d["timestamp"], "user": d["user"],
                "wikitext": d["wikitext"],
            })
    print(f"got {len(reply_wikitext)}", flush=True)

    results = []
    for rev in missing_revs:
        qrevid = rev["revid"]
        mentee = rev["user"]

        q_data = q_wikitext.get(qrevid)
        if not q_data:
            results.append({
                "mentor": mentor, "mentee": mentee, "revid": qrevid,
                "timestamp": rev["timestamp"],
                "recovery_status": "fetch_failed",
                "question_text": None, "mentor_reply": None,
                "tags": rev.get("tags", []), "reverted": rev.get("reverted", False),
            })
            continue

        wt = q_data["wikitext"]
        mentee_at_edit = q_data.get("user", mentee)

        hdr, question_text = find_question_section(wt, mentee)
        if hdr is None and mentee_at_edit.lower() != mentee.lower():
            hdr, question_text = find_question_section(wt, mentee_at_edit)

        reply_revid = question_to_reply_revid.get(qrevid)
        mentor_reply = None

        if reply_revid and hdr is not None:
            r_data = reply_wikitext.get(reply_revid)
            if r_data:
                _, reply_body = find_question_section(r_data["wikitext"], mentee)
                if reply_body is None and mentee_at_edit.lower() != mentee.lower():
                    _, reply_body = find_question_section(r_data["wikitext"], mentee_at_edit)
                if reply_body and reply_body != question_text:
                    mentor_reply = reply_body

        results.append({
            "mentor": mentor, "mentee": mentee,
            "mentee_at_edit_time": mentee_at_edit,
            "revid": qrevid, "reply_revid": reply_revid,
            "timestamp": rev["timestamp"],
            "article": hdr.group("article") if hdr and hdr.group("article") else None,
            "question_text": question_text,
            "mentor_reply": mentor_reply,
            "recovery_status": "ok" if hdr else "section_not_found",
            "tags": rev.get("tags", []), "reverted": rev.get("reverted", False),
        })

    return results


def fix_mentor_replies(records):
    n_fixed = 0
    for rec in records:
        q = rec.get("question_text")
        mr = rec.get("mentor_reply")
        if not mr or not q:
            continue
        fixed = extract_reply_only(q, mr)
        if fixed != mr.strip():
            rec["mentor_reply"] = fixed
            n_fixed += 1
        q_s = q.strip()
        mr_s = (rec.get("mentor_reply") or "").strip()
        if q_s and mr_s and len(q_s) >= 50 and mr_s.startswith(q_s[:50]):
            rec["mentor_reply"] = None
            rec["reply_revid"] = None
            n_fixed += 1
    return n_fixed


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    by_mentor = defaultdict(list)
    n_skipped = 0
    with open(V3_RESULTS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            mentor = rec["mentor"]
            for rev in rec.get("missing_details", []):
                if rev.get("reverted", False):
                    n_skipped += 1
                else:
                    by_mentor[mentor].append(rev)

    total_records = sum(len(v) for v in by_mentor.values())
    print(f"Mentors with missing records: {len(by_mentor)}")
    print(f"Total missing to recover:     {total_records}")
    print(f"Skipped (reverted):           {n_skipped}")

    done_mentors = set()
    if CKPT_FILE.exists():
        with open(CKPT_FILE, encoding="utf-8") as f:
            done_mentors = set(l.strip() for l in f if l.strip())

    remaining = [m for m in sorted(by_mentor) if m not in done_mentors]
    print(f"Already done: {len(done_mentors)}, remaining: {len(remaining)}")
    print()

    stats = {"ok": 0, "section_not_found": 0, "fetch_failed": 0, "with_reply": 0}
    all_results = []

    if OUT_RAW.exists():
        with open(OUT_RAW, encoding="utf-8") as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except Exception:
                    pass

    for i, mentor in enumerate(remaining, 1):
        missing_revs = by_mentor[mentor]
        print(f"[{i}/{len(remaining)}] {mentor} ({len(missing_revs)} records)")
        try:
            results = process_mentor(mentor, missing_revs)
        except Exception as e:
            print(f"    ERROR: {e}")
            if "429" in str(e):
                time.sleep(300)
            continue

        for rec in results:
            append_jsonl(OUT_RAW, rec)
            all_results.append(rec)
            s = rec["recovery_status"]
            stats[s] = stats.get(s, 0) + 1
            if rec.get("mentor_reply"):
                stats["with_reply"] += 1

        with open(CKPT_FILE, "a", encoding="utf-8") as f:
            f.write(mentor + "\n")

        n_ok = sum(1 for r in results if r["recovery_status"] == "ok")
        n_replied = sum(1 for r in results if r.get("mentor_reply"))
        print(f"    -> ok={n_ok}  with_reply={n_replied}  not_found={len(results)-n_ok}")

    print(f"\n{'='*70}")
    print("Fixing mentor_reply field...")
    n_fixed = fix_mentor_replies(all_results)
    print(f"Fixed {n_fixed} records")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    total_ok = sum(1 for r in all_results if r.get("recovery_status") == "ok")
    total_reply = sum(1 for r in all_results if r.get("mentor_reply"))
    print(f"Total recovered (ok):    {total_ok}")
    print(f"With mentor reply:       {total_reply}")
    print(f"Output: {OUT_FILE}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"s4 recovery report — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total missing: {total_records}\n")
        f.write(f"Recovered ok: {total_ok}\n")
        f.write(f"With reply: {total_reply}\n")
        f.write(f"Reply field fixes: {n_fixed}\n")
        f.write(f"Skipped (reverted): {n_skipped}\n")


if __name__ == "__main__":
    main()
