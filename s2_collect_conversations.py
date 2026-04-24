#!/usr/bin/env python3
"""
s2_collect_conversations.py — Fetch mentor-mentee conversations + fix & merge.

Phase 1: Fetch talk pages for known mentors + search API for unmatched pages.
Phase 2: Fix issues (missing subpages, misclassified unmatched) and merge into
         a single clean output file.

Reads:  data/s1/s1_mentor_list.jsonl
Output: data/s2/s2_mentor_conversation_merged.jsonl
        data/s2/s2_mentor_conversation_unmatched_clean.jsonl

All intermediate and checkpoint files are stored in data/s2/.
Resume-safe: restart at any point and completed work is skipped.
"""
import json, re, time, ssl, sys
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
DATA = BASE / "data" / "s2"
S1_DATA = BASE / "data" / "s1"
API = "https://en.wikipedia.org/w/api.php"
UA = "MentorResearch/1.0 (academic research)"
BATCH = 10

MENTOR_LIST = S1_DATA / "s1_mentor_list.jsonl"

CKPT_FETCHED  = DATA / "s2_checkpoint_fetched_pages.txt"
CKPT_PAGELIST = DATA / "s2_checkpoint_page_list.json"
CKPT_SEARCH   = DATA / "s2_checkpoint_search_results.jsonl"

OUT_MATCHED   = DATA / "s2_mentor_conversation_matched.jsonl"
OUT_UNMATCHED = DATA / "s2_mentor_conversation_unmatched.jsonl"

OUT_MERGED          = DATA / "s2_mentor_conversation_merged.jsonl"
OUT_UNMATCHED_CLEAN = DATA / "s2_mentor_conversation_unmatched_clean.jsonl"
OUT_REPORT          = DATA / "s2_fix_merge_report.txt"

CKPT_ALLPAGES  = DATA / "s2_fix_checkpoint_allpages.json"
CKPT_NEWDATA   = DATA / "s2_fix_checkpoint_newdata.jsonl"
CKPT_BFETCHED  = DATA / "s2_fix_checkpoint_bfetched.txt"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


class AdaptiveThrottle:
    def __init__(self, init=1.5, lo=0.8, hi=30.0):
        self.delay = init
        self.lo = lo
        self.hi = hi
        self.floor = lo

    def success(self):
        self.delay = max(self.floor, self.delay * 0.97)

    def rate_limited(self):
        self.floor = min(self.hi, max(self.floor, self.delay) * 1.5)
        self.delay = min(self.hi, self.delay * 2.5)

    def wait(self):
        time.sleep(self.delay)

    def __repr__(self):
        return f"{self.delay:.2f}s(floor={self.floor:.2f})"


throttle = AdaptiveThrottle()


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def progress_bar(current, total, start_time, width=30, extra=""):
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - start_time
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = fmt_time(eta)
    else:
        eta_str = "?"
    line = f"\r  [{bar}] {current}/{total} ({pct*100:.1f}%) elapsed={fmt_time(elapsed)} ETA={eta_str}"
    if extra:
        line += f"  {extra}"
    print(line, end="", flush=True)


def api_get(params, retries=6):
    params["format"] = "json"
    url = f"{API}?{urlencode(params)}"
    for i in range(retries):
        try:
            r = urlopen(Request(url, headers={"User-Agent": UA}), context=ctx, timeout=60)
            throttle.success()
            return json.loads(r.read())
        except Exception as e:
            if "429" in str(e):
                throttle.rate_limited()
                if i >= 3:
                    w = 60 * (i - 2)
                    print(f"\n  429 (attempt {i+1}), cooldown {w}s...", flush=True)
                else:
                    w = throttle.delay + i * 3
                    print(f"\n  429, delay→{throttle}, wait {w:.1f}s", flush=True)
                time.sleep(w)
            elif i < retries - 1:
                w = 5 * (i + 1)
                print(f"\n  err: {e}, retry in {w}s", flush=True)
                time.sleep(w)
            else:
                raise
    return {}


def fetch_batch_wikitext(titles):
    result = {}
    p = {
        "action": "query", "titles": "|".join(titles),
        "prop": "revisions", "rvprop": "content", "rvslots": "main", "rvlimit": "1",
    }
    while True:
        d = api_get(p)
        for pid, info in d.get("query", {}).get("pages", {}).items():
            t = info.get("title", "")
            if "revisions" in info:
                slot = info["revisions"][0].get("slots", {}).get("main", {})
                wt = slot.get("*", "") or info["revisions"][0].get("*", "")
                result[t] = wt
        if "continue" in d:
            p.update(d["continue"])
            throttle.wait()
        else:
            break
    return result


# ─── Question parsing ───

Q_RE = re.compile(
    r'^==\s*Question from \[\[User:(?P<user>[^\]|]+)(?:\|[^\]]+)?\]\]'
    r'(?:\s+on \[\[(?P<article>[^\]]+)\]\])?\s*\((?P<ts>[^)]+)\)\s*==$',
    re.MULTILINE,
)
H2_RE = re.compile(r'^==\s[^=].*[^=]\s==$', re.MULTILINE)


def parse_questions(wt, page, mentor):
    out = []
    ms = list(Q_RE.finditer(wt))
    if not ms:
        return out
    hs = list(H2_RE.finditer(wt))
    for m in ms:
        end = len(wt)
        for h in hs:
            if h.start() > m.end():
                end = h.start()
                break
        out.append({
            "mentor": mentor,
            "mentee": m.group("user").strip(),
            "timestamp": m.group("ts").strip(),
            "article_context": m.group("article"),
            "page_title": page,
            "header": m.group(0),
            "content": wt[m.end():end].strip(),
        })
    return out


def extract_mentor_from_title(title):
    if title.startswith("User talk:"):
        return title[len("User talk:"):].split("/")[0]
    return None


# ─── Checkpoint helpers ───

def load_fetched():
    if CKPT_FETCHED.exists():
        with open(CKPT_FETCHED) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def mark_fetched(titles):
    with open(CKPT_FETCHED, "a") as f:
        for t in titles:
            f.write(t + "\n")


# ─── Phase 1A: Collect page titles (archive subpages only) ───

def get_archive_talk_pages(mentors):
    all_pages = {}
    done_usernames = set()
    if CKPT_PAGELIST.exists():
        with open(CKPT_PAGELIST) as f:
            saved = json.load(f)
        all_pages.update(saved)
        done_usernames = set(saved.values())

    remaining = [m for m in mentors if m["username"] not in done_usernames]
    if not remaining:
        print(f"  Loaded cached page list: {len(all_pages)} pages (all {len(mentors)} mentors done)", flush=True)
        return all_pages

    print(f"  Loaded {len(all_pages)} pages from checkpoint, {len(remaining)} mentors remaining", flush=True)

    t0 = time.time()
    n = len(remaining)
    for idx, m in enumerate(remaining):
        u = m["username"]
        all_pages[f"User talk:{u}"] = u
        p = {
            "action": "query", "list": "allpages",
            "apprefix": f"{u}/", "apnamespace": "3", "aplimit": "500",
        }
        while True:
            d = api_get(p)
            for pg in d.get("query", {}).get("allpages", []):
                t = pg["title"]
                if any(k in t.lower() for k in ["archive", "archives"]):
                    all_pages[t] = u
            if "continue" in d:
                p.update(d["continue"])
            else:
                break
            throttle.wait()

        if (idx + 1) % 20 == 0 or idx + 1 == n:
            with open(CKPT_PAGELIST, "w") as f:
                json.dump(all_pages, f)
        progress_bar(idx + 1, n, t0, extra=f"pages={len(all_pages)} delay={throttle}")
        throttle.wait()

    print()
    with open(CKPT_PAGELIST, "w") as f:
        json.dump(all_pages, f)
    print(f"  Collected {len(all_pages)} pages for {len(mentors)} mentors")
    return all_pages


# ─── Phase 1C: Search API ───

def load_search_results():
    if CKPT_SEARCH.exists():
        with open(CKPT_SEARCH) as f:
            return [json.loads(l) for l in f if l.strip()]

    results = []
    params = {
        "action": "query", "list": "search",
        "srsearch": '"Question from"',
        "srnamespace": "3", "srlimit": "500",
        "srinfo": "totalhits", "srprop": "size",
    }
    while True:
        d = api_get(params)
        results.extend(d.get("query", {}).get("search", []))
        total = d.get("query", {}).get("searchinfo", {}).get("totalhits", "?")
        print(f"    search: {len(results)}/{total}", flush=True)
        if "continue" in d:
            params.update(d["continue"])
        else:
            break
        throttle.wait()

    with open(CKPT_SEARCH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


# ─── Generic fetch & write loop ───

def fetch_and_write(titles, page_to_mentor, out_path, fetched, label):
    todo = [t for t in titles if t not in fetched]
    if not todo:
        print(f"  [{label}] All {len(titles)} pages already fetched", flush=True)
        return 0, 0

    print(f"  [{label}] {len(todo)} pages to fetch ({len(titles)-len(todo)} already done)", flush=True)

    total_pages = 0
    total_conv = 0
    mc = defaultdict(int)
    total_batches = (len(todo) - 1) // BATCH + 1
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as fout:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            try:
                texts = fetch_batch_wikitext(batch)
            except Exception as e:
                print(f"\n  Batch ERROR: {e}", flush=True)
                continue

            missing = [t for t in batch if t not in texts]
            for t in missing:
                try:
                    d = api_get({"action": "parse", "page": t, "prop": "wikitext"})
                    if "parse" in d:
                        texts[t] = d["parse"]["wikitext"]["*"]
                    throttle.wait()
                except Exception:
                    pass

            fetched_batch = []
            for t, wt in texts.items():
                mentor = page_to_mentor.get(t, extract_mentor_from_title(t))
                if not mentor:
                    continue
                total_pages += 1
                fetched_batch.append(t)

                qs = parse_questions(wt, t, mentor)
                if qs:
                    rec = {
                        "mentor": mentor,
                        "page": t,
                        "len": len(wt),
                        "q_count": len(qs),
                        "wikitext": wt,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_conv += len(qs)
                    mc[mentor] += len(qs)

            for t in batch:
                if t not in fetched_batch and t in texts:
                    fetched_batch.append(t)
            mark_fetched(fetched_batch)
            fetched.update(fetched_batch)

            bn = i // BATCH + 1
            progress_bar(bn, total_batches, t0,
                         extra=f"pages={total_pages} conv={total_conv} delay={throttle}")
            if bn % 20 == 0:
                fout.flush()
            throttle.wait()

    print()
    if mc:
        top = sorted(mc.items(), key=lambda x: -x[1])[:10]
        print(f"  [{label}] Top mentors: {', '.join(f'{n}:{c}' for n,c in top)}", flush=True)

    return total_pages, total_conv


# ─── Phase 2: Fix & Merge ───

def get_all_subpages(username):
    pages = [f"User talk:{username}"]
    params = {
        "action": "query", "list": "allpages",
        "apprefix": f"{username}/", "apnamespace": "3", "aplimit": "500",
    }
    while True:
        d = api_get(params)
        for pg in d.get("query", {}).get("allpages", []):
            pages.append(pg["title"])
        if "continue" in d:
            params.update(d["continue"])
            throttle.wait()
        else:
            break
    return pages


def phase2_fix_and_merge(mentors, fetched):
    s1_names = {m["username"] for m in mentors}

    matched_records = []
    matched_pages = set()
    if OUT_MATCHED.exists():
        with open(OUT_MATCHED, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    matched_records.append(r)
                    matched_pages.add(r.get("page", ""))
                except Exception:
                    pass
    print(f"  Existing matched records: {len(matched_records)}")

    unmatched_records = []
    if OUT_UNMATCHED.exists():
        with open(OUT_UNMATCHED, encoding="utf-8") as f:
            for line in f:
                try:
                    unmatched_records.append(json.loads(line))
                except Exception:
                    pass
    print(f"  Existing unmatched records: {len(unmatched_records)}")

    reclassified = []
    truly_unmatched = []
    for r in unmatched_records:
        if r.get("mentor", "") in s1_names:
            reclassified.append(r)
        else:
            truly_unmatched.append(r)

    reclassified_q = sum(r.get("q_count", 0) for r in reclassified)
    print(f"  Reclassified (unmatched -> matched): {len(reclassified)} pages, {reclassified_q} questions")

    # Step A: Discover ALL subpages
    print(f"\n  {'─'*60}")
    print("  Phase 2A: Discover all subpages for s1 mentors")

    all_known_pages = {}
    done_mentors = set()

    if CKPT_ALLPAGES.exists():
        with open(CKPT_ALLPAGES, encoding="utf-8") as f:
            saved = json.load(f)
        all_known_pages.update(saved)
        done_mentors = set(saved.values())
        print(f"    Checkpoint: {len(all_known_pages)} pages for {len(done_mentors)} mentors")

    remaining = [m for m in mentors if m["username"] not in done_mentors]
    if remaining:
        print(f"    Remaining: {len(remaining)} mentors to discover")
        t0 = time.time()
        for i, m in enumerate(remaining):
            u = m["username"]
            pages = get_all_subpages(u)
            for p in pages:
                all_known_pages[p] = u
            done_mentors.add(u)
            if (i + 1) % 50 == 0 or i + 1 == len(remaining):
                with open(CKPT_ALLPAGES, "w", encoding="utf-8") as f:
                    json.dump(all_known_pages, f, ensure_ascii=False)
                print(f"    [{i+1}/{len(remaining)}] {len(all_known_pages)} pages ({fmt_time(time.time()-t0)})", flush=True)
            throttle.wait()

    print(f"    Total s1-mentor pages: {len(all_known_pages)}")

    # Find pages needing fetch
    already_have = set(matched_pages)
    for r in reclassified:
        already_have.add(r.get("page", ""))

    need_fetch = []
    for page, mentor in all_known_pages.items():
        if page not in already_have and page not in fetched:
            need_fetch.append((page, mentor))

    print(f"    Pages never fetched: {len(need_fetch)}")

    # Step B: Fetch new pages
    new_records = []
    b_checked_pages = set()

    if CKPT_NEWDATA.exists():
        with open(CKPT_NEWDATA, encoding="utf-8") as f:
            for line in f:
                try:
                    new_records.append(json.loads(line))
                except Exception:
                    pass

    if CKPT_BFETCHED.exists():
        with open(CKPT_BFETCHED, encoding="utf-8") as f:
            b_checked_pages = set(l.strip() for l in f if l.strip())

    need_fetch = [(p, m) for p, m in need_fetch if p not in b_checked_pages]

    if need_fetch:
        print(f"\n  Phase 2B: Fetching {len(need_fetch)} new pages")
        throttle.delay = 1.5
        throttle.floor = 0.8

        total_batches = (len(need_fetch) - 1) // BATCH + 1
        t0 = time.time()
        new_q = sum(r.get("q_count", 0) for r in new_records)

        for bi in range(0, len(need_fetch), BATCH):
            batch = need_fetch[bi:bi + BATCH]
            titles = [p for p, m in batch]
            page_to_mentor = {p: m for p, m in batch}

            try:
                texts = fetch_batch_wikitext(titles)
            except Exception as e:
                if "429" in str(e):
                    time.sleep(60)
                continue

            for t in titles:
                if t not in texts:
                    try:
                        d = api_get({"action": "parse", "page": t, "prop": "wikitext"})
                        if "parse" in d:
                            texts[t] = d["parse"]["wikitext"]["*"]
                        throttle.wait()
                    except Exception:
                        pass

            batch_new = []
            for t, wt in texts.items():
                mentor = page_to_mentor.get(t)
                if not mentor or not wt:
                    continue
                qs = parse_questions(wt, t, mentor)
                if qs:
                    rec = {"mentor": mentor, "page": t, "len": len(wt),
                           "q_count": len(qs), "wikitext": wt}
                    new_records.append(rec)
                    batch_new.append(rec)
                    new_q += len(qs)

            if batch_new:
                with open(CKPT_NEWDATA, "a", encoding="utf-8") as f:
                    for r in batch_new:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            with open(CKPT_BFETCHED, "a", encoding="utf-8") as f:
                for t in titles:
                    if t in texts:
                        f.write(t + "\n")
                        b_checked_pages.add(t)

            bn = bi // BATCH + 1
            print(f"\r    [{bn}/{total_batches}] new_pages={len(new_records)} new_q={new_q} ({fmt_time(time.time()-t0)})", end="", flush=True)
            throttle.wait()
        print()

    # Step C: Merge and deduplicate
    print(f"\n  Phase 2C: Merge and deduplicate")
    all_records = []
    seen_pages = set()

    for r in matched_records:
        page = r.get("page", "")
        if page not in seen_pages:
            seen_pages.add(page)
            all_records.append(r)

    n_recl = 0
    for r in reclassified:
        page = r.get("page", "")
        if page not in seen_pages:
            seen_pages.add(page)
            all_records.append(r)
            n_recl += 1

    n_new = 0
    for r in new_records:
        page = r.get("page", "")
        if page not in seen_pages:
            seen_pages.add(page)
            all_records.append(r)
            n_new += 1

    total_q = sum(r.get("q_count", 0) for r in all_records)
    total_mentors = len(set(r.get("mentor", "") for r in all_records))

    with open(OUT_MERGED, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_UNMATCHED_CLEAN, "w", encoding="utf-8") as f:
        for r in truly_unmatched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"    MERGED: {len(all_records)} pages, {total_q} questions, {total_mentors} mentors")
    print(f"    Reclassified: +{n_recl}, Newly fetched: +{n_new}")
    print(f"    -> {OUT_MERGED}")

    # Write report
    lines = [
        "=" * 70,
        "  s2 DATA MERGE REPORT",
        "=" * 70, "",
        f"  Matched records:     {len(matched_records)}",
        f"  Unmatched records:   {len(unmatched_records)}",
        f"  Reclassified:        {n_recl} pages ({reclassified_q} questions)",
        f"  Newly fetched:       {n_new} pages",
        f"  MERGED TOTAL:        {len(all_records)} pages, {total_q} questions, {total_mentors} mentors",
        f"  Truly unmatched:     {len(truly_unmatched)} pages",
    ]
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    if not MENTOR_LIST.exists():
        print(f"ERROR: {MENTOR_LIST} not found. Run s1_collect_mentors.py first.")
        sys.exit(1)

    with open(MENTOR_LIST) as f:
        mentors = [json.loads(l) for l in f]
    mentor_set = {m["username"] for m in mentors}
    print(f"Loaded {len(mentors)} mentors from s1", flush=True)

    fetched = load_fetched()

    if not fetched:
        bootstrapped = set()
        for out_path in (OUT_MATCHED, OUT_UNMATCHED):
            if out_path.exists():
                with open(out_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            if r.get("page"):
                                bootstrapped.add(r["page"])
                        except Exception:
                            pass
        if bootstrapped:
            with open(CKPT_FETCHED, "w") as f:
                for t in sorted(bootstrapped):
                    f.write(t + "\n")
            fetched = bootstrapped
            print(f"Checkpoint bootstrapped: {len(fetched)} pages", flush=True)

    # === Phase 1: Collection ===
    print(f"\n{'='*70}")
    print("PHASE 1: Collect conversations")
    print(f"{'='*70}")

    print("\n  Step 1: Collect page titles for known mentors")
    known_pages = get_archive_talk_pages(mentors)

    print("\n  Step 2: Fetch known mentor pages")
    m_pages, m_conv = fetch_and_write(
        list(known_pages.keys()), known_pages, OUT_MATCHED, fetched, "matched")
    print(f"  New: {m_pages} pages, {m_conv} conversations", flush=True)

    print("\n  Step 3: Search API for unmatched pages")
    search_results = load_search_results()
    unmatched_pages = {}
    for r in search_results:
        title = r["title"]
        mentor_name = extract_mentor_from_title(title)
        if mentor_name and mentor_name not in mentor_set:
            unmatched_pages[title] = mentor_name

    print(f"  Unmatched: {len(unmatched_pages)} pages")

    print("\n  Step 4: Fetch unmatched pages")
    u_pages, u_conv = fetch_and_write(
        list(unmatched_pages.keys()), unmatched_pages, OUT_UNMATCHED, fetched, "unmatched")

    # === Phase 2: Fix & Merge ===
    print(f"\n{'='*70}")
    print("PHASE 2: Fix & Merge")
    print(f"{'='*70}")
    phase2_fix_and_merge(mentors, fetched)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"  Final output: {OUT_MERGED}")


if __name__ == "__main__":
    main()
