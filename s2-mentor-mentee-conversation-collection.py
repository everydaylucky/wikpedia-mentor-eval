#!/usr/bin/env python3
"""
Fetch mentor-mentee conversations from Wikipedia User talk pages.

Reads: s1_mentor_list.jsonl (from s1-mentor-collection.py)
Fetches: User talk pages + archives for each mentor, plus search API for unmatched

Output (raw wikitext per page, with parsed question metadata):
  s2_mentor_conversation_matched.jsonl   — pages belonging to known mentors
  s2_mentor_conversation_unmatched.jsonl — search results not matching any known mentor

Checkpoint files (resume-safe, prefixed s2_):
  s2_checkpoint_fetched_pages.txt    — one page title per line
  s2_checkpoint_page_list.json       — mentor → [page titles] mapping cache
  s2_checkpoint_search_results.jsonl — search API results cache

Import mode: on first run, imports existing data from old pipeline if available.
"""
import json, re, time, ssl, sys, os
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from collections import defaultdict

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
API = "https://en.wikipedia.org/w/api.php"
UA = "MentorResearch/1.0 (academic research)"
BATCH = 10

DELAY_MIN = 0.8
DELAY_MAX = 30.0
DELAY_INIT = 1.5


class AdaptiveThrottle:
    def __init__(self, init=DELAY_INIT, lo=DELAY_MIN, hi=DELAY_MAX):
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


CKPT_FETCHED = BASE / "s2_checkpoint_fetched_pages.txt"
CKPT_PAGELIST = BASE / "s2_checkpoint_page_list.json"
CKPT_SEARCH = BASE / "s2_checkpoint_search_results.jsonl"

OUT_MATCHED = BASE / "s2_mentor_conversation_matched.jsonl"
OUT_UNMATCHED = BASE / "s2_mentor_conversation_unmatched.jsonl"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


# ─── Progress bar ───

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


# ─── API ───

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


# ─── Checkpoint: fetched pages ───

def load_fetched():
    if CKPT_FETCHED.exists():
        with open(CKPT_FETCHED) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def mark_fetched(titles):
    with open(CKPT_FETCHED, "a") as f:
        for t in titles:
            f.write(t + "\n")



# ─── Phase A: Collect page titles for all known mentors ───

def get_all_talk_pages(mentors):
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


# ─── Phase C: Search API for "Question from" ───

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


# ─── Parse questions from wikitext ───

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


# ─── Main ───

def main():
    mentor_path = BASE / "s1_mentor_list.jsonl"
    if not mentor_path.exists():
        print(f"ERROR: {mentor_path} not found. Run s1-mentor-collection.py first.")
        sys.exit(1)

    with open(mentor_path) as f:
        mentors = [json.loads(l) for l in f]
    mentor_set = {m["username"] for m in mentors}
    print(f"Loaded {len(mentors)} mentors from s1_mentor_list.jsonl", flush=True)

    fetched = load_fetched()

    # ── Bootstrap checkpoint from existing output files ──
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
            print(f"Checkpoint bootstrapped from existing output: {len(fetched)} pages", flush=True)
        else:
            print("No existing checkpoint or output found, starting fresh", flush=True)
    else:
        print(f"Checkpoint: {len(fetched)} pages already fetched", flush=True)

    # ── Step 1: Collect known mentor page titles ──
    print()
    print("=" * 70)
    print("STEP 1: Collect page titles for all known mentors")
    print("=" * 70)
    known_pages = get_all_talk_pages(mentors)
    print(f"  Total known mentor pages: {len(known_pages)}", flush=True)

    # ── Step 2: Fetch known mentor pages → matched ──
    print()
    print("=" * 70)
    print("STEP 2: Fetch known mentor pages → s2_mentor_conversation_matched.jsonl")
    print("=" * 70)
    m_pages, m_conv = fetch_and_write(
        list(known_pages.keys()), known_pages, OUT_MATCHED, fetched, "matched")
    print(f"  New: {m_pages} pages, {m_conv} conversations", flush=True)

    # ── Step 3: Search API for unmatched ──
    print()
    print("=" * 70)
    print("STEP 3: Search API for 'Question from' pages")
    print("=" * 70)
    search_results = load_search_results()
    print(f"  Search results: {len(search_results)} pages", flush=True)

    unmatched_pages = {}
    for r in search_results:
        title = r["title"]
        mentor_name = extract_mentor_from_title(title)
        if mentor_name and mentor_name not in mentor_set:
            unmatched_pages[title] = mentor_name

    print(f"  Unmatched pages: {len(unmatched_pages)} ({len(set(unmatched_pages.values()))} unique mentors)", flush=True)

    # ── Step 4: Fetch unmatched pages ──
    print()
    print("=" * 70)
    print("STEP 4: Fetch unmatched pages → s2_mentor_conversation_unmatched.jsonl")
    print("=" * 70)
    u_pages, u_conv = fetch_and_write(
        list(unmatched_pages.keys()), unmatched_pages, OUT_UNMATCHED, fetched, "unmatched")
    print(f"  New: {u_pages} pages, {u_conv} conversations", flush=True)

    # ── Summary ──
    matched_total = sum(1 for _ in open(OUT_MATCHED)) if OUT_MATCHED.exists() else 0
    unmatched_total = sum(1 for _ in open(OUT_UNMATCHED)) if OUT_UNMATCHED.exists() else 0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Matched page records:     {matched_total}")
    print(f"Unmatched page records:   {unmatched_total}")
    print(f"Total pages fetched:      {len(fetched)}")
    print(f"Output:")
    print(f"  {OUT_MATCHED}")
    print(f"  {OUT_UNMATCHED}")

if __name__ == "__main__":
    main()
