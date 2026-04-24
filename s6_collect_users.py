#!/usr/bin/env python3
"""
s6_collect_users.py — Collect comprehensive user data for all mentors & mentees.

Reads data/s5/s5_all_conversations.jsonl to get all unique users, then fetches:
  Phase 1: User profiles (registration, editcount, groups, gender, block status)
  Phase 2: User contributions (edit history, 2020-01-01 ~ present)
  Phase 3: Log events (blocks, rights changes, account creation)
  Phase 4: Abuse filter log (edit filter triggers)

All phases are resume-safe with per-user checkpoints.
Uses bot authentication for higher API rate limits + concurrent workers.

Output:
  data/s6/s6_user_profiles.jsonl
  data/s6/s6_user_contribs.jsonl
  data/s6/s6_user_logevents.jsonl
  data/s6/s6_user_abuselog.jsonl

Usage:
  python s6_collect_users.py                  # run all phases
  python s6_collect_users.py --phase 1        # run only Phase 1
  python s6_collect_users.py --phase 2        # run only Phase 2
  python s6_collect_users.py --workers 5      # set concurrent workers (default 3)
"""
import argparse, json, os, ssl, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE = Path(__file__).parent
DATA = BASE / "data" / "s6"
API = "https://en.wikipedia.org/w/api.php"
UA = "WikiMentorResearch/1.0 (academic; Bobocicada@mentor_research)"

S5_FILE = BASE / "data" / "s5" / "s5_all_conversations.jsonl"
S1_FILE = BASE / "data" / "s1" / "s1_mentor_list.jsonl"

OUT_PROFILES   = DATA / "s6_user_profiles.jsonl"
OUT_CONTRIBS   = DATA / "s6_user_contribs.jsonl"
OUT_LOGEVENTS  = DATA / "s6_user_logevents.jsonl"
OUT_ABUSELOG   = DATA / "s6_user_abuselog.jsonl"

CKPT_PROFILES  = DATA / "s6_checkpoint_profiles.txt"
CKPT_CONTRIBS  = DATA / "s6_checkpoint_contribs.txt"
CKPT_LOGEVENTS = DATA / "s6_checkpoint_logevents.txt"
CKPT_ABUSELOG  = DATA / "s6_checkpoint_abuselog.txt"

REPORT_FILE = DATA / "s6_collection_report.txt"

CTX = ssl.create_default_context()

CONTRIBS_START = "2020-01-01T00:00:00Z"
PER_THREAD_DELAY = 0.5


# ─── Bot authentication ─────────────────────────────────────────────────────

def load_env():
    for env_path in [BASE / ".env", BASE.parent / ".env",
                     Path("/Users/Shared/baiduyun/00 Code/0Wiki/.env")]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break


_bot_tokens = threading.local()


def bot_login():
    params = {
        "action": "query", "meta": "tokens", "type": "login", "format": "json",
    }
    url = f"{API}?{urlencode(params)}"
    for _ in range(5):
        try:
            r = urlopen(Request(url, headers={"User-Agent": UA}), context=CTX, timeout=30)
            break
        except HTTPError as e:
            if e.code == 429:
                print("  429 during login, waiting 30s...", flush=True)
                time.sleep(30)
            else:
                raise
    else:
        print("  Login failed after retries", flush=True)
        return None
    cookies = {}
    for h in r.headers.get_all("Set-Cookie") or []:
        parts = h.split(";")[0].split("=", 1)
        if len(parts) == 2:
            cookies[parts[0].strip()] = parts[1].strip()
    data = json.loads(r.read())
    login_token = data["query"]["tokens"]["logintoken"]

    post_data = urlencode({
        "action": "login",
        "lgname": "Bobocicada@mentor_research",
        "lgpassword": os.environ.get("WIKI_BOT_PASSWORD", ""),
        "lgtoken": login_token,
        "format": "json",
    }).encode()
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
    req = Request(API, data=post_data, headers={"User-Agent": UA, "Cookie": cookie_str})
    r2 = urlopen(req, context=CTX, timeout=30)
    for h in r2.headers.get_all("Set-Cookie") or []:
        parts = h.split(";")[0].split("=", 1)
        if len(parts) == 2:
            cookies[parts[0].strip()] = parts[1].strip()
    login_result = json.loads(r2.read())
    if login_result.get("login", {}).get("result") != "Success":
        print(f"  Bot login failed: {login_result}", flush=True)
        return None
    return cookies


_global_cookies = None
_cookies_lock = threading.Lock()


def get_cookies():
    global _global_cookies
    if _global_cookies is None:
        with _cookies_lock:
            if _global_cookies is None:
                _global_cookies = bot_login()
                if _global_cookies:
                    print("  Bot login successful", flush=True)
                else:
                    print("  Bot login failed, falling back to anonymous", flush=True)
    return _global_cookies


# ─── Per-thread rate-limited API ─────────────────────────────────────────────

_last_req = threading.local()
_global_429_backoff = 0.0
_backoff_lock = threading.Lock()


def api_get(params, max_retries=8):
    global _global_429_backoff
    params["format"] = "json"
    url = f"{API}?{urlencode(params)}"
    cookies = get_cookies()
    headers = {"User-Agent": UA}
    if cookies:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())

    for attempt in range(max_retries):
        now = time.time()
        freeze_until = _global_429_backoff
        if now < freeze_until:
            time.sleep(freeze_until - now)
        since = time.time() - getattr(_last_req, "t", 0)
        if since < PER_THREAD_DELAY:
            time.sleep(PER_THREAD_DELAY - since)

        try:
            r = urlopen(Request(url, headers=headers), context=CTX, timeout=60)
            _last_req.t = time.time()
            return json.loads(r.read())
        except HTTPError as e:
            _last_req.t = time.time()
            if e.code == 429:
                wait = 5 * (2 ** attempt)
                with _backoff_lock:
                    _global_429_backoff = max(_global_429_backoff, time.time() + wait)
                print(f"\n  429 (attempt {attempt+1}/{max_retries}), all threads freeze {wait:.0f}s", flush=True)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise
        except Exception as e:
            _last_req.t = time.time()
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise
    return {}


# ─── Helpers ──────────────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def load_checkpoint(path):
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    done.add(u)
    return done


def mark_done(path, username):
    with open(path, "a", encoding="utf-8") as f:
        f.write(username + "\n")


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def fmt_time(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


def load_all_users(mentors_only=False):
    mentors = set()
    mentees = set()

    if S5_FILE.exists():
        with open(S5_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                mentors.add(r["mentor"])
                mentees.add(r["mentee"])

    if S1_FILE.exists():
        with open(S1_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                mentors.add(r["username"])

    if mentors_only:
        return sorted(mentors), mentors, mentees

    all_users = sorted(mentors | mentees)
    return all_users, mentors, mentees


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: User Profiles (batch 50)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_user_profiles_batch(usernames):
    params = {
        "action": "query",
        "list": "users",
        "ususers": "|".join(usernames),
        "usprop": "blockinfo|editcount|registration|groups|implicitgroups|rights|gender|emailable|centralids",
    }
    d = api_get(params)
    results = {}
    for u in d.get("query", {}).get("users", []):
        name = u.get("name", "")
        if name:
            results[name] = u
    return results


def phase1_profiles(users, mentors_set, mentees_set):
    print(f"\n{'='*70}")
    print("PHASE 1: User Profiles")
    print(f"{'='*70}")

    done = load_checkpoint(CKPT_PROFILES)
    remaining = [u for u in users if u not in done]
    print(f"Total users: {len(users)}, done: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All profiles already collected.")
        return

    t0 = time.time()
    batch_size = 50
    total_batches = (len(remaining) - 1) // batch_size + 1

    for bi in range(0, len(remaining), batch_size):
        batch = remaining[bi:bi + batch_size]
        try:
            profiles = fetch_user_profiles_batch(batch)
        except Exception as e:
            print(f"\n  Batch error: {e}", flush=True)
            time.sleep(30)
            continue

        for username in batch:
            profile = profiles.get(username, {})
            role = []
            if username in mentors_set:
                role.append("mentor")
            if username in mentees_set:
                role.append("mentee")

            rec = {
                "username": username,
                "role": role,
                "userid": profile.get("userid"),
                "editcount": profile.get("editcount"),
                "registration": profile.get("registration"),
                "groups": profile.get("groups", []),
                "implicitgroups": profile.get("implicitgroups", []),
                "gender": profile.get("gender", "unknown"),
                "emailable": "emailable" in profile,
                "missing": "missing" in profile,
                "blocked": "blockid" in profile,
                "block_reason": profile.get("blockreason"),
                "block_expiry": profile.get("blockexpiry"),
                "block_by": profile.get("blockedby"),
                "centralids": profile.get("centralids", {}),
            }
            append_jsonl(OUT_PROFILES, rec)
            mark_done(CKPT_PROFILES, username)

        bn = bi // batch_size + 1
        elapsed = time.time() - t0
        eta = elapsed / bn * (total_batches - bn) if bn > 0 else 0
        print(f"\r  [{bn}/{total_batches}] ({bn/total_batches*100:.1f}%) "
              f"elapsed={fmt_time(elapsed)} ETA={fmt_time(eta)}",
              end="", flush=True)

    print(f"\n  Phase 1 complete. {len(remaining)} profiles collected.")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: User Contributions — full pagination, concurrent
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_user_contribs(username, start=CONTRIBS_START):
    edits = []
    params = {
        "action": "query",
        "list": "usercontribs",
        "ucuser": username,
        "uclimit": "500",
        "ucprop": "ids|title|timestamp|comment|size|sizediff|tags|flags",
        "ucdir": "newer",
        "ucstart": start,
    }
    while True:
        d = api_get(params)
        for c in d.get("query", {}).get("usercontribs", []):
            edits.append({
                "revid": c.get("revid"),
                "parentid": c.get("parentid"),
                "ns": c.get("ns"),
                "title": c.get("title"),
                "timestamp": c.get("timestamp"),
                "comment": c.get("comment", ""),
                "size": c.get("size"),
                "sizediff": c.get("sizediff"),
                "tags": c.get("tags", []),
                "minor": "minor" in c,
                "new": "new" in c,
            })
        if "continue" in d:
            params.update(d["continue"])
        else:
            break
    return edits


def phase2_contribs(users, mentors_set, mentees_set, n_workers=4):
    print(f"\n{'='*70}")
    print("PHASE 2: User Contributions (Edit History)")
    print(f"{'='*70}")

    done = load_checkpoint(CKPT_CONTRIBS)
    remaining = [u for u in users if u not in done]
    print(f"Total users: {len(users)}, done: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All contributions already collected.")
        return

    t0 = time.time()
    completed = [0]
    total_edits = [0]
    errors = [0]
    failed_users = []
    fail_lock = threading.Lock()

    def process_user(username):
        is_mentor = username in mentors_set
        try:
            edits = fetch_user_contribs(username)
        except Exception as e:
            with fail_lock:
                errors[0] += 1
                failed_users.append(username)
            if errors[0] <= 5:
                print(f"\n  {username} ERROR: {e}", flush=True)
            return

        rec = {
            "username": username,
            "role": "mentor" if is_mentor else "mentee",
            "edit_count": len(edits),
            "edits": edits,
        }
        with _write_lock:
            append_jsonl(OUT_CONTRIBS, rec)
            mark_done(CKPT_CONTRIBS, username)
            completed[0] += 1
            total_edits[0] += len(edits)
            n = completed[0]

        if n % 200 == 0 or n <= 5 or len(edits) > 1000:
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - n) / rate if rate > 0 else 0
            print(f"\r  [{n}/{len(remaining)}] {username}: {len(edits)} edits  "
                  f"total={total_edits[0]:,}  err={errors[0]}  "
                  f"({fmt_time(elapsed)}, ETA={fmt_time(eta)})",
                  end="", flush=True)

    print(f"  {n_workers} concurrent workers, {PER_THREAD_DELAY}s/req per thread")
    print(f"  Warming up with single-threaded test...", flush=True)

    if remaining:
        process_user(remaining[0])
        if errors[0] > 0 and completed[0] == 0:
            print(f"\n  First request failed — API may be rate-limiting your IP.")
            print(f"  Wait a few minutes and retry. Or use --workers 1 for slower but safer mode.")
            return
        print(f"  Warm-up OK, starting concurrent...", flush=True)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, u in enumerate(remaining[1:], 1):
            futures[pool.submit(process_user, u)] = u
            if i <= n_workers:
                time.sleep(1.0)
        try:
            for f in as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            print(f"\n  Interrupted. {completed[0]} users saved (resume-safe).")
            pool.shutdown(wait=False, cancel_futures=True)
            return

    elapsed = time.time() - t0
    print(f"\n  Phase 2 complete. {completed[0]} users, {total_edits[0]:,} edits, "
          f"{errors[0]} errors in {fmt_time(elapsed)}.")
    if failed_users:
        fail_path = DATA / "s6_failed_contribs.txt"
        print(f"  Failed users saved to {fail_path}")
        with open(fail_path, "w") as f:
            f.write("\n".join(failed_users) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Log Events — concurrent
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_user_logevents(username):
    events = []
    for le_type in ["block", "rights", "newusers"]:
        params = {
            "action": "query",
            "list": "logevents",
            "letitle": f"User:{username}",
            "letype": le_type,
            "lelimit": "500",
            "leprop": "ids|title|type|user|timestamp|comment|details",
        }
        while True:
            d = api_get(params)
            for evt in d.get("query", {}).get("logevents", []):
                events.append({
                    "logid": evt.get("logid"),
                    "type": evt.get("type"),
                    "action": evt.get("action"),
                    "timestamp": evt.get("timestamp"),
                    "user": evt.get("user"),
                    "title": evt.get("title"),
                    "comment": evt.get("comment", ""),
                    "params": evt.get("params", {}),
                    "direction": "target",
                })
            if "continue" in d:
                params.update(d["continue"])
            else:
                break
    return events


def phase3_logevents(users, mentors_set, mentees_set, n_workers=4):
    print(f"\n{'='*70}")
    print("PHASE 3: Log Events (blocks, rights, account creation)")
    print(f"{'='*70}")

    done = load_checkpoint(CKPT_LOGEVENTS)
    remaining = [u for u in users if u not in done]
    print(f"Total users: {len(users)}, done: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All log events already collected.")
        return

    t0 = time.time()
    completed = [0]
    total_events = [0]
    errors = [0]

    def process_user(username):
        try:
            events = fetch_user_logevents(username)
        except Exception as e:
            errors[0] += 1
            print(f"\n  {username} ERROR: {e}", flush=True)
            return

        rec = {
            "username": username,
            "event_count": len(events),
            "has_block": any(e["type"] == "block" for e in events),
            "has_rights_change": any(e["type"] == "rights" for e in events),
            "events": events,
        }
        with _write_lock:
            append_jsonl(OUT_LOGEVENTS, rec)
            mark_done(CKPT_LOGEVENTS, username)
            completed[0] += 1
            total_events[0] += len(events)
            n = completed[0]

        if n % 100 == 0 or n <= 3:
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - n) / rate if rate > 0 else 0
            print(f"\r  [{n}/{len(remaining)}] events={total_events[0]:,}  err={errors[0]}  "
                  f"({fmt_time(elapsed)}, ETA={fmt_time(eta)})",
                  end="", flush=True)

    print(f"  {n_workers} concurrent workers")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, u in enumerate(remaining):
            futures[pool.submit(process_user, u)] = u
            if i < n_workers:
                time.sleep(1.0)
        try:
            for f in as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            print(f"\n  Interrupted. {completed[0]} users saved.")
            pool.shutdown(wait=False, cancel_futures=True)
            return

    elapsed = time.time() - t0
    print(f"\n  Phase 3 complete. {completed[0]} users, {total_events[0]:,} events "
          f"in {fmt_time(elapsed)}.")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Abuse Filter Log — concurrent
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_user_abuselog(username):
    entries = []
    params = {
        "action": "query",
        "list": "abuselog",
        "afluser": username,
        "afllimit": "500",
        "aflprop": "ids|filter|user|title|action|result|timestamp",
    }
    while True:
        d = api_get(params)
        for entry in d.get("query", {}).get("abuselog", []):
            entries.append({
                "id": entry.get("id"),
                "filter_id": entry.get("filter_id"),
                "filter": entry.get("filter"),
                "timestamp": entry.get("timestamp"),
                "title": entry.get("title"),
                "action": entry.get("action"),
                "result": entry.get("result"),
            })
        if "continue" in d:
            params.update(d["continue"])
        else:
            break
    return entries


def phase4_abuselog(users, mentors_set, mentees_set, n_workers=4):
    print(f"\n{'='*70}")
    print("PHASE 4: Abuse Filter Log")
    print(f"{'='*70}")

    done = load_checkpoint(CKPT_ABUSELOG)
    remaining = [u for u in users if u not in done]
    print(f"Total users: {len(users)}, done: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All abuse logs already collected.")
        return

    t0 = time.time()
    completed = [0]
    total_entries = [0]
    errors = [0]

    def process_user(username):
        try:
            entries = fetch_user_abuselog(username)
        except Exception as e:
            errors[0] += 1
            print(f"\n  {username} ERROR: {e}", flush=True)
            return

        rec = {
            "username": username,
            "abuse_count": len(entries),
            "entries": entries,
        }
        with _write_lock:
            append_jsonl(OUT_ABUSELOG, rec)
            mark_done(CKPT_ABUSELOG, username)
            completed[0] += 1
            total_entries[0] += len(entries)
            n = completed[0]

        if n % 100 == 0 or n <= 3:
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - n) / rate if rate > 0 else 0
            print(f"\r  [{n}/{len(remaining)}] abuse={total_entries[0]:,}  err={errors[0]}  "
                  f"({fmt_time(elapsed)}, ETA={fmt_time(eta)})",
                  end="", flush=True)

    print(f"  {n_workers} concurrent workers")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for i, u in enumerate(remaining):
            futures[pool.submit(process_user, u)] = u
            if i < n_workers:
                time.sleep(1.0)
        try:
            for f in as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            print(f"\n  Interrupted. {completed[0]} users saved.")
            pool.shutdown(wait=False, cancel_futures=True)
            return

    elapsed = time.time() - t0
    print(f"\n  Phase 4 complete. {completed[0]} users, {total_entries[0]:,} entries "
          f"in {fmt_time(elapsed)}.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(users, mentors_set, mentees_set):
    lines = [
        f"s6_collect_users report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"Total unique users:  {len(users)}",
        f"  Mentors:           {len(mentors_set)}",
        f"  Mentees:           {len(mentees_set)}",
        f"  Both roles:        {len(mentors_set & mentees_set)}",
        "",
    ]

    for label, ckpt, out in [
        ("Profiles", CKPT_PROFILES, OUT_PROFILES),
        ("Contributions", CKPT_CONTRIBS, OUT_CONTRIBS),
        ("Log Events", CKPT_LOGEVENTS, OUT_LOGEVENTS),
        ("Abuse Log", CKPT_ABUSELOG, OUT_ABUSELOG),
    ]:
        done = len(load_checkpoint(ckpt))
        size = out.stat().st_size / 1024 / 1024 if out.exists() else 0
        lines.append(f"{label}: {done}/{len(users)} users  ({size:.1f} MB)")

    report = "\n".join(lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n{report}")


def main():
    load_env()
    DATA.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None,
                    help="Run only this phase (1=profiles, 2=contribs, 3=logevents, 4=abuselog)")
    ap.add_argument("--mentors-only", action="store_true",
                    help="Only collect data for mentors (not mentees)")
    ap.add_argument("--workers", type=int, default=3,
                    help="Concurrent workers (default 3)")
    args = ap.parse_args()

    users, mentors_set, mentees_set = load_all_users(mentors_only=args.mentors_only)
    print(f"Users to collect: {len(users)} ({len(mentors_set)} mentors, {len(mentees_set)} mentees)")

    if args.phase is None or args.phase == 1:
        phase1_profiles(users, mentors_set, mentees_set)
    if args.phase is None or args.phase == 2:
        phase2_contribs(users, mentors_set, mentees_set, n_workers=args.workers)
    if args.phase is None or args.phase == 3:
        phase3_logevents(users, mentors_set, mentees_set, n_workers=args.workers)
    if args.phase is None or args.phase == 4:
        phase4_abuselog(users, mentors_set, mentees_set, n_workers=args.workers)

    write_report(users, mentors_set, mentees_set)


if __name__ == "__main__":
    main()
