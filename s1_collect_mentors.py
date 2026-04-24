#!/usr/bin/env python3
"""
Collect ALL Wikipedia Growth Team mentors (2021-05 ~ present) directly from API.

Two API sources:
  1. Wikipedia:Growth_Team_features/Mentor_list  (wikitext, 2021-05 ~ 2022-10)
  2. MediaWiki:GrowthMentors.json                (JSON,     2022-10 ~ present)

Checkpoint files (resume-safe):
  data/s1/s1_checkpoint_revisions_wikitext.jsonl
  data/s1/s1_checkpoint_revisions_json.jsonl
  data/s1/s1_checkpoint_uid_map.json

Output:
  data/s1/s1_mentor_list.jsonl   — one row per mentor, complete timeline
  data/s1/s1_mentor_change.jsonl — every state change event
"""
import json, time, re, sys
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data" / "s1"
API_URL = "https://en.wikipedia.org/w/api.php"
UA = "MentorResearch/1.0 (academic research)"

CKPT_WIKI = DATA / "s1_checkpoint_revisions_wikitext.jsonl"
CKPT_JSON = DATA / "s1_checkpoint_revisions_json.jsonl"
CKPT_UID  = DATA / "s1_checkpoint_uid_map.json"

JSON_CUTOFF = "2022-10-26T13:19:10Z"


def api_get(params):
    params["format"] = "json"
    url = f"{API_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def load_checkpoint(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_checkpoint(path, revisions):
    with open(path, "w", encoding="utf-8") as f:
        for r in revisions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_page_revisions(title, ckpt_path):
    existing = load_checkpoint(ckpt_path)
    seen_revids = {r["revid"] for r in existing}
    latest_ts = max((r["timestamp"] for r in existing), default=None) if existing else None

    print(f"  Checkpoint: {len(existing)} revisions cached", end="")
    if latest_ts:
        print(f" (up to {latest_ts})")
    else:
        print()

    new_revs = []
    rvcontinue = None
    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "ids|timestamp|user|comment|content",
            "rvslots": "main",
            "rvlimit": "50",
            "rvdir": "newer",
        }
        if latest_ts:
            params["rvstart"] = latest_ts
        if rvcontinue:
            params["rvcontinue"] = rvcontinue

        data = api_get(params)
        for pid, pdata in data.get("query", {}).get("pages", {}).items():
            for rev in pdata.get("revisions", []):
                if rev["revid"] not in seen_revids:
                    new_revs.append(rev)
                    seen_revids.add(rev["revid"])

        cont = data.get("continue", {})
        if "rvcontinue" in cont:
            rvcontinue = cont["rvcontinue"]
            if len(new_revs) % 200 == 0 and len(new_revs) > 0:
                print(f"    fetched {len(new_revs)} new revisions...")
            time.sleep(0.3)
        else:
            break

    all_revs = existing + new_revs
    all_revs.sort(key=lambda r: r.get("timestamp", ""))
    save_checkpoint(ckpt_path, all_revs)
    print(f"  Total: {len(all_revs)} revisions ({len(new_revs)} new)")
    return all_revs


def load_uid_cache():
    if not CKPT_UID.exists():
        return {}
    with open(CKPT_UID, encoding="utf-8") as f:
        return json.load(f)


def save_uid_cache(cache):
    with open(CKPT_UID, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def resolve_usernames(names, cache):
    need = [n for n in names if n not in cache]
    if not need:
        return cache

    print(f"  Resolving {len(need)} usernames via API...")
    for i in range(0, len(need), 50):
        batch = need[i:i+50]
        params = {
            "action": "query",
            "list": "users",
            "ususers": "|".join(batch),
            "usprop": "editcount",
        }
        data = api_get(params)
        for u in data.get("query", {}).get("users", []):
            name = u.get("name", "")
            uid = u.get("userid")
            if name and uid:
                cache[name] = uid
            elif name:
                cache[name] = None
        time.sleep(0.3)

    failed = [n for n in need if cache.get(n) is None]
    if failed:
        for n in failed:
            print(f"    WARNING: could not resolve '{n}'")

    save_uid_cache(cache)
    return cache


def norm_user(u):
    if not u:
        return ""
    return u.replace("_", " ").replace(" ", " ").strip()


def parse_wikitext_mentors(wikitext):
    mentors = []
    for line in wikitext.split('\n'):
        line = line.strip()
        if not line.startswith('*'):
            continue
        match = re.search(r'\[\[User:([^\]|]+)', line, re.IGNORECASE)
        if match:
            username = match.group(1).strip()
            msg_match = re.search(r'\]\]\|(.+)', line)
            if not msg_match:
                msg_match = re.search(r'\]\]\s*[-–—]\s*(.+)', line)
            if not msg_match:
                msg_match = re.search(r'\]\](.+)', line)
            message = msg_match.group(1).strip() if msg_match else ""
            mentors.append({"username": username, "message": message})
    return mentors


def parse_json_mentors(content_str):
    try:
        data = json.loads(content_str)
    except Exception:
        return {}
    md = data.get("Mentors", data)
    if isinstance(md, dict):
        return {k: v for k, v in md.items() if isinstance(v, dict)}
    return {}


def get_content(rev):
    return rev.get("slots", {}).get("main", {}).get("*", rev.get("*", ""))


def replay_wikitext(revisions, uid_map):
    changes = []
    prev_names = set()

    for rev in revisions:
        ts = rev.get("timestamp", "")
        if ts >= JSON_CUTOFF:
            break
        editor = rev.get("user", "")
        comment = rev.get("comment", "")
        content = get_content(rev)
        if not content:
            continue

        cur_mentors = parse_wikitext_mentors(content)
        cur_names = {norm_user(m["username"]) for m in cur_mentors}

        if not cur_names:
            if prev_names:
                for name in sorted(prev_names):
                    uid = uid_map.get(name)
                    if uid is None:
                        continue
                    changes.append({
                        "timestamp": ts, "user_id": uid, "username": name,
                        "event": "left", "source": "wikitext",
                        "editor": editor, "comment": comment,
                    })
                prev_names = set()
            continue

        for name in sorted(cur_names - prev_names):
            uid = uid_map.get(name)
            if uid is None:
                continue
            md = next((m for m in cur_mentors if norm_user(m["username"]) == name), {})
            changes.append({
                "timestamp": ts, "user_id": uid, "username": name,
                "event": "joined", "weight": 1,
                "message": md.get("message", ""),
                "source": "wikitext",
                "editor": editor, "comment": comment,
            })

        for name in sorted(prev_names - cur_names):
            uid = uid_map.get(name)
            if uid is None:
                continue
            changes.append({
                "timestamp": ts, "user_id": uid, "username": name,
                "event": "left", "source": "wikitext",
                "editor": editor, "comment": comment,
            })

        prev_names = cur_names

    return changes


def replay_json(revisions):
    uid_to_names = {}
    changes = []
    prev_state = {}

    for rev in revisions:
        ts = rev.get("timestamp", "")
        editor = rev.get("user", "")
        comment = rev.get("comment", "")
        content = get_content(rev)
        if not content:
            continue

        cur_state = parse_json_mentors(content)

        for uid in sorted(set(cur_state) - set(prev_state)):
            props = cur_state[uid]
            uname = props.get("username", "")
            if uname:
                uid_to_names[uid] = uname
            changes.append({
                "timestamp": ts, "user_id": int(uid),
                "username": uname or uid_to_names.get(uid, f"UID:{uid}"),
                "event": "joined",
                "weight": props.get("weight"),
                "message": props.get("message", ""),
                "auto_assigned": props.get("automaticallyAssigned"),
                "source": "json",
                "editor": editor, "comment": comment,
            })

        for uid in sorted(set(prev_state) - set(cur_state)):
            old = prev_state[uid]
            uname = old.get("username", uid_to_names.get(uid, f"UID:{uid}"))
            changes.append({
                "timestamp": ts, "user_id": int(uid),
                "username": uname,
                "event": "left",
                "weight": old.get("weight"),
                "source": "json",
                "editor": editor, "comment": comment,
            })

        for uid in sorted(set(cur_state) & set(prev_state)):
            old_p, new_p = prev_state[uid], cur_state[uid]
            uname = new_p.get("username", old_p.get("username", ""))
            if uname:
                uid_to_names[uid] = uname
            display = uname or uid_to_names.get(uid, f"UID:{uid}")

            if old_p.get("weight") != new_p.get("weight"):
                changes.append({
                    "timestamp": ts, "user_id": int(uid),
                    "username": display,
                    "event": "weight_change",
                    "weight_from": old_p.get("weight"),
                    "weight_to": new_p.get("weight"),
                    "source": "json",
                    "editor": editor, "comment": comment,
                })

            if old_p.get("automaticallyAssigned") != new_p.get("automaticallyAssigned"):
                changes.append({
                    "timestamp": ts, "user_id": int(uid),
                    "username": display,
                    "event": "auto_assigned_change",
                    "from": old_p.get("automaticallyAssigned"),
                    "to": new_p.get("automaticallyAssigned"),
                    "source": "json",
                    "editor": editor, "comment": comment,
                })

        prev_state = cur_state

    return changes, prev_state, uid_to_names


def weight_to_pool(w):
    if w is None:
        return "unknown"
    return "auto" if w >= 1 else "manual"


def build_mentor_list(all_changes, json_last_state, uid_to_names_json):
    mentor_events = defaultdict(list)
    mentor_username = {}

    for c in all_changes:
        uid = c["user_id"]
        mentor_events[uid].append(c)
        uname = c.get("username", "")
        if uname and not uname.startswith("UID:"):
            mentor_username[uid] = uname

    for uid_str, uname in uid_to_names_json.items():
        uid = int(uid_str)
        if uname:
            mentor_username[uid] = uname

    current_uids = set(int(k) for k in json_last_state.keys())

    mentors = []
    for uid in sorted(mentor_events.keys()):
        evts = mentor_events[uid]
        username = mentor_username.get(uid, f"UID:{uid}")
        is_current = uid in current_uids

        joins = [e["timestamp"] for e in evts if e["event"] == "joined"]
        leaves = [e["timestamp"] for e in evts if e["event"] == "left"]
        first_joined = joins[0] if joins else None
        last_left = leaves[-1] if leaves else None

        weight_history = []
        for e in evts:
            if e["event"] == "joined" and "weight" in e:
                w = e["weight"]
                weight_history.append({
                    "timestamp": e["timestamp"], "weight": w,
                    "pool_status": weight_to_pool(w),
                    "source": e.get("source", ""),
                })
            elif e["event"] == "weight_change":
                w = e["weight_to"]
                weight_history.append({
                    "timestamp": e["timestamp"], "weight": w,
                    "pool_status": weight_to_pool(w),
                    "weight_from": e.get("weight_from"),
                    "source": e.get("source", ""),
                })
            elif e["event"] == "left":
                weight_history.append({
                    "timestamp": e["timestamp"], "weight": None,
                    "pool_status": "left",
                    "source": e.get("source", ""),
                })

        cur_weight = None
        cur_pool = None
        if is_current:
            uid_str = str(uid)
            if uid_str in json_last_state:
                cur_weight = json_last_state[uid_str].get("weight")
                cur_pool = weight_to_pool(cur_weight)

        sources = sorted(set(e.get("source", "") for e in evts))
        exited = not is_current and len(leaves) > 0

        ever_auto = any(h["pool_status"] == "auto" for h in weight_history)
        ever_manual = any(h["pool_status"] == "manual" for h in weight_history)

        rec = {
            "user_id": uid,
            "username": username,
            "is_current": is_current,
            "exited": exited,
            "current_weight": cur_weight,
            "current_pool_status": cur_pool,
            "ever_auto": ever_auto,
            "ever_manual": ever_manual,
            "first_joined": first_joined,
            "last_left": last_left,
            "join_count": len(joins),
            "leave_count": len(leaves),
            "weight_history": weight_history,
            "sources": sources,
            "total_events": len(evts),
        }
        mentors.append(rec)

    return mentors


def build_change_log(all_changes, json_last_state):
    current_uids = set(int(k) for k in json_last_state.keys())
    active = set()
    records = []

    for c in all_changes:
        uid = c["user_id"]
        evt = c["event"]
        rec = {
            "timestamp": c["timestamp"],
            "user_id": uid,
            "username": c.get("username", ""),
            "event": evt,
            "source": c.get("source", ""),
        }

        if evt == "joined":
            w = c.get("weight")
            rec["weight"] = w
            rec["pool_status"] = weight_to_pool(w)
            rec["message"] = c.get("message", "")
            active.add(uid)
        elif evt == "left":
            rec["weight"] = c.get("weight")
            rec["pool_status"] = "left"
            active.discard(uid)
            rec["exited"] = uid not in current_uids and uid not in active
        elif evt == "weight_change":
            rec["weight_from"] = c.get("weight_from")
            rec["weight_to"] = c.get("weight_to")
            rec["pool_status_from"] = weight_to_pool(c.get("weight_from"))
            rec["pool_status_to"] = weight_to_pool(c.get("weight_to"))
        elif evt == "auto_assigned_change":
            rec["from"] = c.get("from")
            rec["to"] = c.get("to")

        records.append(rec)

    return records


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Fetch wikitext revisions (Wikipedia:Growth Team features/Mentor list)")
    print("=" * 70)
    wiki_revs = fetch_page_revisions(
        "Wikipedia:Growth Team features/Mentor list", CKPT_WIKI)

    print()
    print("=" * 70)
    print("STEP 2: Fetch JSON revisions (MediaWiki:GrowthMentors.json)")
    print("=" * 70)
    json_revs = fetch_page_revisions(
        "MediaWiki:GrowthMentors.json", CKPT_JSON)

    print()
    print("=" * 70)
    print("STEP 3: Resolve wikitext usernames -> user_id")
    print("=" * 70)
    all_wiki_names = set()
    for rev in wiki_revs:
        content = get_content(rev)
        if content:
            for m in parse_wikitext_mentors(content):
                all_wiki_names.add(norm_user(m["username"]))

    json_name_to_uid = {}
    for rev in json_revs:
        content = get_content(rev)
        if not content:
            continue
        for uid_str, props in parse_json_mentors(content).items():
            uname = props.get("username", "")
            if uname:
                json_name_to_uid[uname] = int(uid_str)

    uid_cache = load_uid_cache()
    for name in all_wiki_names:
        if name in json_name_to_uid and name not in uid_cache:
            uid_cache[name] = json_name_to_uid[name]

    need_resolve = [n for n in all_wiki_names if n not in uid_cache]
    if need_resolve:
        uid_cache = resolve_usernames(need_resolve, uid_cache)
    else:
        print(f"  All {len(all_wiki_names)} wikitext usernames already resolved")

    uid_map = {n: uid for n, uid in uid_cache.items() if uid is not None}
    print(f"  Usable mappings: {len(uid_map)}/{len(all_wiki_names)}")

    print()
    print("=" * 70)
    print("STEP 4: Replay wikitext revisions")
    print("=" * 70)
    wiki_changes = replay_wikitext(wiki_revs, uid_map)
    print(f"  {len(wiki_changes)} change events (before {JSON_CUTOFF[:10]})")

    print()
    print("=" * 70)
    print("STEP 5: Replay JSON revisions")
    print("=" * 70)
    json_changes, json_last_state, uid_to_names_json = replay_json(json_revs)
    print(f"  {len(json_changes)} change events")

    print()
    print("=" * 70)
    print("STEP 6: Merge & build outputs")
    print("=" * 70)
    all_changes = wiki_changes + json_changes
    all_changes.sort(key=lambda x: (x.get("timestamp", ""), x.get("user_id", 0)))
    print(f"  Merged: {len(wiki_changes)} wikitext + {len(json_changes)} JSON = {len(all_changes)} total")

    mentors = build_mentor_list(all_changes, json_last_state, uid_to_names_json)
    list_path = DATA / "s1_mentor_list.jsonl"
    with open(list_path, "w", encoding="utf-8") as f:
        for rec in mentors:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(mentors)} mentors -> {list_path}")

    change_log = build_change_log(all_changes, json_last_state)
    change_path = DATA / "s1_mentor_change.jsonl"
    with open(change_path, "w", encoding="utf-8") as f:
        for rec in change_log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(change_log)} change events -> {change_path}")

    n_current = sum(1 for m in mentors if m["is_current"])
    n_exited = sum(1 for m in mentors if m["exited"])
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique mentors:   {len(mentors)}")
    print(f"  Currently active:     {n_current}")
    print(f"  Exited (left & gone): {n_exited}")
    print(f"Total change events:    {len(change_log)}")


if __name__ == "__main__":
    main()
