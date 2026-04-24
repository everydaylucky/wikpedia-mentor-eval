#!/usr/bin/env python3
"""
s10_perspective_api.py — Fetch Perspective API scores for mentee questions + mentor replies.

Resume-safe: reads existing output CSV to find already-completed conversation_ids,
then continues from where it left off. Also imports old perspective data from
data/legacy/.

Rate limit: ~1 QPS for Perspective API.

Usage:
  python s10_perspective_api.py                    # process all
  python s10_perspective_api.py --type mentee      # mentee questions only
  python s10_perspective_api.py --type mentor      # mentor replies only
  python s10_perspective_api.py --limit 1000       # process N then stop
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

BASE = Path(__file__).parent
DATA = BASE / "data" / "s10"

FIRST_TURNS = BASE / "data" / "s8" / "s8_first_turns.jsonl"

OUT_MENTEE = DATA / "s10_perspective_mentee.csv"
OUT_MENTOR = DATA / "s10_perspective_mentor.csv"

OLD_MENTEE = BASE / "data" / "legacy" / "s7_perspective_mentee_old.csv"
OLD_MENTOR = BASE / "data" / "legacy" / "s7_perspective_mentor_old.csv"

ALL_ATTRIBUTES = [
    "TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT",
    "SEXUALLY_EXPLICIT", "FLIRTATION",
    "AFFINITY_EXPERIMENTAL", "COMPASSION_EXPERIMENTAL", "CURIOSITY_EXPERIMENTAL",
    "NUANCE_EXPERIMENTAL", "PERSONAL_STORY_EXPERIMENTAL", "REASONING_EXPERIMENTAL",
    "RESPECT_EXPERIMENTAL",
    "ATTACK_ON_AUTHOR", "ATTACK_ON_COMMENTER", "INCOHERENT", "INFLAMMATORY",
    "LIKELY_TO_REJECT", "OBSCENE", "SPAM", "UNSUBSTANTIAL",
]

HEADER = ["conversation_id"] + ALL_ATTRIBUTES


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


def load_old_perspective(path):
    data = {}
    if not path.exists():
        return data
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["conversation_id"])
            scores = {}
            has_data = False
            for attr in ALL_ATTRIBUTES:
                val = row.get(attr, "")
                if val and val != "":
                    scores[attr] = val
                    has_data = True
                else:
                    scores[attr] = ""
            if has_data:
                data[cid] = scores
    return data


def load_existing_output(path):
    done = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["conversation_id"])
            has_data = any(row.get(a, "") for a in ALL_ATTRIBUTES)
            if has_data:
                done[cid] = {a: row.get(a, "") for a in ALL_ATTRIBUTES}
    return done


def get_perspective_scores(text, api_key):
    if not text or len(text.strip()) < 2:
        return {attr: "" for attr in ALL_ATTRIBUTES}

    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
    payload = {
        "comment": {"text": text[:15000]},
        "languages": ["en"],
        "requestedAttributes": {attr: {} for attr in ALL_ATTRIBUTES},
        "doNotStore": True,
    }

    for attempt in range(5):
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"\n  429 rate limit, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"\n  API error {resp.status_code}: {resp.text[:100]}", flush=True)
                return {attr: "" for attr in ALL_ATTRIBUTES}
            data = resp.json()
            scores = {}
            for attr in ALL_ATTRIBUTES:
                try:
                    scores[attr] = data["attributeScores"][attr]["summaryScore"]["value"]
                except KeyError:
                    scores[attr] = ""
            return scores
        except requests.exceptions.Timeout:
            print(f"\n  Timeout (attempt {attempt+1}), retrying...", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"\n  Error: {e}", flush=True)
            return {attr: "" for attr in ALL_ATTRIBUTES}

    return {attr: "" for attr in ALL_ATTRIBUTES}


def process_type(conv_type, conversations, api_key, limit):
    if conv_type == "mentee":
        out_path = OUT_MENTEE
        old_path = OLD_MENTEE
        text_field = "question_emb"
        filter_fn = lambda r: r.get("is_first_conversation") and bool(r.get("question_emb"))
    else:
        out_path = OUT_MENTOR
        old_path = OLD_MENTOR
        text_field = "reply_emb"
        filter_fn = lambda r: r.get("is_first_conversation") and r.get("has_reply") and bool(r.get("reply_emb"))

    print(f"\n{'='*60}")
    print(f"Processing: {conv_type}")
    print(f"{'='*60}")

    old_data = load_old_perspective(old_path)
    print(f"  Old perspective data: {len(old_data):,} conversations")

    existing = load_existing_output(out_path)
    print(f"  Already in output:   {len(existing):,} conversations")

    all_done = {}
    all_done.update(old_data)
    all_done.update(existing)

    to_process = []
    for r in conversations:
        cid = r["conversation_id"]
        if cid in all_done:
            continue
        if not filter_fn(r):
            continue
        to_process.append(r)

    eligible = sum(1 for r in conversations if filter_fn(r))
    print(f"  Eligible conversations:  {eligible:,}")
    print(f"  Already done (old+new):  {len(all_done):,}")
    print(f"  Need API calls:          {len(to_process):,}")

    if limit:
        to_process = to_process[:limit]
        print(f"  Limited to:              {limit:,}")

    if not to_process and not existing:
        print(f"  Writing old data to {out_path.name}...")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
            for cid in sorted(all_done.keys()):
                row = {"conversation_id": cid}
                row.update(all_done[cid])
                writer.writerow(row)
        print(f"  Wrote {len(all_done):,} rows")
        return

    if not out_path.exists():
        print(f"  Initializing {out_path.name} with old data...")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
            for cid in sorted(old_data.keys()):
                row = {"conversation_id": cid}
                row.update(old_data[cid])
                writer.writerow(row)
        print(f"  Wrote {len(old_data):,} old rows")

    est_time = len(to_process) * 1.05
    print(f"\n  Estimated time: {est_time/3600:.1f}h ({len(to_process):,} calls @ ~1 QPS)")
    print(f"  Starting API calls...\n")

    t0 = time.time()
    done_count = 0
    err_count = 0

    for i, r in enumerate(to_process):
        cid = r["conversation_id"]
        text = r[text_field]

        scores = get_perspective_scores(text, api_key)

        has_score = any(v != "" for v in scores.values())
        if not has_score:
            err_count += 1

        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            row = {"conversation_id": cid}
            row.update(scores)
            writer.writerow(row)

        done_count += 1
        time.sleep(1.0)

        if (i + 1) % 100 == 0 or i < 3:
            elapsed = time.time() - t0
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - done_count) / rate if rate > 0 else 0
            print(f"\r  [{done_count}/{len(to_process)}] err={err_count} "
                  f"({elapsed/60:.1f}m, ETA={eta/3600:.1f}h)    ",
                  end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n\n  Done: {done_count:,} API calls in {elapsed/60:.1f}m, {err_count} errors")

    final = load_existing_output(out_path)
    print(f"  Total in output: {len(final):,}")


def main():
    load_env()
    api_key = os.environ.get("PERSPECTIVE_API_KEY")
    if not api_key:
        print("Error: PERSPECTIVE_API_KEY not found in .env")
        return

    DATA.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["mentee", "mentor", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not FIRST_TURNS.exists():
        print(f"Error: {FIRST_TURNS} not found. Run s8_extract_first_turns.py first.")
        return

    print("Loading data/s8/s8_first_turns.jsonl...")
    conversations = []
    with open(FIRST_TURNS, encoding="utf-8") as f:
        for line in f:
            conversations.append(json.loads(line))
    print(f"  Total conversations: {len(conversations):,}")

    if args.type in ("mentee", "both"):
        process_type("mentee", conversations, api_key, args.limit or None)
    if args.type in ("mentor", "both"):
        process_type("mentor", conversations, api_key, args.limit or None)

    print("\nAll done.")


if __name__ == "__main__":
    main()
