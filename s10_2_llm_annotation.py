#!/usr/bin/env python3
"""
s10_2_llm_annotation.py — Full corpus LLM annotation using DeepSeek V4 Flash (non-thinking).

Annotates all 41,339 s8 first-turn questions with Morrison (1993) dimensions:
  Q0: Substantive question
  Q2: Referent (lacks task direction)
  Q3: Appraisal (requesting feedback on own work)
  Q4: Normative (seeking rules/norms/permissions)
  Q5: Own work (has prior editing activity)

Features:
  - Reads question text from s8_first_turns.jsonl
  - Uses codebook_prompt.md as system prompt
  - Resume-safe: successful results cached in JSONL, errors tracked separately
  - On re-run: skips successes, retries errors + unannotated
  - Prefix caching: identical system prompt across all calls → automatic cache hit
  - High concurrency (DeepSeek has no rate limit)

Usage:
  python3 s10_2_llm_annotation.py              # run full annotation (retries errors)
  python3 s10_2_llm_annotation.py --check      # check progress only
  python3 s10_2_llm_annotation.py --errors     # only retry error cases
  python3 s10_2_llm_annotation.py --compact    # compact output file (remove dupes/errors)

Output:
  data/s10/corpus_annotations_v2.jsonl         # all results (append-only)
  data/s10/corpus_annotations_v2_errors.jsonl   # error log
"""
import asyncio, json, os, sys, time, re
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv("/Users/Shared/baiduyun/00 Code/0Wiki/.env")

BASE = Path(os.path.dirname(os.path.abspath(__file__)))
S8_FILE = BASE / "data" / "s8" / "s8_first_turns.jsonl"
CODEBOOK = Path("/Users/Shared/baiduyun/00 Code/0Wiki/2026-4/2026-4-24/codebook_prompt.md")
OUT_DIR = BASE / "data" / "s10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "corpus_annotations_v2.jsonl"
ERR_FILE = OUT_DIR / "corpus_annotations_v2_errors.jsonl"

DIMS = ["Q0", "Q2", "Q3", "Q4", "Q5"]
MAX_CONCURRENT = 1000
MAX_RETRIES = 3

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus():
    corpus = {}
    with open(S8_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["conversation_id"]] = d.get("question_clean", "")
    return corpus


def load_cache():
    """Load successful annotations. Only keeps the LAST valid entry per cid."""
    done = {}
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(d in entry for d in DIMS) and "error" not in entry:
                    done[entry["cid"]] = entry
    return done


def load_errors():
    """Load error cids from error log."""
    errors = {}
    if ERR_FILE.exists():
        with open(ERR_FILE, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    errors[entry["cid"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return errors


def compact_output():
    """Remove duplicate/error lines, keep only latest success per cid."""
    if not OUT_FILE.exists():
        print("No output file to compact.")
        return

    done = load_cache()
    backup = OUT_FILE.with_suffix(".jsonl.bak")
    OUT_FILE.rename(backup)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for cid in sorted(done.keys()):
            f.write(json.dumps(done[cid], ensure_ascii=False) + "\n")

    print(f"Compacted: {len(done):,} valid entries (backup: {backup.name})")


# ══════════════════════════════════════════════════════════════════════════════
# LLM call + parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_json_result(raw: str) -> dict | None:
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
    try:
        m = re.search(r'\{[^}]*\}', cleaned)
        if m:
            parsed = json.loads(m.group())
            result = {d: parsed.get(d, "?") for d in DIMS}
            if all(result[d] in ("Y", "N") for d in DIMS):
                return result
    except json.JSONDecodeError:
        pass
    result = {}
    for d in DIMS:
        m2 = re.search(rf'"{d}"\s*:\s*"([YN])"', cleaned)
        if m2:
            result[d] = m2.group(1)
    if len(result) == len(DIMS):
        return result
    return None


async def annotate_one(client, codebook_text: str, question: str):
    messages = [
        {"role": "system", "content": codebook_text},
        {"role": "user", "content": f"Annotate the following message for dimensions Q0, Q2, Q3, Q4, Q5.\n\nMessage:\n{question}"},
    ]

    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                max_completion_tokens=200,
                temperature=0,
            )
            raw = resp.choices[0].message.content
            if not raw:
                last_error = f"empty_response (finish={resp.choices[0].finish_reason})"
                continue
            raw = raw.strip()
            result = parse_json_result(raw)
            if result:
                return result, raw, None
            last_error = f"parse_fail: {raw[:200]}"

        except Exception as e:
            last_error = str(e)[:300]
            if "429" in last_error or "rate" in last_error.lower():
                await asyncio.sleep(2 ** attempt)
                continue
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue

    return None, None, last_error


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    check_only = "--check" in sys.argv
    errors_only = "--errors" in sys.argv
    do_compact = "--compact" in sys.argv

    print("=" * 70)
    print("  s10_2: Full Corpus LLM Annotation (DeepSeek V4 Flash)")
    print("=" * 70)

    if do_compact:
        compact_output()
        return

    # Load
    print("\nLoading corpus from s8...")
    corpus = load_corpus()
    print(f"  Total questions: {len(corpus):,}")

    print("Loading cache...")
    done = load_cache()
    print(f"  Successfully annotated: {len(done):,}")

    prev_errors = load_errors()
    print(f"  Previous errors logged: {len(prev_errors):,}")

    if errors_only:
        # Only retry cases that errored last time
        todo_cids = [cid for cid in prev_errors if cid not in done and cid in corpus]
        print(f"  Retrying errors: {len(todo_cids):,}")
    else:
        # All unannotated (includes errors + never attempted)
        todo_cids = [cid for cid in corpus if cid not in done]
        print(f"  Remaining (unannotated + errors): {len(todo_cids):,}")

    if check_only:
        if done:
            counts = {d: {"Y": 0, "N": 0} for d in DIMS}
            for entry in done.values():
                for d in DIMS:
                    v = entry.get(d, "")
                    if v in ("Y", "N"):
                        counts[d][v] += 1
            print(f"\n  Current distribution (N={len(done):,}):")
            for d in DIMS:
                total = counts[d]["Y"] + counts[d]["N"]
                pct = counts[d]["Y"] / total * 100 if total > 0 else 0
                print(f"    {d}: Y={counts[d]['Y']:,} ({pct:.1f}%)  N={counts[d]['N']:,}")
        n_missing = len(corpus) - len(done)
        print(f"\n  Missing: {n_missing:,} / {len(corpus):,}")
        return

    if not todo_cids:
        print("\n  All questions already annotated!")
        return

    # Load codebook
    codebook_text = CODEBOOK.read_text("utf-8")
    print(f"  Codebook: {len(codebook_text):,} chars")

    # Setup client
    client = AsyncOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        timeout=120,
    )

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()
    stats = {"done": 0, "success": 0, "errors": 0, "total": len(todo_cids)}
    t0 = time.time()

    out_f = open(OUT_FILE, "a", encoding="utf-8")
    err_f = open(ERR_FILE, "a", encoding="utf-8")

    async def process_one(cid):
        question = corpus[cid]

        async with sem:
            result, raw, error = await annotate_one(client, codebook_text, question)

        async with lock:
            if result is not None:
                entry = {"cid": cid}
                for d in DIMS:
                    entry[d] = result[d]
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                stats["success"] += 1
            else:
                err_entry = {
                    "cid": cid,
                    "error": error or "unknown",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                err_f.write(json.dumps(err_entry, ensure_ascii=False) + "\n")
                err_f.flush()
                stats["errors"] += 1

            stats["done"] += 1
            if stats["done"] % 200 == 0 or stats["done"] == stats["total"]:
                elapsed = time.time() - t0
                rate = stats["done"] / elapsed if elapsed > 0 else 0
                eta = (stats["total"] - stats["done"]) / rate if rate > 0 else 0
                print(f"  [{stats['done']:,}/{stats['total']:,}] "
                      f"ok={stats['success']:,}  err={stats['errors']}  "
                      f"rate={rate:.1f}/s  ETA={eta/60:.1f}min", flush=True)

    print(f"\nStarting annotation (concurrency={MAX_CONCURRENT})...")
    # Process in batches to avoid creating too many coroutines at once
    BATCH_SIZE = 2000
    for batch_start in range(0, len(todo_cids), BATCH_SIZE):
        batch = todo_cids[batch_start:batch_start + BATCH_SIZE]
        await asyncio.gather(*[process_one(cid) for cid in batch])

    out_f.close()
    err_f.close()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Complete: {stats['success']:,} success, {stats['errors']} errors")
    print(f"  Time: {elapsed:.0f}s ({stats['done']/elapsed:.1f}/s)")
    print(f"  Output: {OUT_FILE}")
    if stats["errors"] > 0:
        print(f"  Errors: {ERR_FILE}")
        print(f"  → Re-run with --errors to retry failed cases")
    print(f"{'=' * 70}")

    # Final distribution
    all_done = load_cache()
    counts = {d: {"Y": 0, "N": 0} for d in DIMS}
    for entry in all_done.values():
        for d in DIMS:
            v = entry.get(d, "")
            if v in ("Y", "N"):
                counts[d][v] += 1
    print(f"\n  Final distribution (N={len(all_done):,}):")
    for d in DIMS:
        total = counts[d]["Y"] + counts[d]["N"]
        pct = counts[d]["Y"] / total * 100 if total > 0 else 0
        print(f"    {d}: Y={counts[d]['Y']:,} ({pct:.1f}%)  N={counts[d]['N']:,}")

    n_missing = len(corpus) - len(all_done)
    if n_missing > 0:
        print(f"\n  Still missing: {n_missing:,} — run again with --errors to retry")


if __name__ == "__main__":
    asyncio.run(main())
