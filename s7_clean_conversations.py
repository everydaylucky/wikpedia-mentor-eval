#!/usr/bin/env python3
"""
s7_clean_conversations.py — Clean s5 conversation wikitext for analysis.

Reads data/s5/s5_all_conversations.jsonl, applies wikitext→plaintext cleaning
(signatures, timestamps, markup, templates, links), then produces
embedding-ready text.

Two-level cleaning:
  1. clean_wikitext()  — remove wiki markup, signatures, templates → plain text with semantic tokens
  2. make_emb_text()   — remove usernames, residual signatures → embedding-ready text

Reply truncation: keeps only the mentor's first signed reply block.
Reply timestamp extraction: from mentor's signature in reply_raw.

Output: data/s7/s7_conversations_cleaned.jsonl
"""
import json, re
from pathlib import Path

BASE = Path(__file__).parent
INPUT = BASE / "data" / "s5" / "s5_all_conversations.jsonl"
OUTPUT = BASE / "data" / "s7" / "s7_conversations_cleaned.jsonl"

TS_RE = re.compile(r'\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)', re.I)
USER_LINK_RE = re.compile(r'\[\[User(?:[_ ]talk)?:([^\]|#]+)[^\]]*\]\]', re.I)


MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_wiki_timestamp(ts_text):
    """Parse '01:01, 6 May 2024 (UTC)' → '2024-05-06T01:01:00Z'."""
    m = re.match(r'(\d{2}):(\d{2}),\s*(\d{1,2})\s+(\w+)\s+(\d{4})', ts_text)
    if not m:
        return None
    hh, mm, day, month_str, year = m.groups()
    month = MONTH_MAP.get(month_str.lower())
    if not month:
        return None
    return f"{year}-{month:02d}-{int(day):02d}T{hh}:{mm}:00Z"


def _find_signed_timestamps(raw):
    """Return list of (ts_end_pos, signer_name_lower, ts_iso) for each timestamp with a nearby User link."""
    results = []
    for ts_match in TS_RE.finditer(raw):
        window = raw[max(0, ts_match.start() - 500):ts_match.start()]
        user_matches = list(USER_LINK_RE.finditer(window))
        if user_matches:
            signer = user_matches[-1].group(1).strip().replace("_", " ").lower()
            ts_iso = _parse_wiki_timestamp(ts_match.group())
            results.append((ts_match.end(), signer, ts_iso))
    return results


def extract_reply_timestamp(reply_raw, mentor_name):
    """Extract the mentor's reply timestamp from reply_raw."""
    if not reply_raw:
        return None
    mentor_norm = mentor_name.strip().replace("_", " ").lower()
    sigs = _find_signed_timestamps(reply_raw)
    for _end, signer, ts_iso in sigs:
        if signer == mentor_norm:
            return ts_iso
    for ts_match in TS_RE.finditer(reply_raw):
        return _parse_wiki_timestamp(ts_match.group())
    return None


def truncate_reply_to_mentor_first(reply_raw, mentor_name):
    """Keep only the mentor's first signed reply, cutting off subsequent turns."""
    if not reply_raw:
        return reply_raw
    mentor_norm = mentor_name.strip().replace("_", " ").lower()
    sigs = _find_signed_timestamps(reply_raw)
    if len(sigs) < 2:
        return reply_raw
    for end, signer, _ts in sigs:
        if signer == mentor_norm:
            return reply_raw[:end]
    return reply_raw

# ── Signature / timestamp patterns ──

DT_SIG = re.compile(
    r'\[\[User(?:\s+talk)?:[^\]]*?#c-[^\]]*?\|'
    r'(\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\))\]\]'
)

SIG_FULL = re.compile(
    r'--\s*\[\[User:[^\]]+\]\]'
    r'\s*\(\[\[User talk:[^\]]+\|talk\]\]\)\s*'
    r'\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)',
    re.I
)

SIG_FULL_2 = re.compile(
    r'\[\[User:[^\]]+\|[^\]]+\]\]'
    r'\s*\(\[\[User talk:[^\]]+\|talk\]\]\)\s*'
    r'\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)',
    re.I
)

# ── Embedding-prep patterns ──

SIG_PATTERNS = re.compile(
    r'Qwerfjkl\w*'
    r'|Fritzmann\b'
    r'|(?<!\w)ABG\b'
    r'|Destinyokhiria\b'
    r'|Toadette\b'
    r"|Muffin\([^)]*\)"
    r'|Tenshi!\s*\([^)]*\)'
    r"|Just'i'yaya"
    r"|Intentionally'?Dense\s*\([^)]*\)"
    r'|cyberdog\d+\w*'
    r'|thetechie@enwiki\b'
    r'|Lightbluerain[^(]*'
    r'|Hennedits\|label=Henn'
    r'|Cowboygilbert'
    r'|Polygnotus'
    r'|Bbb23'
    r'|Quentin23J'
    r'|64andtim'
    , re.I
)

SIG_PARENS = re.compile(
    r'\(she/her[^)]*\)'
    r'|\(he/him[^)]*\)'
    r'|\(she/they[^)]*\)'
    r'|\(they/them[^)]*\)'
    r'|\(talk\s*/\s*cont(?:rib)?s?\)'
    r'|\(Talk/[Rr]eport[^)]*\)'
    r'|\(message me\)'
    r'|\(lets? chat\)'
    r'|\(talk\s*·\s*(?:he|she|they)/[^)]*\)'
    r'|\(Talk\s+Contribs\)'
    r'|\(top\s*[•·]\s*contribs?\)'
    r'|\(t\s*[•·;]\s*c\)'
    r'|\(T,\s*C(?:,\s*L)?\)'
    r'|\(t\?\s*-\s*c\)'
    r'|\(Chat\)\s*\([^)]*\)'
    r'|\(talk/contributions\)'
    r'|\(talk\s*\|\s*contribs?\)'
    r'|\(user/tlk/ctrbs\)'
    r'|\(discuss\s*[•·]\s*contribs\)'
    r'|\(chat\s+to\s+me[^)]*\)'
    r'|\(yap\)\s*\|\s*\(things[^)]*\)'
    r'|\(contribs\)[🔥]*'
    r'|\(DM[^)]*\)\s*:3'
    r'|\(🐾,\s*⛈\)'
    r'|\(WAVEDASH\)'
    r'|\(2025[^)]*\)'
    r"|\(Let's talk\s*·\s*📜My work\)"
    r'|\(Merry Christmas[^)]*\)'
    r'|\(Spreading democracy[^)]*\)'
    r'|\(Talk page\)'
    , re.I
)

EXTRA_SIG = re.compile(
    r'ping mewhen u reply'
    r'|⟲\s*@MENTOR'
    r'|Signed,\s*@MENTOR'
    r'|🌙Eclipse\b[^.!?\n]*'
    r'|🌀@MENTOR'
    r'|Geoff Who, me\?'
    , re.I
)


def clean_wikitext(text):
    """Remove wikitext markup, signatures, timestamps → plain text."""
    text = DT_SIG.sub(r'\1', text)
    text = re.sub(SIG_FULL, '', text)
    text = re.sub(SIG_FULL_2, '', text)
    text = re.sub(
        r'--\s*\[\[User:[^\]]+\|[^\]]+\]\]\s*'
        r'\(\[\[User talk:[^\]]+\|[^\]]+\]\]\)\s*'
        r'\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)',
        '', text
    )
    text = re.sub('[​-‏ - ⁠﻿]', '', text)
    # ── Templates ──
    text = re.sub(r'\{\{[Cc]lear\}\}', '', text)
    text = re.sub(r'\{\{(?:tpw|tps)\|[^}]*\}\}', '', text)
    text = re.sub(r'\{\{(?:pb|snd|parabr|-|!|pipe)\}\}', ' ', text)
    text = re.sub(r'\{\{(?:ping|re|yo|replyto|u|reply(?:\s*to)?)\|([^}]+)\}\}', r'\1', text, flags=re.I)
    text = re.sub(r'\{\{(?:smiley|smiley2)\}\}', '', text)
    text = re.sub(r'\{\{(?:ivmbox|ombox)[^}]*\}\}', '[BOT_NOTICE]', text)
    text = re.sub(r'\{\{(?:tq|tqq|tquote|quote box)\|([^}]+)\}\}', r'[QUOTE] \1 [/QUOTE]', text, flags=re.I)
    text = re.sub(r'\{\{(?:tl|tlx|tlc|tls)\|([^}|]+)[^}]*\}\}', r'[TEMPLATE_REF]', text, flags=re.I)
    text = re.sub(r'\{\{(?:done|resolved|not\s*done)\}\}', '[STATUS]', text, flags=re.I)
    text = re.sub(r'\{\{(?:talk\s*page\s*stalker)[^}]*\}\}', '', text, flags=re.I)
    text = re.sub(r'\{\{(?:cite\s*web|cite\s*news|cite\s*book|cite\s*journal)[^}]*\}\}', '[CITATION]', text, flags=re.I)
    text = re.sub(r'\{\{(?:reflist|refs)[^}]*\}\}', '', text, flags=re.I)
    text = re.sub(r'\{\{(?:subst:)?submit\}\}', '', text, flags=re.I)
    text = re.sub(r'\{\{[^}]{200,}\}\}', '[TEMPLATE]', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # ── Links: User/Special (remove entirely, including display text) ──
    text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)
    text = re.sub(r'\[\[User(?:[_ ]talk)?:[^\]]*?\|talk\s*page\]\]', '[TALK_PAGE]', text, flags=re.I)
    text = re.sub(
        r'\(\s*\[\[User(?:[_ ]talk)?:[^\]]*?\|talk\]\]'
        r'\s*[·•]\s*'
        r'\[\[Special:Contrib[^\]]*?\|contribs?\]\]'
        r'\s*\)',
        '', text, flags=re.I
    )
    text = re.sub(r'\[\[User(?:[_ ]talk)?:[^\]]*?\]\]', '', text, flags=re.DOTALL | re.I)
    text = re.sub(r'\[\[Special:[^\]]*?\]\]', '', text, flags=re.I)
    text = re.sub(r'Special:Contrib(?:ution)?s/[\w\-. ]+', '', text, flags=re.I)
    # ── Strip HTML tags ──
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\bUser(?:[_ ]talk)?:[\w\-. ]+(?:/\w+)*', '', text, flags=re.I)
    # ── Links: Policy / Help / Draft / Category / Template (→ tokens) ──
    text = re.sub(r'\[\[(?:WP|Wikipedia|Wikipedia talk|W|Wp|wikipedia):[^\]|]+\|([^\]]+)\]\]', r'[POLICY]', text, flags=re.I)
    text = re.sub(r'\[\[(?:WP|Wikipedia|Wikipedia talk|W|Wp|wikipedia):[^\]]+\]\]', r'[POLICY]', text, flags=re.I)
    text = re.sub(r'\[\[(?:MOS|MOS talk):[^\]|]+\|([^\]]+)\]\]', r'[POLICY]', text, flags=re.I)
    text = re.sub(r'\[\[(?:MOS|MOS talk):[^\]]+\]\]', r'[POLICY]', text, flags=re.I)
    text = re.sub(r'\[\[(?:Help|H|HELP):[^\]|]+\|([^\]]+)\]\]', r'[HELP_PAGE]', text, flags=re.I)
    text = re.sub(r'\[\[(?:Help|H|HELP):[^\]]+\]\]', r'[HELP_PAGE]', text, flags=re.I)
    text = re.sub(r'\[\[Draft:[^\]|]+\|([^\]]+)\]\]', r'[DRAFT]', text, flags=re.I)
    text = re.sub(r'\[\[Draft:[^\]]+\]\]', r'[DRAFT]', text, flags=re.I)
    text = re.sub(r'\[\[(?:Category):[^\]|]+\|([^\]]+)\]\]', r'\1', text, flags=re.I)
    text = re.sub(r'\[\[(?:Category):[^\]]+\]\]', '', text, flags=re.I)
    text = re.sub(r'\[\[(?:Template|Template talk):[^\]|]+\|([^\]]+)\]\]', r'[TEMPLATE_REF]', text, flags=re.I)
    text = re.sub(r'\[\[(?:Template|Template talk):[^\]]+\]\]', r'[TEMPLATE_REF]', text, flags=re.I)
    text = re.sub(r'\[\[(?:Talk):[^\]|]+\|([^\]]+)\]\]', r'[WIKILINK]', text, flags=re.I)
    text = re.sub(r'\[\[(?:Talk):[^\]]+\]\]', r'[WIKILINK]', text, flags=re.I)
    # ── Links: interwiki ──
    text = re.sub(r'\[\[:?[a-z]{2,3}:[^\]|]+\|([^\]]+)\]\]', r'[WIKILINK]', text)
    text = re.sub(r'\[\[:?[a-z]{2,3}:[^\]]+\]\]', r'[WIKILINK]', text)
    text = re.sub(r'\[\[(?:commons|m|c|wikt|s|n|b|q|v):[^\]|]+\|([^\]]+)\]\]', r'[WIKILINK]', text, flags=re.I)
    text = re.sub(r'\[\[(?:commons|m|c|wikt|s|n|b|q|v):[^\]]+\]\]', r'[WIKILINK]', text, flags=re.I)
    # ── Links: normal article links ──
    text = re.sub(r'\[\[\]\]', '', text)
    text = re.sub(r'(?<!\[)\b[\w\-. ]{3,30}\|[\w\-. ]{2,20}(?!\])', lambda m: m.group().split('|')[-1], text)
    text = re.sub(r'\[\[:?([^\]|]+)\|([^\]]+)\]\]', r'[WIKILINK]', text)
    text = re.sub(r'\[\[:?([^\]]+)\]\]', r'[WIKILINK]', text)
    # ── Links: external URLs ──
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'[LINK]', text)
    text = re.sub(r'\[https?://[^\s\]]+\]', '[LINK]', text)
    text = re.sub(r'https?://\S+', '[LINK]', text)
    # ── Bare namespace references ──
    text = re.sub(r'\bDraft:[\w\-. /]+', '[DRAFT]', text)
    text = re.sub(r'\b(?:WP|Wikipedia):[\w\-. /]+', '[POLICY]', text, flags=re.I)
    text = re.sub(r'\bHelp:[\w\-. /]+', '[HELP_PAGE]', text, flags=re.I)
    text = re.sub(r'\bMOS:[\w\-. /]+', '[POLICY]', text, flags=re.I)
    # ── Collapse repeated tokens ──
    text = re.sub(r'(\[(?:POLICY|HELP_PAGE|DRAFT|WIKILINK|LINK|TEMPLATE_REF|CITATION)\])(?:\s*\1)+', r'\1', text)
    text = re.sub(r'\[\[\]\]', '', text)
    text = re.sub(r"'{2,5}", '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(
        r'(?:^|(?<=\s)|(?<=[~\-–—]))[\w.·★🍀🥪🦝⚡️\-–—~]{1,40}'
        r'\s*\(talk\)[\s·•]*(?:\(cont(?:ribs?)?\))?'
        r'\s*\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)',
        '', text
    )
    text = re.sub(r'\d{2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(UTC\)', '', text)
    text = re.sub(r'\([💬✏️\s]*talk\s*[·•/]?\s*[💬✏️\s]*contribs?\s*\)', '', text)
    text = re.sub(r'[💬✏️\s]*\btalk\s*[·•/]?\s*[💬✏️\s]*contribs?\b', '', text)
    text = re.sub(r'\(chat\s*[·•]?\s*edits\)', '', text)
    text = re.sub(r'\(talk\)', '', text)
    text = re.sub(r'\([^\w\s]*-?\)', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\(Ping me or leave a message on my talk page\)', '', text, flags=re.I)
    text = re.sub(r'[🍀💬✏️★🥪🦝⚡️]+', '', text)
    text = re.sub(r'\(he/(?:him|they)\)', '', text)
    text = re.sub(r'\(she/her\)', '', text)
    text = re.sub(r'\s*-{1,2}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r"[~\-–—]\s*[\w.]+['']?[\w]*Hello\b", '', text)
    text = re.sub(r'\s*[—–]\s*Preceding unsigned comment added by\b.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\(Talk\)\s*$', '', text, flags=re.I | re.MULTILINE)
    text = re.sub(r'\s*[―–—]\s*[\w\s]{2,30}?Talk(?:[\w\s#.]*)?$', '', text)
    text = re.sub(r"(?:^|\s)Lord'serious'pig\s*$", '', text, flags=re.MULTILINE)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Protect emoticons like :) :( :D :P :/ before stripping wikitext indent colons
        protected = re.sub(r':([)D(P/|\\])', r'EMOTICON_COLON\1', line)
        stripped = protected.lstrip(':*').strip()
        stripped = stripped.replace('EMOTICON_COLON', ':')
        if stripped:
            cleaned_lines.append(stripped)
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def make_emb_text(clean_text, mentee, mentor):
    """Further clean for embedding input: remove @mentions, residual signatures."""
    t = clean_text
    if mentee:
        t = re.sub(r'@?' + re.escape(mentee) + r'(?=[\s,;:!?\.\)]|$)', '', t, flags=re.I)
        t = re.sub(r'@?' + re.escape(mentee.replace('_', ' ')) + r'(?=[\s,;:!?\.\)]|$)', '', t, flags=re.I)
    if mentor:
        t = re.sub(r'@?' + re.escape(mentor) + r'(?=[\s,;:!?\.\)]|$)', '', t, flags=re.I)
        t = re.sub(r'@?' + re.escape(mentor.replace('_', ' ')) + r'(?=[\s,;:!?\.\)]|$)', '', t, flags=re.I)
    if mentee:
        t = re.sub(r'\(?username\s+' + re.escape(mentee) + r'\)?', '', t, flags=re.I)
        t = re.sub(r'\(?username\s+' + re.escape(mentee.replace('_', ' ')) + r'\)?', '', t, flags=re.I)
    t = re.sub(r'\(\s*username\s*\)', '', t, flags=re.I)
    t = re.sub(r'\b\d[\d\- ]{6,}\d\b', '', t)
    t = re.sub(r'<small>Signed,</small>|Signed,', '', t)
    t = re.sub(r'\(Talk\)\s*\|?\s*\([|]?(?:Contributions|Edits|Contribs?)\)', '', t, flags=re.I)
    t = re.sub(r'\|\|\|\|', '', t)
    t = re.sub(r'\bUser(?:[_ ]talk)?:\S*', '', t, flags=re.I)
    t = re.sub(r'Special:Contrib(?:ution)?s\S*', '', t, flags=re.I)
    t = SIG_PATTERNS.sub('', t)
    t = SIG_PARENS.sub('', t)
    t = EXTRA_SIG.sub('', t)
    t = re.sub(r'\(\s*[·•/\s]*\)', '', t)
    t = re.sub(r'\(\s*\[(?:WIKILINK|LINK)\]\s*[·•]\s*\)', '', t)
    t = re.sub(r'\{\s*\}', '', t)
    t = re.sub(r'\[\s*\]', '', t)
    t = re.sub(r'[◇◆♠♣♦♥☆★⚡🌙🌀⟲꧁꧂꧃-꧟᧠-᧿⌘☙❧❦❖✦✧⟡⟢᭄᭢᭣᭤᭥᭦᭧]+', '', t)
    t = re.sub(r'\^U\^', '', t)
    t = re.sub(r'\s*[-–—]\s*[/\\]+\s*$', '', t)
    t = re.sub(r'\s*[-–—]\s*$', '', t)
    t = re.sub(r'[—–~]\s*$', '', t, flags=re.MULTILINE)
    t = re.sub(r'\s*-{1,2}\s*$', '', t, flags=re.MULTILINE)
    t = re.sub(r'Courtesy ping\s*', '', t)
    t = re.sub(r'\w+UwU\s*\(talk\s*/\s*contributions?\)', '', t, flags=re.I)
    t = re.sub(r'}}\s*$', '', t)
    t = re.sub(r'\s*[•·]\s*talk\b', '', t, flags=re.I)
    t = re.sub(r'\btalk\s*$', '', t, flags=re.I | re.MULTILINE)
    t = re.sub(r'〜\s*$', '', t, flags=re.MULTILINE)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'^[\s,;:·•\-–—~]+', '', t)
    t = re.sub(r'[\s,;:·•\-–—~]+$', '', t)
    t = re.sub(r'\s+([,;:!?\.])', r'\1', t)
    t = re.sub(r'([!?.])\s*\.\s*$', r'\1', t)
    t = re.sub(r',\s*\.', '.', t)
    t = re.sub(r'^\.\s*', '', t)
    t = re.sub(r'\s*@\s*$', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'^[\s,;:·•\-–—~]+', '', t)
    t = re.sub(r'[\s,;:·•\-–—~]+$', '', t)
    return t


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data/s5/s5_all_conversations.jsonl...")
    records = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Total conversations: {len(records):,}")

    out_records = []
    empty_q = 0
    empty_r = 0

    for i, r in enumerate(records):
        mentee = r["mentee"]
        mentor = r["mentor"]

        q_raw = r.get("question_text") or ""
        r_raw = r.get("mentor_reply") or ""

        q_raw = re.split(r'\n==.+==\s*\n', q_raw)[0]

        # Strip question text duplicated at start of reply (s4 recovery artifact):
        # Some s4_recovered replies contain the full question + mentee signature
        # before the actual reply. Detect by checking if reply starts with question
        # text (after optional templates), then cut at the mentee's first signature.
        if q_raw and r_raw and len(q_raw) > 20:
            r_no_tpl = re.sub(r'^\{\{[^}]+\}\}\s*', '', r_raw)
            if r_no_tpl.startswith(q_raw[:50]):
                mentee_norm = mentee.strip().replace("_", " ").lower()
                sigs = _find_signed_timestamps(r_raw)
                for end, signer, _ts in sigs:
                    if signer == mentee_norm:
                        r_raw = r_raw[end:].lstrip()
                        break

        reply_ts = extract_reply_timestamp(r_raw, mentor) if r_raw else None

        r_raw = truncate_reply_to_mentor_first(r_raw, mentor)

        q_clean = clean_wikitext(q_raw) if q_raw else ""
        r_clean = clean_wikitext(r_raw) if r_raw else ""

        q_emb = make_emb_text(q_clean, mentee, mentor) if q_clean else ""
        r_emb = make_emb_text(r_clean, mentee, mentor) if r_clean else ""

        if not q_emb:
            empty_q += 1
        if r_raw and not r_emb:
            empty_r += 1

        rec = {
            "conversation_id": i,
            "mentor": mentor,
            "mentee": mentee,
            "revid": r.get("revid"),
            "timestamp": r.get("timestamp"),
            "article": r.get("article"),
            "source": r.get("source"),
            "page": r.get("page"),
            "question_raw": q_raw,
            "question_clean": q_clean,
            "question_emb": q_emb,
            "reply_raw": r_raw,
            "reply_clean": r_clean,
            "reply_emb": r_emb,
            "reply_timestamp": reply_ts,
            "has_reply": bool(r_raw),
            "question_words": len(q_emb.split()) if q_emb else 0,
            "reply_words": len(r_emb.split()) if r_emb else 0,
        }
        out_records.append(rec)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}...", flush=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    has_reply = sum(1 for r in out_records if r["has_reply"])
    print(f"\n  Results:")
    print(f"    Total conversations: {len(out_records):,}")
    print(f"    With mentor reply:   {has_reply:,}")
    print(f"    Empty question after clean: {empty_q:,}")
    print(f"    Empty reply after clean:    {empty_r:,}")
    print(f"    Avg question words:  {sum(r['question_words'] for r in out_records)/len(out_records):.1f}")
    print(f"    Avg reply words:     {sum(r['reply_words'] for r in out_records if r['has_reply'])/max(has_reply,1):.1f}")
    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
