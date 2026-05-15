# Wikipedia Mentor Program — Mentee Message Classification Codebook

You are annotating messages sent by new Wikipedia editors (mentees) to their assigned mentors. For each message, assign binary labels (Y/N) across 5 dimensions: Q0, Q2, Q3, Q4, Q5.

**Annotation principles:**
- Base judgments only on the message text itself — do not consider the mentor's reply or subsequent conversation
- Do not infer hidden motives; code only observable features in the text
- When in doubt, default to conservative (N)
- If Q0=N, then Q2/Q3/Q4/Q5 must all be N

---

## Q0: Substantive Question, Request, or Intent to Contribute

Decision tree:

```
Node 1: Does the message contain any question, request, or expression of intent 
        related to Wikipedia editing, contributing, accounts, processes, or community?
  ├─ YES → Q0=Y
  └─ NO → Node 2

Node 2: Does it fall into one of these exhaustive exclusion categories?
         (a) Greeting only — no further content
         (b) Thanks only
         (c) Too short, gibberish, or incomprehensible
         (d) Not Wikipedia-related — other platforms, general life questions,
             asking whether Wikipedia pays editors, or requesting content 
             types that Wikipedia does not host (Wikipedia is an encyclopedia;
             requests that misunderstand its purpose belong here)
         (e) Not in English
         (f) Only asking whether the mentor is a bot, with no other content
         (g) Sharing feelings with no editing reference and no specific question
         (h) Pure self-introduction with no request
  ├─ YES to any → Q0=N
  └─ NO → Q0=Y
```

Test: could the mentor provide a **meaningfully differentiated response**? If the mentor can only reply with a generic welcome template or "this is not a Wikipedia issue," then Q0=N.

Note: username issues, password/account recovery, interface settings (dark mode, gadgets, CSS), and user page editing are all Wikipedia-platform-related → Q0=Y. Mentioning one's own editing activity (even casually) counts as expressing editing intent → Q0=Y.

Edge cases:
- **Article subjects** (people asking about pages about themselves): "Can a Wikipedia page be created about me without my consent?" → Q0=Y (asking about rules). "There are inaccuracies in the article about me" → Q0=Y (requesting content correction). "Someone wrote false information about me, how do I get it taken down?" → Q0=Y.
- **Excitement about contributing vs sharing feelings**: "I'm new here and excited to contribute!" → Q0=Y (expresses contributing intent). "It is my first day to join Wikipedia and I am so excited" → Q0=N (sharing feelings, no editing intent or specific question). "Just made my first edit on a stub article!" → Q0=Y (mentions own editing activity). The key distinction: does the message express intent to edit/contribute, or only share feelings without any editing reference?

---

## Q2 Referent: Lacks Task Direction (Morrison 1993)

Does the user lack task clarity — not knowing what to do, where to focus, or where to begin? This corresponds to Referent information seeking: the newcomer seeks information about role and task requirements.

Decision tree:

```
Node 1: Has the user identified any task at all?
  ├─ NO (completely directionless) → Q2=Y
  └─ YES → Node 2

Node 2: Does the task have a specific topic or content category?
  ├─ YES (specific topic or category) → Q2=N
  └─ NO (only generic terms: article / page / wiki / entry) → Node 3

Node 3: What verb describes the user's intent?
  ├─ create / write / make / start + generic term (no qualifier) → Q2=Y
  ├─ post / publish / submit + generic term → Q2=N (implies existing content)
  └─ Single-step operation → Q2=N
```

**Specific content categories that indicate direction (Q2=N)** — any term one level more specific than "article/page/wiki" suffices:
- People: biography, profile, someone, a person
- Organizations: company, business, brand, agency, church
- Media/culture: film, movie, TV show, series, album, song, book, novel
- Places: place, city, town, country, school, university
- Sports: team, player, athlete
- Wikipedia-specific types: template, userbox, user page, list, redirect, disambiguation, category page, draft, essay, portal, stub, infobox, help page, talk page, WikiProject page
- Domain qualifiers: "contribute on AI/technology," "edit history articles," "interested in history"

**Edge cases for Q2:**
- **"any advice" / "any suggestions"**: If this is the message's **core request** with no specific task identified → Q2=Y ("Any advice you have re editing or creating Wikipedia entries most welcome"). If it follows a description of a specific problem as a **polite closing** → Q2=N ("I went to the page for Joel Hurt and added his grandchild...I'm open to any advice").
- **"don't know where to start" + specific topic**: If the user has a specific topic but says "I don't know where to start" → Q2=N. The "don't know where to start" is asking about process/steps, not lacking direction. Example: "how to create an article about the Battle of Midway but I do not know where to start" → Q2=N (has specific topic).
- **"don't know where to start" without topic** → Q2=Y ("How do I get started editing?" / "Where should I begin?").
- **"what can I do about this" / "what should I do"** in response to a specific problem (e.g., edits being reverted, draft declined): This is asking for a solution to a concrete situation, NOT lacking direction → Q2=N.

**Single-step operations (Q2=N)** — self-contained actions where the mentor can directly teach steps:
- post / publish / submit (implies existing content, asking about publishing process)
- add/upload image, picture, photo
- add citation, reference, source
- add infobox, template, userbox, table, chart
- add link, URL, external link, wikilink
- edit a section, fix an error, correct a mistake
- format text, add bold/italic/heading
- change signature, change username
- move/rename/redirect a page
- add categories/tags
- revert/undo an edit

---

## Q3 Appraisal: Requesting Feedback on Own Work

Is the user requesting evaluation of their own existing work? This corresponds to Appraisal information seeking (Morrison 1993): seeking feedback on one's own performance.

Decision tree:

```
Node 1: Does the user reference their own work on Wikipedia?
         (A draft they wrote, edits they made, a page they created or 
          submitted, or an ongoing editing action with observable results)
  ├─ NO → Q3=N
  └─ YES → Node 2 (Note: passing Node 1 does NOT mean Q3=Y.
           Node 2 and Node 3 still apply. Many messages reference
           own work but end up Q3=N.)

Node 2: Is the user's question or request ABOUT that specific work?
         (As opposed to mentioning the work as background while asking 
          about something else entirely)
  ├─ NO (work is just context; actual question is about process, 
  │      rules, timelines, or an unrelated topic) → Q3=N
  └─ YES → Node 3

Node 3: What does the user want the mentor to do regarding their work?
  ├─ EXAMINE AND JUDGE → Q3=Y
  ├─ PERFORM AN ACTION → Q3=N
  └─ PROVIDE INFORMATION → Q3=N
```

**Node 3 — EXAMINE AND JUDGE (Q3=Y):** The user wants the mentor to look at the work and assess its quality, identify problems, or suggest improvements. The mentor would need to inspect the actual content to respond. This includes:
- **Explicit requests** using evaluation verbs: review, check, look at, let me know, tell me what's wrong, how can I improve, is this okay, is this fine, headed in the right direction, critique, suggestions
- **Implicit requests:** the user describes a problem with their specific work and asks for help understanding or resolving it, where the mentor would need to examine the work's content to give useful guidance. Test: would the mentor need to open and read the user's draft/edits/page to answer?

Implicit request examples (Q3=Y):
- "I submitted my draft and was declined...I'm wondering if I've done something incorrectly after updating?" — mentor needs to look at the updated draft to judge whether the update was correct
- "I reorganized the lead section and added three new sources. Could you take a look and tell me if it reads better now? [LINK]" — provides link + describes changes = implicit request to look and judge
- "for my article I get the feeling that more citations are better but not sure if that is the case: [LINK]" — asking about own work's citation quality with a link

Common Q3=N cases where the user references own work but is NOT requesting evaluation:
- "I keep getting an error when adding a reference" — technical problem, mentor provides troubleshooting steps, not content evaluation → Q3=N
- "I've Drafted an article! Can you help me Publish it?" — requesting action (publish), not evaluation → Q3=N
- "How do I add images to my draft?" — operational question that happens to mention own draft → Q3=N
- "My article was deleted, why?" — asking about rules/enforcement (Q4), not requesting content feedback → Q3=N

**Node 3 — PERFORM AN ACTION (Q3=N):** The user wants the mentor to execute an action — publish, approve, move to mainspace, delete, restore — not to evaluate the work.

**Node 3 — PROVIDE INFORMATION (Q3=N):** The user asks a general question (timeline, process, status, or how-to) that happens to reference their work. The mentor can answer without examining the content itself. Examples:
- "How long does it usually take for a draft to get reviewed?" — asks about timeline, not content quality
- "When will my draft be approved?" — asks about process
- "Is it normal for it to take this long?" — asks about general timeline expectations
- "How do I know if the article I edited has improved?" — asks about evaluation method/criteria, not requesting someone to evaluate. Compare: "Has my article improved?" → Q3=Y

Note: a single message can contain both PROVIDE INFORMATION and EXAMINE AND JUDGE requests. For example: "Is it normal for it to take this long? I'm wondering if I've done something incorrectly after updating" — the first sentence is Q3=N (timeline), but the second is Q3=Y (implicit request to check the work). When any part of the message requests examination, Q3=Y.

Note: Q3 and Q4 are independent. A message can be Q3=Y AND Q4=Y simultaneously. Example: "My draft was declined for lack of reliable sources. I added three new references — can you check if it meets the criteria now?" — Q3=Y (requesting review of updated work) AND Q4=Y ("declined" = enforcement outcome).

**Node 3 — Guidance on a specific piece of work:** When the user asks for advice or guidance (advise, best way to proceed, how should I) about publishing, submitting, or otherwise advancing a specific piece of work they have produced, apply the implicit-request test: would the mentor need to open and read the user's content to give a meaningful answer? If yes → EXAMINE AND JUDGE → Q3=Y. If the mentor can answer with generic process steps without inspecting the content → PROVIDE INFORMATION → Q3=N.

**Additional Node 2 clarifications:**
- If the user's primary purpose is **complaint or dispute** (expressing frustration about what happened to their work rather than seeking improvement suggestions), Q3=N even if they reference their work extensively. Example: "you noted that my contribution lacked notability. I disagree..." — this is contesting a previous judgment, not requesting new feedback.
- If the user mentions their work only as **background context** while asking an unrelated question, Q3=N. Example: "I created an account to edit an Article and create one about Marcel Bonin. I know that Wikipedia needs sources and I wondered how all that works" — mentions a project but asks about general process (how sources work), not about their specific work's quality.

---

## Q4 Normative: Seeking Information About Rules, Norms, or Permissions

Does the mentee's text contain **observable signals** of seeking information about rules, norms, or permissions? This corresponds to Normative information seeking (Morrison 1993): seeking information about expected behaviors in the organization.

**Core principle:** Q4 codes the mentee's **observable information-seeking behavior**, not what the mentor's answer would cover. Even if a mentor would proactively raise policy concerns (e.g., COI when a mentee wants to edit their company's page), Q4=N unless the mentee's own language contains normative-seeking signals. Do NOT use your knowledge of Wikipedia policies to infer normative intent that the mentee did not express.

Decision tree:

```
Node 1: Has the user's work been affected by a policy enforcement action?
         (deleted, reverted, blocked, declined, rejected, removed, 
          warned, undone, locked, semi-protected, not published,
          not incorporated, not accepted, disappeared)
  ├─ YES → Node 2
  └─ NO → Node 3

Node 2: Is the user's primary request CENTERED ON the enforcement outcome
         itself — or does the user pivot to a completely different request?
  ├─ CENTERED ON the enforcement outcome → Q4=Y
  │  This includes ALL of:
  │  (a) Explicitly asking why it happened
  │  (b) Asking how to resolve or reverse it
  │  (c) Expressing confusion about what went wrong ("what happened?",
  │      "what shall I do?", "what am I doing wrong?")
  │  (d) Simply reporting the enforcement with no other request — the 
  │      implicit question is still "why / what do I do about this"
  │  (Resolving any enforcement outcome necessarily requires the mentor 
  │   to explain the underlying rule or norm that triggered it.)
  │  Note: Q4=Y here is independent of Q3. A message can simultaneously
  │  seek feedback on work (Q3=Y) AND involve normative issues (Q4=Y).
  └─ PIVOTS to a completely different request (the enforcement is 
     background context, not the focus) → proceed to Node 3, 
     evaluating only the pivoted request.
     PIVOT example: "my first article was deleted (I was not 
     done)...Can you look at my NEW article?" → the deletion is 
     background; the actual request is reviewing a different, new 
     article → evaluate via Node 3.
     NOT a pivot: "my draft was declined, can you help me fix it 
     and resubmit?" → still centered on the declined draft → Q4=Y

Node 3: Does the mentee's text contain any normative-seeking signal
         from the lists below?
  ├─ YES → Q4=Y
  └─ NO → Q4=N
```

**Signals that indicate Q4=Y:**

*Normative-seeking language:*
- allowed, permitted, rules, policy, guidelines, terms
- violations, violating
- consent, authorize
- permission
- prohibited, banned, blocked (when asking about the rule, not the technical state)
- appropriate, legitimate
- ok/okay (in compliance context, not casual speech)
- qualify, eligible, criteria
- compliant, compliance
- legal, illegal
- proper/properly (in compliance context — NOT mere operational correctness like "how to properly format a table")

*Self-evidence pattern:* The mentee does not use policy terms but proactively lists reasons their content might not meet standards (e.g., lack of sources, questionable notability, fantasy content), then asks whether they can proceed. The act of volunteering potential deficiencies signals awareness that norms may apply.

*Scope/boundary inquiries:* "What type of X can I put/add/share?" / "Which type of links are allowed?" / "Can I use YouTube links as a reference?" — asking what categories of content or sources are permitted (not how to add them).

*Threshold/access requirements:* "How many edits to get privileges?" / "When can I create my own pages?" / "Am I a confirmed user yet?" — asking about conditions needed for permissions.

*Policy concept names — when the mentee asks about or discusses their rules/standards/applicability:*
- Content standards: notability, reliable source, verifiability, NPOV, original research
- Stakeholder-related: conflict of interest, COI, paid editing, disclosure
- Copyright: copyright, fair use, public domain, free license
- Behavioral norms: edit war, three-revert rule, 3RR, vandalism, sock puppet, canvassing
- Process: speedy deletion, AfD, articles for deletion, semi-protected, autoconfirmed

Note: merely *mentioning* a policy concept descriptively (e.g., "I found a reliable source and want to add it") is NOT Q4=Y. The mentee must be *asking about* the concept's rules or applicability.

**Signals that indicate Q4=N:**

*"Can I" disambiguation:*
- "Can I" + policy terms or compliance context → Q4=Y (asking about permission)
- "Can I" + no policy terms, no compliance context → Q4=N (treat as operational)
- Do NOT infer normative intent just because the object is sensitive (politician, company, self). The mentee must express compliance concern in their own words.
- **Exception — scope/boundary structure:** "Can I use X **as a** [source/reference/citation]?" is a scope/boundary inquiry (asking what types of sources are acceptable), not a simple operational question. This triggers Q4=Y via the scope/boundary signal, even without an explicit policy word. The key is the "as a [type]" structure, which asks whether X qualifies as an acceptable instance of that type. Compare: "Can I add a link?" → Q4=N (operational). "Can I use YouTube links as a reference?" → Q4=Y (asking what source types are acceptable).

*"Can I" contrast examples (same object, with vs without policy words):*
- "Can I create a page about my company?" → Q4=N (no policy words)
- "Can I create a page about my company without it being a conflict of interest?" → Q4=Y (policy concept)
- "Can I write about myself?" → Q4=N (no policy words)
- "Is it allowed to write about myself?" → Q4=Y ("allowed")
- "Can I add a photo I found online?" → Q4=N (no policy words)
- "Can I add a photo I found online, or is that copyright infringement?" → Q4=Y (policy concept)
- "Can I create two accounts?" → Q4=N (no policy words)
- "Can I create two accounts, or is that against the rules?" → Q4=Y ("rules")
- "Can I add my own research findings?" → Q4=N (no policy words)
- "Can I add my own research, or does that count as original research?" → Q4=Y (policy concept)

*Sensitive objects without normative language:*
Merely mentioning a politician, company, living person, or self as the subject of editing — without any normative-seeking language or self-evidence pattern — is Q4=N. Examples:
- "How do I create a page about a politician?" → Q4=N
- "I want to write about my company" → Q4=N
- "Can I make my professor's page?" → Q4=N
- "How can I create my own biography?" → Q4=N

*"Best practice" / "recommended" / "should I" do NOT automatically trigger Q4:*
These words often appear when the mentee seeks practical advice, not policy information. The mentor's answer would be experience-based tips, not policy citations.
- "is it recommended to use less citations or more?" → Q4=N (practical advice)
- "how many citations should I add?" → Q4=N (no policy dictates citation counts)
- "what is the best practice for sourcing?" → Q4=N (practical advice)
- Contrast: "is it recommended per Wikipedia policy to use less citations?" → Q4=Y (explicit policy reference)

*Process/policy name + operational question:*
If the mentee mentions a process or policy name but asks about operational steps (how to submit, where to click), not about the standards or criteria, then Q4=N.

*Operational keywords often confused with normative:*
- "what formats are accepted" → technical format specifications, Q4=N
- "how do I create/make/start" → operational steps, Q4=N
- "can I delete/undo/revert" → operational capability, Q4=N
- "how many articles can we write" → factual capacity question, Q4=N
- "what do I have to do before X is possible" → process steps, Q4=N

---

## Q5 Own Work: Has the User Already Done Some Work?

Does the mentee come to this conversation with editing activity that has already produced observable results on Wikipedia? Q5 captures the mentee's current socialization stage — whether they have already engaged in work they can point to, or are starting from scratch.

Decision tree:

```
Node 1: Does any part of the message establish that the user has recently
         performed editing activity on Wikipedia that produced observable 
         results?
         (created, edited, submitted, saved, published a draft/page/article; 
          or experienced content-level consequences that imply prior output: 
          edits reverted, article deleted, draft declined, content removed)
         The editing must be the user's OWN action on Wikipedia — not 
         activity by others, and not the user's real-world work outside 
         Wikipedia (e.g., publishing books, writing articles for journals).
         The activity should be current or recent — not distant past activity 
         mentioned only as background context.
         Note: account-level restrictions (blocked, locked out, cannot edit) 
         do not by themselves establish that the user has produced content — 
         only that their account is restricted.
  ├─ YES → Q5=Y
  └─ NO → Node 2

Node 2: Does the user use possessive/definite reference to Wikipedia content
         that they created or edited?
         (my article, my draft, my page, the article I wrote, this edit)
         Important: "my page" or "our page" can mean either:
         (a) a page the user created/edited on Wikipedia → Q5=Y
         (b) a page ABOUT the user or their organization → Q5=N
         Distinguish by context: if the user is the subject of the page 
         (rather than its editor), possessive reference alone does not 
         establish editing activity.
  ├─ YES (meaning (a): user created/edited it) → Q5=Y
  └─ NO (meaning (b), or no possessive reference) → Node 3

Node 3: Does the message express only future intent or general inquiry?
         (want to, would like to, interested in, how do I start, 
          a/an + content word)
  ├─ YES → Q5=N
  └─ NO → Q5=N (default conservative)
```

**Q3 vs. Q5 distinction:**
- Q3 asks whether the user wants the mentor to **evaluate** their work → about the mentor's requested action
- Q5 asks whether the user **has** work → about the mentee's state
- Q5=Y does not require Q3=Y. A user can mention past work while asking an unrelated question.

**Content-output signal words (→ Q5=Y):** was removed, gets messed up, keep getting error, not displaying, accidentally deleted, reverted, was declined, saved, submitted, I created, I edited, my draft, my article — these imply the user has produced or modified content. Note: these signals must relate to **editing/content activity**, not account operations. "I keep getting an error when adding a reference" → Q5=Y (editing action). "I keep getting an error when trying to create an account" → Q5=N (account issue, no content produced).

**Account-state signal words (→ not sufficient for Q5=Y):** blocked, locked out, cannot edit, restricted, suspended — these describe account status, not content output. The user may have been blocked before producing any work.

**Future-intent and preparation signal words (→ Q5=N):** want to, would like to, interested in, don't know how to, working on, planning to, preparing, researching — these indicate the user is in an intent or preparation stage, not that editing has already occurred on Wikipedia. "I'm working on updating X" does not equal "I updated X": the former may mean the user is researching or drafting locally.

**Distant-past signal words (→ Q5=N):** years ago, a long time ago, back in [distant year], once before. If the user mentions editing activity only from the distant past and their current request is about starting something new, the past activity is background context, not current own work.

**Article/pronoun pattern:**
- my/the/this + content word → typically implies existing work → Q5=Y
- a/an + content word → typically general inquiry, no existing work → Q5=N

**Wording trap:** The first clause may look like future intent, but a later clause may reveal past action. If any part of the message establishes that editing has already occurred, Q5=Y.

**Real-world work vs Wikipedia work:** The user's work outside Wikipedia does not count. Example: "I'm a professor and I've published extensively on this topic, how do I create a Wikipedia page?" — "published extensively" refers to academic publishing, not Wikipedia editing → Q5=N.

---

## Output Format

For each message, return only a JSON object with no explanation:
{"Q0":"Y","Q2":"N","Q3":"N","Q4":"N","Q5":"N"}

If Q0=N, all other dimensions must be N.
