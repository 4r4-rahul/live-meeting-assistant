# Amazon / AWS — interview-specific prep (from the official onsite pack)

Loaded as extra grounding when the session is for Amazon/AWS. The universal frameworks
(STAR, coding methodology, system design) are in the coach's core instructions; this adds Amazon's
specifics on top.

## Loop structure
4 technical rounds: **3 coding + 1 system design**, and **every** round also asks **Leadership Principle
(behavioral)** questions. Coding is on a **shared live code link — no IDE/compiler to test**, so trace by
hand. Any OO language you're most comfortable with. Goal: an **optimal solution in ~30 min**.

## Leadership Principles (behavioral) — Amazon grades these hard
Use **STAR**, **"I" not "we"**, deep specifics, **measurable metrics**, tradeoffs, and learnings. Map each
story to the principle being probed:

- **Customer Obsession** — start from the customer, work backwards; went above & beyond for a customer.
- **Ownership** — long-term thinking; took on something outside your area; "that's not my job" is a red flag.
- **Bias for Action** — speed; calculated risk; acted decisively under a time-critical situation.
- **Earn Trust** — vocally self-critical; a mistake you made; hard feedback; a conflict you handled.
- **Learn & Be Curious** — how you ramp on something new/unknown.
- **Deliver Results** — removed roadblocks, hit the deadline, the tradeoffs/sacrifices, the impact.
- **Dive Deep** — stay in the details; skeptical when metrics and anecdote differ; no task beneath you.
- **Have Backbone, Disagree & Commit** — disagreed with a manager/peer; unpopular stance you held.
- **Invent & Simplify** — a complex problem solved with a simple solution; a process you improved.
- **Insist on the Highest Standards** — refused to settle for "good enough"; raised the quality bar.

Behavioral tips: strong example with real complexity/scope/impact; data-driven; talk about learnings from
BOTH success and failure; be customer-centric; prepare several distinct examples (they ask "something you
haven't mentioned yet").

## Coding bar
Right data structure → optimal solution → **discuss tradeoffs** (why one over another) → Big-O of each →
clean, maintainable, testable code, OO design basics, edge cases. ASK CLARIFYING QUESTIONS FIRST (the #1
mistake is not doing this). Think out loud; take hints; test your code by hand (no compiler).

## System design bar
Requirements/scope → components → **anticipate scaling early** → defend tradeoffs. Spend 5-7 min on **health
metrics** (performance, availability, scalability, security). Bottlenecks to address: traffic/TPS
(1M→100M users), latency (caching, pre-fetch), memory (vertical scaling), storage (objects/year), databases
(horizontal scaling/sharding). **Consider Amazon's scale.** Be a consultant; drive the conversation; know *why*.

## Sample behavioral questions they pull from
Most interesting project not on your resume · greatest success not on your resume · a hard/challenging
decision · worked backwards from a customer problem · analyzed facts quickly & responded immediately · used
feedback to drive improvement · sacrificed short-term gain for a long-term goal · took an unpopular stance ·
solved a complex problem with a simple solution · not enough data to decide (ambiguity) · helped a teammate
grow · raised the quality bar · communicated an unpopular change in direction.
