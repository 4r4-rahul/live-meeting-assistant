# Interview Playbook (universal — applies to every interview)

Distilled from Amazon's onsite prep materials, but these frameworks are industry-standard and the coach
applies them in **every** interview (any company). The company-specific raw material lives in
`whisper-backend/company_prep/<company>.md` and is auto-loaded as extra context when you deep-link with
`?company=`.

---

## 1. Behavioral — STAR + ownership + metrics

Every strong behavioral answer is a **STAR** story with real depth:

- **S**ituation — the context, briefly.
- **T**ask — what you specifically needed to achieve.
- **A**ction — what **you** did (the bulk of the answer).
- **R**esult — the measurable outcome + what you learned.

Rules interviewers grade on:
- **"I" not "we"** — say "I decided / I built / I championed", not "we did". They need *your* contribution.
- **Dive deep** — the nitty-gritty: which technology, what tradeoffs, why this over that. No top-level fluff.
- **Metrics** — quantify impact (latency, %, $, scale, time saved). Data-driven beats adjectives.
- **Learnings** — what you'd do differently, from successes *and* failures.
- **Customer-centric** — work backwards from the user/customer where relevant.

Competencies these map to (Amazon's LPs, but every company probes the same traits): Customer Obsession,
Ownership, Bias for Action, Earn Trust, Dive Deep, Deliver Results, Have Backbone (disagree & commit),
Invent & Simplify, Learn & Be Curious, Insist on the Highest Standards, Dealing with Ambiguity.

---

## 2. Coding — the Problem-Solving Methodology (time-boxed)

1. **Clarify** the question — ask clarifying questions, address ambiguity (~2 min). *The #1 thing engineers skip.*
2. **Rephrase** it back — restate in your own words; state the signature, inputs, output (~1 min).
3. **Lay out the algorithm** — the approach, **and state time/space complexity**, before/while coding (~3-4 min).
4. **Write the code** — clean, maintainable, close to production; discrete reusable functions; good names.
5. **Review & test** — trace an example, cover edge cases, fix bugs, discuss optimizations (~5 min).

Bar: pick the right data structure, reach the **optimal** solution, **discuss tradeoffs** (why this over that),
know Big-O, think out loud, take hints, handle edge cases, test your code.

---

## 3. System Design — the framework

1. **Requirements & goals** — clarify scope (what it does AND what it won't), functional + non-functional.
2. **Interface / API** — the core operations the system exposes.
3. **Estimate scale** — users, QPS/TPS, read:write ratio, storage/year, bandwidth. (Drives every later choice.)
4. **High-level design** — major components + data model; draw the boxes and the data flow.
5. **Deep-dive** — pick a component and go deep; defend the choice.
6. **Scale it** — vertical vs horizontal scaling, **caching**, **load balancing**, DB **replication** &
   **partitioning/sharding**, CDN, queues/async, map-reduce.
7. **Bottlenecks & tradeoffs** — traffic (1M→100M users = ? TPS), latency, memory, storage, security;
   **health metrics**: performance, availability, scalability, security. Spend 5-7 min here.

Posture: **be a consultant** (interviewer is your customer), **drive the conversation**, **know *why***
(defending tradeoffs is the whole point), consider cost/performance/scalability.

---

## 4. Study resources (from the Amazon pack — good for any interview)

- *Cracking the Coding Interview* — esp. Big-O, Behavioral, Technical Qs, Data Structures/Algorithms.
- Grokking the Coding Interview & Grokking the System Design (educative.io) — pattern-by-pattern.
- Top-75 / Top Amazon LeetCode lists; solve without an IDE or Google.
- System design: highscalability.com case studies; common designs — URL shortener, parking lot, elevator,
  rate limiter, news feed, chat, web crawler.
