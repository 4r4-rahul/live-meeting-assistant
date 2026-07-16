# Robinhood — SRE / Software Engineer

Loaded as grounding when interviewing at Robinhood. Layer on top of the universal STAR / coding /
system-design frameworks. (Confirm the exact req/role on the call — SRE vs backend SWE.)

## Why Robinhood / Why SRE
Why Robinhood: **reliability is the product** — real-time trading, market hours, uptime = money; scale and
infra problems that matter directly to users. Why SRE: his JPMC work is already reliability- and infra-heavy;
he wants to own reliability end-to-end — SLOs, on-call, incident response, automation — not just feature work.

## 60-second intro
Senior Software Engineer, Data Platforms at JPMorgan, ~6 yrs, building reliable secure data infra on AWS (ECS
Fargate, EMR, Snowflake, Airflow). Day-to-day is SRE-shaped: multi-GB pipelines inside SLA, zero-downtime auth
and secrets, making distributed systems fail safely. Drawn to Robinhood because reliability there is the
product and he wants to go deeper on the reliability/infra side.

## SRE stories (STAR — problem → what I did → impact)
- **SLA under a security constraint (best SRE story):** multi-GB cross-region Snowflake streaming forced
  through a corporate forward proxy that would saturate and blow SLA → designed a **regional split-tunnel**
  (scoped NO_PROXY so same-region AWS traffic bypasses the proxy, cross-region still routes through) and fixed
  proxy state leaking across concurrent requests. Stayed in SLA, passed security review.
- **Zero-secrets auth:** built AWS **SigV4** signing from scratch so an ECS task authenticates to Snowflake
  with NO long-lived secrets — only its IAM role via STS. Removed a class of rotation/leak risk; signer reused.
- **Backup (reliability/concurrency):** two processes sharing a state store were silently losing each other's
  writes and could corrupt the DB → built a safe store (cross-process lock, atomic write-and-rename,
  union-merge, corruption guard) locked in with tests. No more lost updates.

## Coding screen — one MEDIUM LeetCode, 60 min, live with an engineer
Use the **UMPIRE** framework out loud every time: **U**nderstand (restate, ask input size / edge cases / dups
/ sorted?) → **M**atch ("this looks like sliding window / hashmap / BFS because…") → **P**lan (approach in
plain English, get a nod; brute force first if stuck) → **I**mplement (code cleanly, narrate) → **R**eview
(trace an example + an edge case: empty, 1 element, dups) → **E**valuate (state time/space). **Never code in
silence.** Python is his language. Prep via NeetCode 150 by pattern (~8 patterns cover ~90% of mediums).

## Logistics answers (recruiters always ask)
- **Why leaving JPMC:** wants to go deeper on reliability/infra at a product company where scale and uptime
  are core, and move faster than a large bank allows. (Positive — never bash the employer.)
- **Work authorization:** "I'm authorized to work in the US on an H-1B, and I'd need the company to support
  visa sponsorship / transfer." Robinhood sponsors — state it matter-of-factly; don't dodge, don't volunteer
  unprompted. (See [[user_work_authorization]].)
- **Comp:** do NOT anchor low. "I'd love to understand the budgeted range — senior SRE roles are typically
  well into $200K+ total comp, and I'm looking for something competitive, but most focused on fit. What range
  are you working with?" Make them share the band first.
- **Location:** Plano, TX (Central); open to remote or relocation depending on the role.
- **Notice:** standard two weeks, flexible start.

## Questions to ask (pick 3-4)
Which SRE team/req is this and what systems would I own? · What's the rest of the process + timeline? · What's
the on-call model and team size? · How does the team measure and defend reliability (SLOs, error budgets)?
