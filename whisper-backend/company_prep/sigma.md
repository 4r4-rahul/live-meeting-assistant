# Sigma Computing — Forward Deployed Engineer (FDE)

Loaded as grounding when interviewing at Sigma. Layer this on top of the universal STAR / coding /
system-design frameworks.

## The role (frame EVERY answer around this)
FDE = customer-facing engineer, **~60% engineering / 40% consulting** (solutions architect + data
engineer + trusted advisor). Embed with customers post-sale: discovery → model their warehouse data →
build in Sigma → drive adoption → expand. Own the outcome end-to-end like a mini-founder. They hire for:
technical depth (SQL, data modeling, warehouses, perf), customer empathy, ambiguity tolerance, ownership.

## Product knowledge (say this confidently — it's the #1 FDE signal)
Sigma is **cloud-native BI with a spreadsheet UX** that runs **directly on the data warehouse** (Snowflake,
BigQuery, Databricks, Redshift). Key innovation: **formula compilation** — every spreadsheet op compiles to
optimized warehouse-dialect SQL and executes **in the warehouse**. No extracts, no data copies, live data,
governance stays at the source. Optimizer picks cache / in-browser / pushdown and merges queries. Governance
in two places: warehouse (OAuth run-as-user + roles) and Sigma (row/column-level security). 2026 positioning:
"AI runtime / AI apps + analytics on the warehouse." Edge vs Tableau/Looker: no data movement = security holds
at source; warehouse compute scales to billions of rows; spreadsheet UX removes the SQL bottleneck.
**Before the technical round: sign up for the free trial, click around (workbooks, input tables, a calc,
a chart/control), and read sigmacomputing.com/product/architecture.**

## Why the candidate fits (memorize this mapping)
Sigma needs warehouse-native analytics (Snowflake) → he runs Snowflake in prod at JPMC (built a zero-secrets
extract service). Optimized SQL/pushdown → daily SQL, multi-GB pipelines in SLA, reasons about query cost.
Data modeling/ELT → Spark/EMR + Airflow, Databricks, Redshift. Cloud + security/governance → AWS (IAM, ECS,
KMS), SigV4, RLS, financial-services rigor. Owns ambiguous end-to-end work → stands up whole services solo.
**Honest gap:** no formal "deployed-at-customer" consulting seat — frame it as: "I've done the hard technical
half my career and embedded with business stakeholders to turn fuzzy asks into shipped data systems; FDE is
where I want those two halves to be the JOB." Show hunger to be customer-facing.

## Why Sigma / Why FDE
Why Sigma: the architecture is genuinely right (analytics on the warehouse, no copies, governance at source —
he felt the pain of extract-and-copy BI at a bank); sits on the stack he knows best; spreadsheet UX is a real
wedge. Why FDE (not pure backend SWE): at his best owning the whole thing and talking to real users; wants
depth pointed at customer outcomes and fast feedback across many problems, not one service for two years.

## Stories (STAR — reuse these)
- **Technical depth / warehouse:** Snowflake zero-secrets extract service on ECS Fargate — hand-rolled AWS
  SigV4 in Python, exchanged a signed STS call for a short-lived warehouse token; split worker from API.
  Result: zero stored secrets, leak/rotation risk gone, signer reused by other teams.
- **Perf under constraint:** multi-GB cross-region Snowflake streaming through a saturating corporate proxy →
  regional split-tunnel (scoped NO_PROXY) + fixed proxy state leaking across concurrent requests. Stayed in SLA.
- **Ownership end-to-end:** stood up a registry service as a full unit (DynamoDB + KMS + IAM + deploy) and
  drove a zero-downtime blue/green ECS migration across four environments.
- **Stakeholder / customer-facing (his weak spot — PREP HARDEST):** a real moment turning a vague business ask
  into a shipped data solution / unblocking a non-technical team (discovery → model → ship → adoption up).

## Technical focus — FDE screens are SQL + data-modeling heavy, NOT LeetCode graph puzzles
- **SQL (most likely):** joins (inner/left/anti), GROUP BY/HAVING, **window functions** (ROW_NUMBER, RANK,
  LAG/LEAD, running totals), CTEs, dedup ("latest row per key" = ROW_NUMBER() OVER (PARTITION BY id ORDER BY
  ts DESC)=1), date bucketing, cohort/retention, pivots. **Talk about query cost** (what scans/shuffles, why
  pushdown beats pulling data out) — very on-brand.
- **Data modeling:** star schema, fact vs dimension, grain, denorm for analytics, SCD, wide BI tables.
- **Warehouse perf (his strength):** Snowflake micro-partitions, clustering, result cache, warehouse sizing,
  pruning; BigQuery slots/partitioning. Cost/perf tradeoffs.
- **Light coding:** Python; arrays/hashing, string parsing; narrate reasoning, never code in silence.

## Behavioral signals they screen for
Non-technical stakeholder collaboration; turning ambiguity into shipped work; customer empathy; ownership.
