# Discovery Engine — build spec

Multi-intent e-commerce recommender. Hackathon MVP. Deadline-driven: prefer working over complete.

## Rules

- Write files. Do not explain code unless asked.
- No new dependencies without asking. Approved list below.
- Every module gets a test in `tests/`. Run `pytest -q` before saying done.
- Type hints on all public functions. No docstrings longer than one line.
- Never read `data/raw/*` or any parquet/csv. Schemas are in this file.
- Never print file contents back to me. Print test results only.
- Commit after each green test run: `git add -A && git commit -m "<phase>: <what>"`
- If a step fails twice, stop and report. Do not retry a third time.

## Stack (pinned, do not change)

python 3.11, polars, numpy, torch (cpu), faiss-cpu, sentence-transformers,
lightgbm, fastapi, uvicorn, pydantic v2, redis, streamlit, pytest, hypothesis,
locust, python-dotenv, anthropic

## Layout

```
src/gateway/      HTTP surface, request validation
src/session/      event ingest, session state, intent inference
src/ranking/      feature assembly, multi-task ranker
src/guardrails/   pure functions, no I/O, no imports from other features
src/compliance/   consent, erasure, audit
src/explain/      reason codes
src/shared/       config, redis/pg clients, telemetry
retrieval_service/  separate FastAPI app, owns Faiss index
trainer/          offline scripts
tests/  bench/  ui/  docs/
```

Dependency rule: a feature imports `shared/` and other features' `interface.py` only. Never internals.

## Data schemas (do not read the files)

`data/raw/articles.csv` — H&M. Columns used:
`article_id` (int), `prod_name` (str), `product_type_name` (str),
`product_group_name` (str), `colour_group_name` (str),
`index_group_name` (str), `department_name` (str), `detail_desc` (str)

`data/raw/customers.csv` — `customer_id` (str), `age` (float, nullable),
`postal_code` (str), `club_member_status` (str)

`data/raw/transactions_train.csv` — `t_dat` (date str), `customer_id` (str),
`article_id` (int), `price` (float), `sales_channel_id` (int)

Sampled outputs (`data/processed/`): `train.parquet`, `val.parquet`,
`items.parquet`, `users.parquet` — all polars.

## Non-negotiables

- **Latency budget**: p99 < 80ms end to end. Retrieval < 12ms, ranking < 25ms.
- **Diversity cap**: no `category_l1` exceeds 35% of any returned list.
- **Guardrails fail closed**: if `guardrails/` raises, return 503. Never serve unfiltered.
- **Time-based split only**: train on earlier weeks, validate on the last week. Random split = leakage = wrong.
- **Erasure removes the vector**, not just the DB row.

## Scale targets

MVP: 20k users, 15k items, ~2M transactions. Index fits in RAM (~20MB).
Design for 10k req/s but load test at whatever the laptop gives.

## Model config

- Embedding dim: 128
- Intent heads: K=4, with orthogonality penalty. Log pairwise cosine each epoch.
  If any pair > 0.8, heads are collapsing — report it.
- Session length: last 20 events
- Retrieval: K parallel ANN queries, k=150 each, merge+dedupe to ~500
- Ranker: LightGBM, 4 objectives (click, cart, purchase, wishlist)
- Content embeddings: `sentence-transformers/clip-ViT-B-32`, text encoder only

## Definition of done per phase

A phase is done when `pytest -q` is green and the commit is made. Not before.
