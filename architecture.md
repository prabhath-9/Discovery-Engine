# Discovery Engine — System Design

Multi-intent personalized recommendation and discovery for e-commerce.

**Status:** design frozen for the 12-hour build. Sections marked `[POST-MVP]` are documented but not implemented in the hackathon submission — they are deliberate scope cuts, not oversights.

---

## 1. Constraints

Design without constraints is decoration. These are the numbers everything else follows from.

### Stated (from the problem statement)

| Constraint | Target |
|---|---|
| End-to-end latency | p99 < 80ms (retrieval + rerank) |
| Peak throughput | 10,000 req/s |
| Diversity | No category > 35% of any list |
| Cost | ANN + light reranker, no brute-force LLM in request path |
| Compliance | DPDP Act 2023, deterministic explainable guardrails |

### Assumed (state these openly; correct them if wrong)

| Assumption | Value | Why it matters |
|---|---|---|
| Team size | 2–4 students | Rules out an operationally heavy topology |
| Build time | 12 hours, single session | Rules out anything needing >30 min of setup |
| Demo hardware | One laptop, 16GB RAM, no GPU | Forces dataset sampling and CPU-only inference |
| Catalog size (MVP) | ~15,000 articles | Index fits in memory trivially |
| Users (MVP) | ~20,000 sampled customers | Training completes in minutes |
| Load test ceiling | ~500 req/s locally | 10k req/s is proven by capacity math, not by running it |

### Non-functionals, ranked

Ranking these is the single most useful thing in this document, because it decides every tie-break below. The order comes from the scoring rubric, not from engineering taste.

1. **Guardrails and compliance** (25% of score) — correctness here is non-negotiable and deterministic
2. **Latency** (p99 < 80ms) — a hard gate; a correct answer at 300ms fails
3. **Cost per inference** (15%) — must be *measured and reported*, not estimated
4. **Recommendation quality** — real, but the rubric rewards a measured baseline-to-model delta more than an absolute number
5. **Throughput** — proven analytically, not empirically

---

## 2. Functional requirements

### In scope

| ID | Requirement | Surface |
|---|---|---|
| F1 | Personalized home feed reflecting multiple concurrent session intents | `POST /v1/feed` |
| F2 | Complementary-item bundles ("Complete the Look") | `POST /v1/bundles` |
| F3 | Natural-language search with semantic re-ranking | `POST /v1/search` |
| F4 | Useful results for a brand-new user within 3 interactions | all of the above |
| F5 | Useful placement for a brand-new item with zero interactions | all of the above |
| F6 | Every recommendation carries a machine-readable reason code | all of the above |
| F7 | Full erasure of a user's data and their presence in the model index | `DELETE /v1/users/{id}` |
| F8 | Clickstream ingestion updating session state in real time | `POST /v1/events` |

### Explicitly out of scope

Graph neural networks over the co-purchase graph; real image ingestion at catalog scale; A/B testing infrastructure; multi-region deployment; a production Kafka cluster. Each is named in the README with the reason.

---

## 3. Service boundaries

### Decision: two online deployables plus an offline batch plane

**API service** — request handling, session assembly, feature fetch, ranking, guardrail enforcement, explanation, consent and erasure.

**Retrieval service** — owns the Faiss index. Exposes `search(vectors, k, filters) -> candidate_ids`. Nothing else.

**Trainer** — a scheduled job, not a service. Produces versioned artifacts.

### Why this split and not more

A service boundary is justified by *divergent scaling or runtime characteristics*, or by *organizational* independence. Neither traffic volume nor architectural fashion justifies one.

The retrieval service qualifies on the first count: it pins the entire index in RAM and scales on memory, while the API service is CPU-bound on ranking and scales on cores. Splitting them lets each scale on its own axis and lets the index be reloaded and hot-swapped without dropping in-flight API requests.

Nothing else qualifies. Session, ranking, guardrails, and explanation all execute inside a single 80ms request, share the same feature payload, and would each add a network hop.

### Why not six services

Concrete cost, in the currency that matters here:

```
6-service mesh:  5 hops × (2ms network + 1.5ms serialize) = ~17ms
2-service split: 1 hop  × (2ms network + 1.5ms serialize) = ~3.5ms
```

That is 21% of the entire latency budget spent on plumbing, for a team that cannot operate six deployables in 12 hours. **Rejected.**

### How "microservices" is still satisfied

Inside the API service, code is organized by **domain feature, not technical layer**:

```
src/
  gateway/        # HTTP surface, request validation, response shaping
  session/        # event ingestion, session state, intent inference
  ranking/        # multi-task ranker, feature assembly
  guardrails/     # diversity, policy, safety — pure functions, no I/O
  compliance/     # consent, erasure, audit log
  explain/        # reason codes, LLM blurb cache
  shared/         # db clients, config, telemetry
retrieval_service/  # separate deployable
trainer/            # offline jobs
```

**Dependency rule:** a feature may import `shared/` and other features' public `interface.py` only. Never another feature's internals. Enforced by a lint rule (an import-linter contract) that runs in CI.

This is a modular monolith with service-shaped seams. Each module can be lifted into its own deployable by replacing its `interface.py` with an HTTP client — no other file changes. That is the property "microservices-ready" should mean, and it is worth more to a judge than six half-built services.

---

## 4. Data flow

### Read path (the 80ms path)

```
Client
  → gateway: validate, resolve consent, load session from Redis        [5ms]
  → session: encode last N events → K intent vectors                   [8ms]
  → retrieval service: K parallel ANN queries, merge + dedupe          [12ms]
  → ranking: fetch features, score ~500 candidates, keep top 50        [25ms]
  → guardrails: policy filter, 35% diversity cap, MMR de-dup           [5ms]
  → explain: attach cached reason codes                                [2ms]
  → gateway: serialize, emit audit record async                        [3ms]
                                                            total: ~60ms
                                                     headroom: 20ms
```

Headroom is deliberate. p99 is not p50 — garbage collection, cache misses, and tail network latency consume it. A design with zero headroom fails its p99 target.

### Write path (off the critical path)

```
Client → POST /v1/events → append to Redis session list (LPUSH + LTRIM to N)
                         → fire-and-forget to durable event log
                         → return 202 immediately
```

The event endpoint must never block a recommendation request. It returns before persistence completes.

### Offline path

```
Nightly:  raw events → feature build → train towers → train ranker
                    → encode all items → build Faiss index
                    → LLM enrichment of new items (tags, blurbs)
                    → write artifacts as version v{N}
                    → retrieval service loads v{N}, health-checks, swaps
```

Artifacts are versioned and immutable. Rollback is pointing at `v{N-1}`.

---

## 5. Data model

### Postgres (source of truth)

```sql
products (
  product_id      text primary key,
  title           text not null,
  brand           text,
  category_l1     text not null,      -- diversity cap operates on this
  category_l2     text,
  price_cents     int  not null,
  in_stock        bool not null default true,
  age_restricted  bool not null default false,
  created_at      timestamptz not null default now()
);
create index on products (category_l1) where in_stock;

complements (                          -- "Complete the Look" edges
  product_id      text references products,
  complement_id   text references products,
  weight          real not null,       -- co-purchase lift, not raw count
  primary key (product_id, complement_id)
);

users (
  user_id         text primary key,
  age_band        text,                -- banded, never a raw birthdate
  region          text,
  created_at      timestamptz not null default now()
);

consent (                              -- DPDP: consent is a first-class record
  user_id         text references users on delete cascade,
  purpose         text not null,       -- 'personalization' | 'analytics'
  granted         bool not null,
  granted_at      timestamptz not null,
  primary key (user_id, purpose)
);

recommendation_log (                   -- audit trail; the explainability evidence
  request_id      uuid primary key,
  user_id         text,                -- nullable: null after erasure
  surface         text not null,
  model_version   text not null,
  returned_ids    text[] not null,
  reason_codes    jsonb not null,
  guardrails_applied jsonb not null,   -- which caps fired, what was dropped
  latency_ms      int  not null,
  created_at      timestamptz not null default now()
);
```

Note `guardrails_applied`. Logging *that a guardrail fired and what it removed* is the difference between claiming explainability and demonstrating it. This column is what you put on screen during the demo.

### Redis (hot state)

| Key | Type | TTL | Contents |
|---|---|---|---|
| `sess:{user_id}` | list | 30 min | Last 20 event tuples, newest first |
| `feat:u:{user_id}` | hash | 6 h | Precomputed user features |
| `feat:i:{product_id}` | hash | 24 h | Item features: CTR, price band, popularity |
| `blurb:{product_id}` | string | 7 d | LLM-generated description |
| `qcache:{sha1(query)}` | string | 1 h | Parsed search intent |
| `pop:{region}:{age_band}` | zset | 1 h | Cold-start fallback ranking |

**Why Redis and not just Postgres:** the session list is read on every single request and written on every event — a read-modify-write at 10k req/s. Postgres can do it; Redis does it in 0.3ms instead of 3ms, and 2.7ms is 3.4% of the budget. This is a measured need, which is the only acceptable reason to add a second datastore.

### In-memory (retrieval service)

Faiss `IndexHNSWFlat`, 128 dims, cosine. Memory: `n_items × 128 × 4 bytes × ~2.5` for HNSW graph overhead.

- 15k items → ~19MB
- 105k items (full H&M) → ~135MB
- 10M items `[POST-MVP]` → ~13GB, at which point the index shards by category

---

## 6. API contracts

REST over HTTP/JSON. gRPC would save ~1ms of serialization and cost an hour of setup — not worth it tonight, and named as a `[POST-MVP]` upgrade.

### `POST /v1/feed`

```jsonc
// request
{
  "user_id": "u_8812",           // null for anonymous
  "session_id": "s_44f1",        // required
  "limit": 20,
  "context": { "device": "mobile", "region": "IN-AP" }
}

// 200 response
{
  "request_id": "3f9c...",
  "model_version": "v20260807-2",
  "detected_intents": [           // the multi-intent story, made visible
    { "label": "seasonal_browsing", "confidence": 0.61, "slots": 9 },
    { "label": "urgent_replacement", "confidence": 0.28, "slots": 7 },
    { "label": "bargain_hunting",    "confidence": 0.11, "slots": 4 }
  ],
  "items": [
    {
      "product_id": "0706016001",
      "score": 0.83,
      "intent": "seasonal_browsing",
      "reason": { "code": "SESSION_AFFINITY", "text": "Because you viewed 3 linen shirts" }
    }
  ],
  "guardrails": { "diversity_cap_applied": true, "items_dropped": 4 },
  "latency_ms": 58
}
```

Returning `detected_intents` and `guardrails` in the response body is a deliberate choice. It costs nothing, and it means the demo UI can render your differentiator without a second endpoint.

### Remaining endpoints

| Method | Path | Purpose | Notes |
|---|---|---|---|
| `POST` | `/v1/bundles` | Complete the Look for a seed product | Complement graph + embedding blend |
| `POST` | `/v1/search` | NL query → re-ranked results | Query cache in front |
| `POST` | `/v1/events` | Clickstream ingest | Returns `202`, never blocks |
| `GET` | `/v1/users/{id}/data` | DPDP access right | Everything held about the user |
| `DELETE` | `/v1/users/{id}` | DPDP erasure | See §8 |
| `POST` | `/v1/consent` | Grant/revoke by purpose | Revocation takes effect on next request |
| `GET` | `/healthz` `/readyz` | Liveness, readiness | Readiness gates on index loaded |
| `GET` | `/metrics` | Prometheus | Latency histogram, cache hit rate, cost counters |

### Error shape (one shape everywhere)

```json
{ "error": { "code": "INVALID_SESSION", "message": "session_id is required",
             "request_id": "3f9c...", "retryable": false } }
```

### Degradation contract

The feed endpoint **never** returns 5xx for a model failure. Each stage has a documented fallback:

| Failure | Fallback | Response signal |
|---|---|---|
| Retrieval service down | Popularity by `region:age_band` from Redis | `degraded: "retrieval"` |
| Ranker fails | Return retrieval order unranked | `degraded: "ranking"` |
| Redis down | Cold-start path, no session personalization | `degraded: "session"` |
| Postgres down | Serve from Redis item features only | `degraded: "catalog"` |

**Guardrails have no fallback.** If the guardrail module raises, the request fails closed with a 503. Serving an unfiltered list is worse than serving nothing — and this is the 25% dimension. Make sure a judge hears that sentence.

---

## 7. Caching strategy

| Layer | What | Invalidation | Expected hit rate |
|---|---|---|---|
| Query intent cache | Parsed NL search intent | TTL 1h | 60–75% (head queries dominate) |
| Item feature cache | CTR, popularity, price band | TTL 24h + write-through on catalog change | >95% |
| LLM blurb cache | Generated descriptions | Only on catalog update | >99% |
| Session | Last 20 events | TTL 30 min, LRU | n/a |

**No caching of the final feed.** It is personalized and session-dependent; a cached feed is a stale feed, and staleness is precisely the failure mode this whole project exists to fix.

The LLM blurb cache is what makes the cost target achievable. Every product description is generated exactly once, offline. At steady state, LLM cost per recommendation request approaches zero.

---

## 8. Guardrails and compliance

The heaviest-weighted dimension. Design principle: **guardrails are deterministic pure functions with unit tests, never model outputs.** A model that usually respects a 35% cap is a model that violates it in the demo.

### Enforcement pipeline (ordered, in `guardrails/`)

Ordering is load-bearing — safety filters run before diversity, so the cap is computed over an already-legal set.

1. `filter_availability` — drop out-of-stock, region-unavailable
2. `filter_policy` — drop age-restricted items for users without a verified age band
3. `filter_sensitive` — hard blocklist on inferring or acting on health, religion, or caste signals
4. `filter_seen` — drop items impressed to this session in the last N requests
5. `cap_category` — greedy pass enforcing ≤35% per `category_l1`
6. `mmr_diversify` — Maximal Marginal Relevance, λ=0.7, over item embeddings
7. `attach_reasons` — every surviving item gets a reason code

Each function: pure, typed, independently unit-tested. `cap_category` gets a property-based test asserting the invariant holds for any input list — that test is worth showing on screen.

### DPDP Act 2023 mapping

| Obligation | Implementation |
|---|---|
| Consent before processing | `consent` table checked in `gateway` before personalization; no consent → cold-start path |
| Purpose limitation | Consent is per-purpose; analytics consent does not authorize personalization |
| Data minimization | Age stored as band, not birthdate; no raw location, only region code |
| Right to access | `GET /v1/users/{id}/data` |
| Right to erasure | `DELETE /v1/users/{id}` — see below |
| Children's data | No behavioural profiling when `age_band = 'under_18'`; popularity feed only |
| Breach-ready audit | `recommendation_log` retains what was shown, why, and which guardrails fired |

### Erasure, done properly

The step teams miss is the model. Deleting the row while the user's embedding remains in the index is not erasure.

```
DELETE /v1/users/{id}:
  1. cascade-delete users, consent, session state, feature cache
  2. null the user_id in recommendation_log, retain the rest for audit
  3. append user_id to the exclusion tombstone set
  4. remove the user vector from the live index
  5. schedule exclusion from the next training run
  6. return 204 with a deletion receipt id
```

Run this live in the demo, then re-request the feed and show the cold-start path taking over. It takes 20 seconds and is the most convincing thing in the presentation.

### Prompt injection defense

Product titles and descriptions are seller-supplied and therefore untrusted. Before any text reaches the LLM:

- Strip control characters, markdown fences, and instruction-shaped phrases
- Pass content inside delimited blocks, never concatenated into the instruction
- Constrain the output with a JSON schema and reject anything that fails validation
- The LLM output is *advisory only* — it may reorder within the top 50, never inject items, never override a guardrail

That last point is architectural: the LLM sits *before* guardrails in the code path, so its output is still subject to every filter.

---

## 9. Cold start

The rubric asks for useful results within 3 clicks. That is an explicit design target, not an emergent property.

| Situation | Strategy | Signal source |
|---|---|---|
| New user, click 0 | Popularity by `region × age_band` | Redis zset |
| New user, clicks 1–2 | Content similarity to viewed items | CLIP item embeddings |
| New user, click 3+ | Full session encoder | Live session |
| New item, 0 interactions | Content embedding places it in the index immediately | CLIP on title + attributes |
| New item, some interactions | Blend content and collaborative, weight shifting with interaction count | `α = n / (n + 10)` |

The two-tower design is what makes new-item cold start nearly free: the item tower consumes content features, so an item with zero interactions still produces a valid vector the moment it is created. Say this explicitly — it is the architectural payoff, and it is easy to miss if you only look at the diagram.

---

## 10. Scale

### Capacity math for 10,000 req/s

```
Mean service time (not p99):        ~35ms
Concurrent requests in flight:      10,000 × 0.035 = 350
Async workers per API pod:          16
Throughput per pod:                 16 / 0.035 ≈ 450 req/s
Pods needed:                        10,000 / 450 ≈ 23
With 40% headroom for tail + deploy: ~32 pods
```

Retrieval service: ANN query is ~2ms of CPU, index is read-only and identical on every replica. `10,000 × 4 queries × 0.002 = 80` cores, so ~20 pods at 4 vCPU. Read-only replicas scale linearly and need no coordination — this is why the split earns its keep.

**Postgres is the bottleneck that isn't:** the read path touches Postgres only on cache miss (<5%). At 10k req/s that is ~500 queries/s against indexed primary keys, comfortably one instance with a read replica.

### Scaling sequence, in order

Do these in order, only when measurement demands the next one:

1. Vertical: bigger pods (holds to ~2k req/s)
2. Horizontal: more API replicas behind a load balancer (to ~10k req/s) ← **target lives here**
3. Shard the Faiss index by `category_l1`, scatter-gather `[POST-MVP]`
4. Read replicas for Postgres `[POST-MVP]`
5. Regional deployment with local indexes `[POST-MVP]`

### Cost per inference

Report this with real numbers — most teams skip it, and it is 15% of the score.

```
Online serving at 10k req/s:
  32 API pods  × 4 vCPU × $0.04/vCPU-hr = $5.12/hr
  20 ANN pods  × 4 vCPU × $0.04/vCPU-hr = $3.20/hr
  Redis + Postgres                       ≈ $1.20/hr
  Total                                  ≈ $9.52/hr

Requests per hour: 10,000 × 3,600 = 36,000,000
Cost per 1,000 recommendations: $9.52 / 36,000 ≈ $0.00026

Offline (amortized):
  Nightly training + full catalog encode ≈ 2 GPU-hours ≈ $2.00/day
  Amortized over 864M daily requests     ≈ $0.0000023 per 1,000
```

The headline: **~$0.26 per million recommendations.** Compare it in the pitch to the naive alternative — one LLM call per request at ~$0.0005 each is $500 per million, roughly **1,900× more expensive**. That single comparison is the strongest cost-efficiency slide you can put up, and it is exactly what the brief is asking you to demonstrate.

### Model routing

| Case | Model | Share of traffic |
|---|---|---|
| Feed, bundles, cached search | No LLM — embeddings and ranker only | ~95% |
| Novel NL search query (cache miss) | Small model, structured extraction | ~4.5% |
| Ambiguous or multi-constraint query | Large model | ~0.5% |

Log which route each request took and report the distribution. Routing you can't measure is routing you can't defend.

---

## 11. Trade-offs

Every one of these is a decision, with the condition that would flip it.

| Decision | Chosen | Alternative | Would flip if |
|---|---|---|---|
| Topology | 2 services + batch | 6-service mesh | Multiple teams shipping independently |
| Ranker | LightGBM multi-task | Deep neural ranker | >10M training rows, GPU available |
| Vector store | Faiss in-process | Milvus / Qdrant | Catalog >5M items or need live upserts |
| Index type | HNSW | IVF-PQ | Memory-constrained; HNSW is faster, IVF-PQ is smaller |
| Item content | CLIP on text | CLIP on images | More than 2 spare hours; ~15% recall gain expected |
| Event transport | Redis list | Kafka | Need replay, multiple consumers, durability guarantees |
| Session model | Transformer (SASRec) | GRU4Rec | Sequences >50 events; transformer wins on longer ones |
| Serialization | JSON | Protobuf / gRPC | Latency budget tightens below 50ms |
| Intent count K | 4 fixed | Dynamic routing | More training data; fixed K is stabler at hackathon scale |

### The two riskiest calls

**Multi-intent with fixed K=4 heads may produce degenerate intents** — all four heads collapsing to the same vector. This is a known failure mode of multi-interest models on small data. Mitigation: an orthogonality penalty in the loss, plus a cosine-similarity check between heads logged every epoch. If they collapse and you are short on time, fall back to K=2. Detecting this early is worth more than the extra heads.

**Sampled data may not reproduce published quality numbers.** Do not compare your recall@20 to a paper's. Compare it to *your own* co-visitation baseline on the *same* sample. A measured relative improvement is a defensible claim; an absolute number against a different dataset is not.

---

## 12. Failure modes

Ranked by blast radius × likelihood, which is the only ranking worth having.

| Failure | Blast radius | Likelihood | Mitigation |
|---|---|---|---|
| Guardrail module raises | Total — could serve non-compliant lists | Low | Fail closed with 503; property-based tests in CI |
| Index and ranker version skew | High — silently degraded quality | Medium | Version stamp on artifacts; readiness check refuses mismatched pairs |
| Redis eviction under memory pressure | Medium — personalization silently lost | Medium | `maxmemory-policy allkeys-lru`, alert on hit rate < 85% |
| Intent head collapse | Medium — differentiator disappears | Medium | Orthogonality penalty, per-epoch similarity logging |
| Cold-start path never exercised | Medium — breaks live in demo | **High** | Test it explicitly; it is a listed success metric |
| Faiss index fails to load on deploy | High — retrieval down | Low | Readiness gate; previous version stays live until new one health-checks |
| LLM latency spike | Low — search only | Medium | 200ms timeout, fall back to lexical parse |

That "cold-start path never exercised" row is the one that actually bites teams. It is the highest-likelihood failure in the table and it fails in front of the judges. Write the test tonight.

---

## 13. What I would revisit as this grows

- **At ~1M items**: shard the Faiss index by category and scatter-gather. HNSW build time becomes the constraint before query time does.
- **At multiple teams**: extract `ranking/` and `guardrails/` into services. The module interfaces are already the seams.
- **At real production traffic**: replace the Redis event list with a proper log (Kafka/Redpanda) the moment you need replay for training — training on lossy session data quietly caps your ceiling.
- **When quality plateaus**: the co-purchase graph is the biggest untapped signal. A GNN over the Amazon-style "also bought" graph is the natural next model, and the two-tower item encoder is already the right place to plug its output in.
- **First thing to delete**: if the multi-intent heads do not measurably beat single-intent on the held-out week, cut them and say so. A negative result honestly reported reads as engineering maturity; a differentiator that does not work reads as noise.

---

## 14. Repository layout

```
discovery-engine/
├── README.md                  # pitch, setup, results, scope cuts
├── CLAUDE.md                  # build spec for Claude Code
├── docs/
│   ├── architecture.md        # this document
│   └── adr/                   # one file per reversed or contested decision
├── src/
│   ├── gateway/
│   ├── session/
│   ├── ranking/
│   ├── guardrails/
│   ├── compliance/
│   ├── explain/
│   └── shared/
├── retrieval_service/
├── trainer/
│   ├── sample_data.py
│   ├── train_towers.py
│   ├── train_ranker.py
│   └── build_index.py
├── tests/
│   ├── test_guardrails.py     # property-based; the 25% dimension
│   ├── test_compliance.py     # erasure completeness
│   └── test_latency.py        # asserts p99 budget
├── bench/
│   ├── locustfile.py
│   └── cost_model.py
├── ui/
└── docker-compose.yml         # postgres + redis only
```

---

## 15. Build order

Dependency-ordered. Each step ends at a committable state.

| # | Deliverable | Done when |
|---|---|---|
| 1 | Repo, `CLAUDE.md`, README, skeleton | `pytest` passes on an empty suite |
| 2 | Data sampling, time-based split | Train/val parquet files exist, no leakage |
| 3 | Co-visitation baseline | A recall@20 number exists to beat |
| 4 | Two-tower training | Val loss decreases; recall@20 beats step 3 |
| 5 | Faiss index + retrieval service | `search()` returns in <5ms |
| 6 | Session encoder + intent heads | Heads are non-degenerate (cosine < 0.8) |
| 7 | Multi-task ranker | Beats retrieval-order baseline |
| 8 | **Guardrails + compliance** | All property tests green |
| 9 | API service wiring | End-to-end request returns a real feed |
| 10 | UI: three surfaces + intent panel | Demoable |
| 11 | Load test + cost model | Latency chart and cost number exist |
| 12 | README with real numbers, demo recorded | Submitted |

**Step 8 is not negotiable and comes before the API.** It is the highest-weighted dimension and it is pure functions with no dependencies — the cheapest points in the entire rubric. If time runs out, an unglamorous system with airtight guardrails scores better than a beautiful one without them.
