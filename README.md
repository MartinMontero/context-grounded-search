# contextual-rag

Event-driven **Contextual RAG** pipeline: n8n (queue mode) orchestrates three
JWT-protected FastAPI services that extract documents, situate every chunk with
Claude (prompt-cached contextual retrieval), index dense + BM25 + ColBERT vectors
into Qdrant, and answer queries with hybrid search, ColBERT late interaction and
Cohere reranking. RAGAS gates quality in CI.

```
                ┌──────────────┐   Bull queue   ┌──────────────┐
  webhooks ───▶ │  n8n-main    │ ──▶ Redis ◀─── │ n8n-worker×N │ ──┐  HTTP + JWT
  webhooks ───▶ │  n8n-webhook │      │         └──────────────┘   │
                └──────┬───────┘      │                            ▼
                       │ PostgreSQL   │  DLQ stream rag.dlq   ┌────────────────────┐
                       ▼              └───────────────────────│ extraction-service │──▶ defuddle-sidecar
                                                              │ chunking-service   │──▶ Anthropic + Qdrant
                                                              │ retrieval-service  │──▶ Qdrant + Cohere
                                                              └────────────────────┘
```

## Quick start

```bash
cp .env.example .env            # or: make env  (generates the secrets for you)
#   fill in ANTHROPIC_API_KEY, COHERE_API_KEY (OPENAI_API_KEY optional)
make up                         # docker compose up -d --build --wait  → all services healthy
make import-workflows           # load n8n/workflows/*.json into n8n-main
make token SUB=n8n              # mint the JWT for n8n's "RAG Service JWT" header credential
```

| Service | Host port | Docs |
|---|---|---|
| n8n editor / webhooks | 5678 (webhook processor: 5679) | http://localhost:5678 |
| extraction-service | 8001 | http://localhost:8001/docs |
| chunking-service | 8002 | http://localhost:8002/docs |
| retrieval-service | 8003 | http://localhost:8003/docs |
| Qdrant dashboard | 6333 | http://localhost:6333/dashboard |
| Jaeger UI (`--profile observability`) | 16686 | http://localhost:16686 |

Every FastAPI service exposes unauthenticated `GET /health` (liveness) and
`GET /ready` (dependency checks: Redis, Qdrant collection, sidecar, API keys → 503 until green).
Everything else needs `Authorization: Bearer <service JWT>` and is rate limited per caller.

```bash
TOKEN=$(make -s token)
curl -s localhost:8001/v1/extract/url -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"url":"https://example.com/article","document_id":"kb-001"}'
curl -s localhost:8002/v1/index -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"document_id":"kb-001","text":"...extracted markdown...","title":"Article","tenant_id":"acme"}'
curl -s localhost:8003/v1/search -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"query":"how does contextual retrieval work","top_k":5,"tenant_id":"acme"}'
```

## Repository layout

```
docker-compose.yml           full topology, health checks, depends_on conditions
.env.example                 every variable, no secrets
pyproject.toml               one distribution, exact pins, extras per service (+ eval, dev)
packages/rag_common/         shared runtime: settings, JSON logs, OTel, JWT, rate limit, DLQ,
                             retry (full jitter), health, middleware order, app factory,
                             Qdrant collection schema, FastEmbed hybrid embedder
services/extraction_service/ MIME (email.policy.default) + trafilatura + SSRF-safe fetch
services/defuddle_sidecar/   Node.js: Defuddle (JSDOM) + Playwright/Chromium, Shadow-DOM flattening
services/contextual_chunking_service/  Claude contextual retrieval + Qdrant indexing
services/retrieval_service/  Universal Query (RRF → ColBERT) + Cohere rerank
n8n/workflows/               ingest, query and DLQ error-handler workflows (importable JSON)
evaluation/                  RAGAS eval (fixtures, thresholds, CLI), corpus for the benchmark
benchmarks/                  ColBERT latency benchmark
schemas/                     exported ContextualChunk JSON Schema (test-enforced)
scripts/                     mint_service_token.py, export_schemas.py
.github/workflows/ci.yml     lint, tests, sidecar tests, compose config/lint, stack smoke test, RAGAS gate
```

## 1 · Infrastructure (docker-compose)

* **PostgreSQL 17** (n8n state; `docker/postgres/initdb` creates the `n8n` database),
  health check `pg_isready -h 127.0.0.1` so the socket-only bootstrap phase never reports healthy.
* **Redis 7.4** with `--requirepass`, `appendonly`, `maxmemory-policy noeviction` (mandatory for Bull);
  db 0 = n8n queue, db 1 = rate limiting + DLQ. Health: authenticated `PING`.
* **Qdrant v1.19.1** with API key, gRPC on 6334; health via bash `/dev/tcp` (the image has no curl).
* **n8n 2.40.1** as `n8n-main`, `n8n-worker` (`command: worker`) and `n8n-webhook` (`command: webhook`),
  sharing one anchored environment:
  `EXECUTIONS_MODE=queue`, `QUEUE_BULL_REDIS_HOST=redis`, one `N8N_ENCRYPTION_KEY`,
  **`WEBHOOK_URL` on every instance** (otherwise workers build `localhost:5678` URLs),
  `N8N_CONCURRENCY_PRODUCTION_LIMIT=10` on workers, `QUEUE_HEALTH_CHECK_ACTIVE=true` for
  worker `/healthz` + `/healthz/readiness`, plus the n8n 2.x defaults made explicit
  (`N8N_RUNNERS_MODE=internal`, `N8N_BLOCK_ENV_ACCESS_IN_NODE=true`, `N8N_DEFAULT_BINARY_DATA_MODE=database`).
  Workers depend on a healthy main (migrations run first); scale with `--scale n8n-worker=3`.
* FastAPI services and the sidecar build from the repo; `depends_on` uses `condition: service_healthy`
  throughout so `docker compose up --wait` returns only when the whole graph is green.
* `--profile observability` adds Jaeger (OTLP gRPC 4317) — set `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317`.

## 2 · Shared service runtime (`rag_common`)

* **Settings**: pydantic-settings v2 with `SettingsConfigDict`, `validation_alias` per env var,
  `SecretStr` for secrets, CSV list parsing, `@lru_cache get_settings()` per service.
* **Middleware order** (asserted by a test): `CORSMiddleware → TrustedHostMiddleware → GZipMiddleware →
  RequestContext` (request-id propagation, security headers, structured access log, 500 envelope).
* **Auth**: HS256 service JWTs (`iss`/`aud`/`sub`/`exp`/`iat`/`jti` required, `alg` pinned).
  Mint with `scripts/mint_service_token.py`; the Node sidecar verifies the same tokens.
* **Rate limiting**: Redis fixed window keyed by JWT subject (`INCR`+`EXPIRE` in a MULTI), standard
  `X-RateLimit-*` headers, `429` + `Retry-After`; fail-open is configurable.
* **DLQ**: Redis Stream `rag.dlq`, fields exactly `document_id, stage, error_type, error_message, timestamp`;
  `dlq.guard()` wraps each stage; `POST /internal/dlq` lets the n8n error workflow publish.
* **Observability**: one JSON object per log line with `trace_id`/`span_id`/`request_id`; OpenTelemetry
  tracer with FastAPI, httpx and redis instrumentation; OTLP export when an endpoint is configured.
* **Errors**: uniform `{"error": {"type", "message", "request_id", "details"}}` envelope.

## 3 · Extraction service

* `POST /v1/extract/mime` — raw `message/rfc822` body or multipart `file`; parsed with the stdlib
  `email` package under `email.policy.default` (`get_body(('html','plain'))`, `iter_attachments()`),
  HTML bodies cleaned by trafilatura in recall mode, attachment inventory with text for `text/*`.
* `POST /v1/extract/html` — trafilatura 2.2 (lxml + `lxml_html_clean`) to Markdown with metadata;
  `engine=auto` escalates thin results to the Defuddle sidecar, `engine=defuddle` forces it.
* `POST /v1/extract/url` — **SSRF-hardened** fetch: http/https only, no userinfo, port allow-list,
  hostname deny-list, DNS resolved *before* connecting with every A/AAAA record required to be public
  (private, loopback, link-local, CGNAT, multicast, reserved, IPv4-mapped IPv6, ULA all blocked),
  connection pinned to the validated IP (Host + SNI keep the name → no DNS rebinding), redirects followed
  manually with re-validation per hop, content-type allow-list, streaming size cap, timeouts.
  `render=true` sends the URL to the sidecar, which repeats the checks on every Chromium request.
* **Defuddle sidecar** (`services/defuddle_sidecar`): Express + `defuddle/node` (JSDOM) and Playwright.
  Before any page script runs it forces `attachShadow` to `mode: 'open'`, then flattens every shadow root
  into its host so Defuddle sees Shadow-DOM content; `context.route` blocks non-public targets, media,
  fonts and websockets. Runs as `pwuser` on the pinned `mcr.microsoft.com/playwright:v1.63.0-noble` image.

## 4 · Contextual chunking service

`POST /v1/chunk` (chunks + contexts) and `POST /v1/index` (chunk → contextualize → embed → upload).

* Paragraph-aware splitting with word overlap (`CHUNK_WORDS`, `CHUNK_OVERLAP_WORDS`).
* **Anthropic contextual retrieval** (`contextualizer.py`): the full document is rendered once into a
  `<document>` text block with `"cache_control": {"type": "ephemeral"}` (5-minute TTL, the single
  breakpoint of the four allowed) and placed **first** in the user message; the chunk-specific prompt
  follows it. The block object is reused verbatim for every chunk of the document, so the cached prefix
  (system prompt + document) is byte-identical across requests. The first chunk is sent alone to write
  the cache; the remaining chunks fan out under a semaphore and read it. `count_tokens` measures the
  prefix and the response reports `cache_eligible`, cache write/read tokens and a warning when no reads
  were observed.
* **Model note**: the spec's `claude-3-5-haiku-20241022` (2048-token cache minimum) was retired on
  2026-02-19. The service defaults to the current Haiku generation, `claude-haiku-4-5`, whose minimum
  is **4096** tokens; both are settings (`ANTHROPIC_MODEL`, `ANTHROPIC_CACHE_MIN_TOKENS`).
* **Strict output schema**: `ContextualChunk` (`extra="forbid"`) — `schemas/contextual_chunk.schema.json`
  is exported from the model and a test fails if it drifts; also served at `/v1/schemas/contextual-chunk`.
* **Qdrant** (`rag_common/qdrant_schema.py`): `qdrant-client==1.19.1`, `prefer_grpc=True`.
  Vectors: `dense` 384-d COSINE, `sparse` `SparseVectorParams(modifier=IDF)` with client-side
  FastEmbed `Qdrant/bm25`, `colbert` 128-d COSINE `MultiVectorComparator.MAX_SIM` with `HnswConfigDiff(m=0)`.
  Payload indexes (`document_id`, `tenant_id`, `source`, `chunk_index`, `created_at`) are created at
  startup and **asserted immediately before** `upload_points(batch_size=64, parallel=2, wait=True)`.
  Point ids are `uuid5(document_id, chunk_index)`; a re-ingest deletes the document's stale points first.

## 5 · Retrieval service

`POST /v1/search` runs one Universal Query request:

```python
client.query_points(
    collection_name=...,
    prefetch=[
        models.Prefetch(  # nested prefetch object
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=100),
                models.Prefetch(query=models.SparseVector(...), using="sparse", limit=100),
            ],
            query=models.RrfQuery(rrf=models.Rrf()),  # fuse dense + sparse
            limit=50,
        )
    ],
    query=colbert_multivector,
    using="colbert",  # MaxSim rerank of the fused candidates
    limit=20,
)
```

The top candidates go to **Cohere `/v2/rerank`** (`rerank-v4.0-pro`) as YAML documents rendered with
`yaml.safe_dump(..., sort_keys=False)` (title → context → text → source_url); the returned `index`
values are mapped back to the candidates and sorted by `relevance_score`. `colbert=false` / `rerank=false`
switch stages off for comparison. Cohere calls use the shared **full-jitter retry**
(`max_retries=5`, `base_delay=1.0s`, `max_delay=30s`, `Retry-After` honoured) for 429/5xx/transport errors,
with the SDK's own retries disabled so the policy is authoritative.

## 6 · Evaluation (RAGAS) and benchmark

```bash
make eval        # python evaluation/run_ragas_eval.py --dataset ... --thresholds evaluation/thresholds.yaml
make bench       # python benchmarks/colbert_latency_benchmark.py --seed --iterations 20
```

* `ragas==0.4.3` (pinned, with `langchain-community==0.3.31` because 0.4.x removed a module ragas imports).
* Fixture `evaluation/fixtures/eval_dataset.json` → `SingleTurnSample(user_input, response,
  retrieved_contexts, reference)` → `EvaluationDataset`.
* `answer_relevancy` embeddings are configured explicitly: OpenAI `text-embedding-3-small` when
  `OPENAI_API_KEY` is set, otherwise the FastEmbed `BAAI/bge-small-en-v1.5` adapter.
* Judge LLM: Anthropic through `ragas.llms.llm_factory`. ragas 0.4 hard-codes `temperature`/`top_p`,
  which Opus 5 / Sonnet 5 reject, so the default judge is `claude-haiku-4-5` (`RAGAS_JUDGE_MODEL`).
* Thresholds (`evaluation/thresholds.yaml`): context_precision ≥ 0.70, context_recall ≥ 0.70,
  faithfulness ≥ 0.80, answer_relevancy ≥ 0.75 — the CLI exits 1 below any of them and writes a JSON report.
* The benchmark seeds `evaluation/fixtures/corpus.json` into a separate collection and reports p50/p95/mean
  latency and recall@k for `rrf` vs `rrf+colbert`.

## 7 · CI

`.github/workflows/ci.yml`: ruff lint + format, Python tests (no external services; fakeredis, mocked
providers), sidecar tests, `docker compose config` + hadolint, a full-stack smoke test on `main`
(`compose up --wait` then `/health` + `/ready` on every service), and a manually triggered RAGAS gate.

Local equivalents: `make lint`, `make test`, `make test-node`, `make typecheck` (mypy clean), `make compose-check`.

## Security model in one paragraph

Secrets only via `.env`; every internal hop authenticated with short-lived HS256 JWTs (`SERVICE_JWT_SECRET`
≥ 32 chars enforced); per-caller Redis rate limits; TrustedHost + CORS allow-lists; `X-Content-Type-Options`,
`Cache-Control: no-store`; payload size caps on every ingress; SSRF defence-in-depth in both Python and Node;
Chromium never runs as root; n8n blocks env access from Code nodes and never receives provider keys;
Qdrant and Redis require credentials even inside the compose network; host ports for data stores bind to
`127.0.0.1` only.
