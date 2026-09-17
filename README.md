# context-grounded-search

## The problem

You've got thousands of newsletters, articles, emails, and documents full of insights
from people you trust — but they're trapped in inboxes and folders designed for reading
one at a time. You don't have the time or headspace to re-read thousands of posts to find
the one insight you need right now. What you actually want is to *ask questions* and get
answers grounded in everything you've already collected.

Most AI search tools chop your documents into small pieces and try to match keywords or
meaning against your question. The problem is that each piece loses the context of the
document it came from — so the AI is searching fragments without understanding the bigger
picture, and the answers suffer for it.

## What this does differently

This system processes your documents so AI can actually understand them in context — and
give you answers that are grounded in what your sources actually said, not hallucinated
from thin air.

When you feed a document in, it doesn't just chop it into pieces and search them later.
It uses Claude (Anthropic's AI) to read the *whole* document and write a short summary
for each piece, explaining where that piece fits in the bigger picture. When you later
ask a question, the system searches those enriched pieces using three different search
strategies at once, then ranks the results with a separate reranking model. The result
is more relevant answers with less hallucination.

Think of it as turning your personal archive — newsletters, articles, emails, research —
into an on-demand knowledge base you can query for advice, answers, and context whenever
you need it.

The whole thing runs as a set of small, independent services coordinated by n8n (a
workflow automation tool), all inside Docker on your own machine. You bring your own API
keys — nothing is hard-coded or shared.

```
                ┌──────────────┐   job queue    ┌──────────────┐
  webhooks ───▶ │  n8n-main    │ ──▶ Redis ◀─── │ n8n-worker×N │ ──┐  HTTP + JWT
  webhooks ───▶ │  n8n-webhook │      │         └──────────────┘   │
                └──────┬───────┘      │                            ▼
                       │ PostgreSQL   │  error queue           ┌────────────────────┐
                       ▼              └───────────────────────│ extraction-service │──▶ defuddle-sidecar
                                                              │ chunking-service   │──▶ Anthropic + Qdrant
                                                              │ retrieval-service  │──▶ Qdrant + Cohere
                                                              └────────────────────┘
```

## Quick start

```bash
cp .env.example .env            # or: make env  (generates secrets for you)
```

Open `.env` and fill in your API keys: `ANTHROPIC_API_KEY` and `COHERE_API_KEY` are
required, `OPENAI_API_KEY` is optional (used only for evaluation embeddings).

```bash
make up                         # builds and starts all services — waits until everything is healthy
make import-workflows           # loads the three n8n workflows into the n8n editor
make token SUB=n8n              # creates the authentication token n8n needs to call the services
```

| What | Where | Notes |
|---|---|---|
| n8n editor (build and run workflows) | [localhost:5678](http://localhost:5678) | webhook processor on port 5679 |
| Extraction service (pull text from URLs/emails) | [localhost:8001/docs](http://localhost:8001/docs) | interactive API docs |
| Chunking service (split + contextualize + index) | [localhost:8002/docs](http://localhost:8002/docs) | interactive API docs |
| Retrieval service (search + rerank) | [localhost:8003/docs](http://localhost:8003/docs) | interactive API docs |
| Qdrant dashboard (see your indexed vectors) | [localhost:6333/dashboard](http://localhost:6333/dashboard) | |
| Jaeger tracing UI (optional, see below) | [localhost:16686](http://localhost:16686) | only with `--profile observability` |

Every service has two health-check endpoints that don't need authentication:
- `GET /health` — "is the service running?" (basic liveness check)
- `GET /ready` — "is everything this service depends on actually working?" (checks Redis, Qdrant, API keys, etc. — returns 503 until all dependencies are confirmed)

Everything else requires a signed token in the `Authorization: Bearer <token>` header,
and every caller is rate-limited to prevent abuse.

```bash
TOKEN=$(make -s token)

# Step 1: Extract text from a URL
curl -s localhost:8001/v1/extract/url \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com/article","document_id":"kb-001"}'

# Step 2: Contextualize, embed, and index the extracted text
curl -s localhost:8002/v1/index \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"document_id":"kb-001","text":"...extracted markdown...","title":"Article","tenant_id":"acme"}'

# Step 3: Search your indexed documents
curl -s localhost:8003/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"query":"how does contextual retrieval work","top_k":5,"tenant_id":"acme"}'
```

## Repository layout

```
docker-compose.yml           all the services and how they connect, with health checks
.env.example                 every configuration variable, no real secrets
pyproject.toml               shared Python project config with pinned dependencies
packages/rag_common/         shared code used by all three Python services (see section 2)
services/extraction_service/ pulls text out of web pages, emails, and raw HTML
services/defuddle_sidecar/   a Node.js helper that renders JavaScript-heavy pages in a real browser
services/contextual_chunking_service/  splits documents, adds context with Claude, stores in Qdrant
services/retrieval_service/  searches Qdrant and reranks results with Cohere
n8n/workflows/               three pre-built workflows: ingest, search, and error handling
evaluation/                  automated quality testing using the RAGAS framework
benchmarks/                  performance benchmarks for the ColBERT search layer
schemas/                     exported data schemas (kept in sync by tests)
scripts/                     utility scripts (token minting, schema export)
.github/workflows/ci.yml     automated checks: linting, tests, type checking, smoke tests
```

## 1 · Infrastructure (Docker Compose)

Everything runs in Docker containers, orchestrated by a single `docker-compose.yml` file.
When you run `docker compose up --wait`, it starts services in the right order and waits
for each one to report healthy before starting anything that depends on it.

The supporting services:

- **PostgreSQL 17** stores n8n's workflow data and execution history. Its health check
  verifies the database is accepting connections (not just starting up).
- **Redis 7.4** does double duty: it's the job queue for n8n (database 0) and also
  handles rate limiting and the error queue (database 1). It requires a password and
  keeps data on disk so nothing is lost on restart.
- **Qdrant 1.19.1** is the vector database where all the document chunks and their
  embeddings are stored. It requires an API key and exposes both HTTP and gRPC ports.

n8n (the workflow orchestrator) runs as three separate containers that work together:

- **n8n-main** handles the editor UI and runs database migrations
- **n8n-worker** picks jobs off the Redis queue and executes them (you can scale this
  to multiple workers with `--scale n8n-worker=3`)
- **n8n-webhook** receives incoming HTTP requests and queues them as jobs

All three share the same configuration: queue mode enabled, a single encryption key,
the same webhook URL (so workers know where to send callbacks), and security settings
that prevent workflows from reading environment variables or accessing the filesystem.

The three Python services and the Node.js sidecar are all built from this repo's Dockerfiles.

**Optional observability:** Add `--profile observability` to your `docker compose up`
command to include Jaeger, a distributed tracing system. Set
`OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317` in your `.env` to start sending traces.

## 2 · Shared service code (`rag_common`)

All three Python services share a common library that handles the basics so each service
only has to implement its own logic:

- **Configuration** — each service reads its settings from environment variables, with
  type checking, defaults, and validation. Secrets (like API keys) are stored as
  `SecretStr` so they never accidentally show up in logs or error messages.
- **Middleware stack** — every incoming request goes through the same layers, in a
  tested order: cross-origin handling, trusted host checking, response compression,
  and then a request-context layer that assigns a unique ID to each request, adds
  security headers, writes a structured log line, and wraps any uncaught errors in a
  consistent format.
- **Authentication** — services authenticate each other using JWT tokens (short-lived
  signed tokens that carry the caller's identity). The token format is locked down:
  it must include who issued it, who it's for, when it expires, and a unique ID.
  The signing algorithm is pinned to HS256 so an attacker can't swap in a weaker one.
  The same tokens work for both the Python services and the Node.js sidecar.
- **Rate limiting** — each authenticated caller gets a fixed number of requests per time
  window, tracked in Redis. Standard rate-limit headers tell the caller how many requests
  they have left and when the window resets. If they exceed the limit, they get a 429
  response with a `Retry-After` header. If Redis goes down, the limiter can be configured
  to either fail open (allow all) or fail closed (deny all).
- **Error queue (DLQ)** — when any processing stage fails, the error details (document ID,
  which stage failed, error type, error message, timestamp) are written to a Redis Stream
  called `rag.dlq`. There's an n8n workflow that monitors this queue and can alert on errors
  or retry failed jobs.
- **Logging and tracing** — every log line is a single JSON object with a trace ID, span
  ID, and request ID so you can follow a request across services. When Jaeger is running,
  traces are exported via OpenTelemetry with instrumentation on HTTP calls, Redis, and
  the API framework itself.
- **Error format** — all error responses follow the same shape:
  `{"error": {"type", "message", "request_id", "details"}}` so callers always know what
  to parse.

## 3 · Extraction service

This service pulls clean text out of different source formats:

- **`POST /v1/extract/url`** — Give it a URL, and it fetches the page, strips out the
  boilerplate (navigation, ads, sidebars), and returns the main content as Markdown.

  This endpoint is hardened against SSRF attacks (where a server is tricked into making
  requests to internal systems). Before connecting to any URL, it: only allows HTTP and
  HTTPS, blocks private/internal IP addresses (localhost, 10.x.x.x, 169.254.x.x,
  and many others including IPv6 equivalents), resolves the hostname to an IP address
  *before* connecting and checks that the resolved address is public, pins the connection
  to that verified IP so DNS can't change out from under it, re-checks every redirect
  hop, limits file size, and enforces timeouts.

  Set `render=true` to send the URL to the Defuddle sidecar for JavaScript-heavy pages
  (the sidecar applies the same security checks to every request the browser makes).

- **`POST /v1/extract/html`** — Give it raw HTML and get back clean Markdown. Uses
  trafilatura for extraction. If the result looks thin, it can automatically escalate
  to the Defuddle sidecar for a better extraction (`engine=auto`), or you can force
  the sidecar with `engine=defuddle`.

- **`POST /v1/extract/mime`** — Give it a raw email file (RFC 822 format), and it
  extracts the body text (cleaning HTML if needed), lists attachments, and pulls text
  from text-type attachments.

- **Defuddle sidecar** — A Node.js service that runs a real Chromium browser (via
  Playwright) for pages that need JavaScript to render their content. Before any page
  scripts run, it forces all Shadow DOM elements to be visible so the content extractor
  can see them. It blocks requests to private IP addresses, media files, fonts, and
  WebSocket connections. Chromium runs as a non-root user on a pinned, versioned image.

## 4 · Contextual chunking service

This is the core of the contextual retrieval approach. It takes extracted text and turns
it into search-ready, context-enriched chunks stored in Qdrant.

**`POST /v1/index`** runs the full pipeline: split → contextualize → embed → store.
**`POST /v1/chunk`** runs just the split + contextualize steps if you want the chunks
without storing them.

**How the splitting works:** The text is divided into paragraph-aware chunks of about
300 words each, with 40 words of overlap between consecutive chunks so nothing falls
through the cracks.

**How contextualizing works (the key innovation):** For each chunk, Claude reads the
*entire* document and writes a short summary explaining how that specific chunk fits
into the whole. This is the "contextual retrieval" technique from Anthropic's research —
it dramatically improves search accuracy because each chunk carries its own context
instead of being an isolated fragment.

To avoid re-reading the entire document for every single chunk (which would be expensive),
the service uses Anthropic's prompt caching. The full document is sent as a cached block
in the first request. Claude reads and caches it once, and all subsequent chunks in that
document reuse the cache (which lasts 5 minutes). The first chunk is sent alone to
establish the cache; then the remaining chunks fan out in parallel under a concurrency
limit. The response includes cache hit/miss statistics so you can verify it's working.

**Model note:** The system defaults to `claude-haiku-4-5` (Anthropic's fast, affordable
model). You can change this with the `ANTHROPIC_MODEL` environment variable. The cache
requires a minimum of 4,096 tokens in the cached block — smaller documents won't benefit
from caching.

**How the embeddings work:** Each chunk is converted into three different kinds of
searchable vectors:

1. **Dense vectors** (384 dimensions) — the standard "meaning-based" embedding. Good
   at finding conceptually similar content even when the exact words differ.
2. **Sparse vectors (BM25)** — keyword-based scoring with IDF weighting. Good at
   finding exact term matches, especially for technical jargon or proper nouns that
   dense embeddings might miss.
3. **ColBERT multi-vectors** (128 dimensions each) — a more granular approach where
   each *token* in the chunk gets its own vector. This enables fine-grained matching
   where individual words in the query are compared against individual words in the
   chunk. Qdrant's MaxSim comparator finds the best alignment.

All three are generated locally using FastEmbed (no external API calls for embeddings).

**How storage works:** Points are uploaded to Qdrant with deterministic IDs (based on
document ID + chunk index), so re-indexing a document automatically replaces its old
chunks. The collection has indexes on `document_id`, `tenant_id`, `source`, `chunk_index`,
and `created_at` for fast filtered queries.

**Strict data validation:** The output schema (`ContextualChunk`) is exported to
`schemas/contextual_chunk.schema.json` and a test fails if the code's model and the
exported schema ever get out of sync.

## 5 · Retrieval service

**`POST /v1/search`** takes a query and returns the most relevant chunks, using all
three search strategies together in a single request to Qdrant:

1. **First pass — hybrid search:** The query is embedded using both dense and sparse
   (BM25) vectors. Both sets of results (up to 100 each) are merged using Reciprocal
   Rank Fusion (RRF), which combines the rankings from both strategies into a single
   list. This catches both conceptual matches and exact keyword matches.

2. **Second pass — ColBERT reranking:** The top 50 results from the hybrid search are
   re-scored using ColBERT's token-level matching (MaxSim). This is more precise than
   either individual search strategy and pushes the most relevant results to the top.

3. **Third pass — Cohere reranking:** The top 20 results go to Cohere's `rerank-v4.0-pro`
   model, which reads the actual text of each chunk and scores how well it answers the
   query. Results are formatted as YAML documents with title, context, text, and source
   URL, giving the reranker full context for its judgment.

You can turn off ColBERT (`colbert=false`) or Cohere reranking (`rerank=false`)
individually to compare how each stage contributes to result quality.

Cohere API calls use a retry strategy with randomized backoff (full jitter): if a
request fails with a rate limit (429) or server error (5xx), it retries up to 5 times
with increasing but randomized delays (1s base, 30s max), and respects Cohere's
`Retry-After` header when present.

## 6 · Quality evaluation (RAGAS) and benchmarks

```bash
make eval        # run the quality evaluation suite
make bench       # run the ColBERT latency benchmark
```

The project includes an automated quality gate using the RAGAS evaluation framework.
It tests the pipeline against a fixed set of questions and expected answers
(`evaluation/fixtures/eval_dataset.json`) and scores four dimensions:

| Metric | What it measures | Minimum to pass |
|---|---|---|
| Context precision | Are the retrieved chunks actually relevant? | 0.70 |
| Context recall | Did we find all the relevant chunks? | 0.70 |
| Faithfulness | Does the answer stick to what the retrieved chunks say? | 0.80 |
| Answer relevancy | Does the answer actually address the question asked? | 0.75 |

If any score falls below its threshold, the evaluation exits with a failure code (useful
in CI to block a merge). Results are also saved as a JSON report.

The evaluation uses `claude-haiku-4-5` as its judge model. For the answer-relevancy
embeddings, it uses OpenAI's `text-embedding-3-small` if you've provided an
`OPENAI_API_KEY`, otherwise it falls back to the local FastEmbed model.

The **ColBERT benchmark** (`make bench`) seeds a test collection and measures search
latency (p50, p95, mean) and recall, comparing hybrid search alone vs. hybrid + ColBERT
reranking.

## 7 · CI (automated checks)

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull
request:

1. **Code style** — ruff checks for lint issues and formatting
2. **Python tests** — runs the full test suite without needing any external services
   (Redis is faked, API providers are mocked)
3. **Sidecar tests** — runs the Node.js sidecar's own test suite
4. **Docker validation** — checks that `docker-compose.yml` is valid and Dockerfiles
   pass hadolint (a Dockerfile linter)
5. **Full stack smoke test** (main branch only) — spins up the entire Docker Compose
   stack and hits every service's `/health` and `/ready` endpoints
6. **RAGAS quality gate** (manually triggered) — runs the full evaluation suite against
   real API endpoints

Run any of these locally: `make lint`, `make test`, `make test-node`, `make typecheck`
(mypy, fully passing), `make compose-check`.

## Security

- **No secrets in the repo.** All API keys, passwords, and tokens come from your `.env`
  file, which is gitignored. `.env.example` shows every variable with safe placeholder
  values.
- **Every internal call is authenticated.** Services talk to each other using short-lived
  signed tokens (JWT with HS256). The signing secret must be at least 32 characters.
- **Rate limiting on every endpoint.** Each authenticated caller gets a fixed request
  budget per time window, tracked in Redis.
- **Request safety.** Trusted host and cross-origin allow-lists, `X-Content-Type-Options`
  header (prevents browsers from guessing file types), `Cache-Control: no-store` (nothing
  cached by intermediaries), and size limits on every input.
- **SSRF protection in depth.** Both the Python extraction service and the Node.js browser
  sidecar independently validate that outbound requests only go to public internet
  addresses — private IPs, localhost, link-local, and other internal ranges are all blocked,
  and DNS lookups are verified before connections are made.
- **Chromium runs unprivileged.** The browser sidecar's Chromium process runs as a
  non-root user.
- **n8n is sandboxed.** Workflows can't read environment variables or access the host
  filesystem, and n8n never receives any API keys for the AI providers.
- **Databases require credentials even internally.** Qdrant and Redis both require
  authentication, even though they're only accessible within the Docker network. Host
  ports for data stores are bound to `127.0.0.1` only (not exposed to your network).
