# n8n workflows

| File | Trigger | What it does |
|---|---|---|
| `ingest-document.json` | `POST /webhook/ingest` | extraction-service `/v1/extract/url` → chunking-service `/v1/index` → responds with chunk/cache stats |
| `query.json` | `POST /webhook/query` | retrieval-service `/v1/search` (hybrid + ColBERT + Cohere rerank) |
| `dlq-error-handler.json` | Error Trigger | posts orchestration failures to `/internal/dlq` (Redis Stream `rag.dlq`) |

## Import

```bash
make import-workflows          # n8n import:workflow --separate --input=/home/node/workflows
```

Then, in the n8n UI:

1. **Credentials → New → Header Auth**, name it `RAG Service JWT`, header `Authorization`,
   value `Bearer <token>` where `<token>` comes from `make token SUB=n8n TTL=2592000`.
2. Open each imported workflow, re-select that credential on the HTTP Request nodes
   (the export carries a placeholder id), and activate the workflow.
3. **Workflow settings → Error workflow** on `RAG - Ingest Document` and `RAG - Query`:
   choose `RAG - DLQ Error Handler`.

Example ingest request:

```bash
curl -X POST http://localhost:5678/webhook/ingest \
  -H 'Content-Type: application/json' \
  -d '{"document_id":"kb-001","url":"https://example.com/article","tenant_id":"acme","render":false}'
```

Example query:

```bash
curl -X POST http://localhost:5678/webhook/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does contextual retrieval work","top_k":5,"tenant_id":"acme"}'
```

Workers execute these HTTP Request nodes; because `WEBHOOK_URL` is set on every
n8n instance, any webhook URL a worker generates resolves to the public address
instead of `localhost:5678`.
