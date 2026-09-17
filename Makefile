# contextual-rag — developer entrypoints
SHELL := /bin/bash
COMPOSE ?= docker compose
PY ?= python

.PHONY: help env up down logs ps build lint fmt test test-node typecheck \
        import-workflows token bootstrap-qdrant eval bench compose-check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example with generated secrets
	@test -f .env && echo ".env already exists" || ( \
	  cp .env.example .env && \
	  for k in POSTGRES_PASSWORD REDIS_PASSWORD QDRANT_API_KEY N8N_ENCRYPTION_KEY SERVICE_JWT_SECRET; do \
	    v=$$(openssl rand -hex 32); sed -i.bak "s|^$$k=change-me|$$k=$$v|" .env; \
	  done; rm -f .env.bak; echo "wrote .env with generated secrets" )

up: ## Start the full stack and wait for every service to be healthy
	$(COMPOSE) up -d --build --wait

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

logs: ## Tail logs
	$(COMPOSE) logs -f --tail=200

ps: ## Show service health
	$(COMPOSE) ps

build: ## Build all images
	$(COMPOSE) build

compose-check: ## Validate compose file interpolation
	$(COMPOSE) --env-file .env.example config --quiet && echo "compose OK"

lint: ## Ruff lint + format check
	ruff check .
	ruff format --check .

fmt: ## Auto-format
	ruff format .
	ruff check --fix .

test: ## Run Python unit tests (no external services required)
	pytest -q

test-node: ## Run Defuddle sidecar tests
	cd services/defuddle_sidecar && npm test

typecheck: ## mypy on shared package + services
	mypy -p rag_common -p extraction_service -p contextual_chunking_service -p retrieval_service

import-workflows: ## Import n8n workflows from ./n8n/workflows into the running n8n-main
	MSYS_NO_PATHCONV=1 $(COMPOSE) exec n8n-main n8n import:workflow --separate --input=/home/node/workflows/

token: ## Mint a service JWT for n8n / curl (SUB=n8n TTL=86400)
	$(PY) scripts/mint_service_token.py --subject $${SUB:-n8n} --ttl $${TTL:-86400}

bootstrap-qdrant: ## Create the Qdrant collection + payload indexes (idempotent)
	$(COMPOSE) exec chunking-service python -m contextual_chunking_service.bootstrap

eval: ## Run the RAGAS evaluation against the fixture dataset with CI thresholds
	$(PY) evaluation/run_ragas_eval.py --dataset evaluation/fixtures/eval_dataset.json --thresholds evaluation/thresholds.yaml

bench: ## ColBERT latency benchmark (requires running Qdrant; seeds the fixture corpus)
	$(PY) benchmarks/colbert_latency_benchmark.py --seed --iterations 20

clean: ## Remove caches
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache .mypy_cache
