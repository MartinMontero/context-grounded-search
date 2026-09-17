# syntax=docker/dockerfile:1.7
# Build context: repository root (shared rag_common package + pinned pyproject).
FROM python:3.12-slim-trixie AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY packages ./packages
COPY services/extraction_service ./services/extraction_service
COPY services/contextual_chunking_service ./services/contextual_chunking_service
COPY services/retrieval_service ./services/retrieval_service
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir ".[retrieval]"

FROM python:3.12-slim-trixie
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/models
RUN groupadd -r app \
 && useradd -r -g app -d /app -s /usr/sbin/nologin app \
 && mkdir -p /app /models \
 && chown -R app:app /app /models
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
USER app
EXPOSE 8000
CMD ["uvicorn", "--factory", "retrieval_service.main:build_app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--no-server-header", "--timeout-graceful-shutdown", "30"]
