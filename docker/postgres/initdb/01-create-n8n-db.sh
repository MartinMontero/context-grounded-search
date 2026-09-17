#!/bin/bash
# Runs once on first cluster initialisation. POSTGRES_DB (the RAG database) is
# created by the official image; this adds the dedicated n8n database.
set -euo pipefail

N8N_DB="${N8N_DB_NAME:-n8n}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE "${N8N_DB}" OWNER "${POSTGRES_USER}"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${N8N_DB}')\gexec
EOSQL
