#!/usr/bin/env python
"""Mint a service-to-service JWT (for n8n's Header Auth credential or curl).

    python scripts/mint_service_token.py --subject n8n --ttl 86400

Reads SERVICE_JWT_SECRET / ISSUER / AUDIENCE from the environment or .env.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "rag_common"))

from rag_common.auth import mint_service_token  # noqa: E402


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Mint a service JWT")
    parser.add_argument("--subject", default="n8n", help="caller identity (JWT sub)")
    parser.add_argument("--ttl", type=int, default=3600, help="lifetime in seconds")
    args = parser.parse_args()
    _load_dotenv(ROOT / ".env")
    secret = os.environ.get("SERVICE_JWT_SECRET", "")
    if len(secret) < 32:
        print("SERVICE_JWT_SECRET missing or shorter than 32 chars", file=sys.stderr)
        return 2
    token = mint_service_token(
        secret=secret,
        issuer=os.environ.get("SERVICE_JWT_ISSUER", "contextual-rag"),
        audience=os.environ.get("SERVICE_JWT_AUDIENCE", "rag-services"),
        subject=args.subject,
        ttl_seconds=args.ttl,
    )
    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
