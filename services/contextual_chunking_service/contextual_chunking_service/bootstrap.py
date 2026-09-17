"""``python -m contextual_chunking_service.bootstrap`` — create collection + payload indexes."""

from __future__ import annotations

import logging

from contextual_chunking_service.settings import get_settings
from rag_common.logging import configure_logging
from rag_common.qdrant_schema import ensure_collection, make_client


def main() -> None:
    settings = get_settings()
    configure_logging(settings.service_name, settings.log_level)
    client = make_client(
        url=settings.qdrant_url,
        grpc_port=settings.qdrant_grpc_port,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
    )
    created = ensure_collection(client, settings.qdrant_collection)
    logging.getLogger(__name__).info(
        "bootstrap complete",
        extra={"collection": settings.qdrant_collection, "created": created},
    )


if __name__ == "__main__":
    main()
