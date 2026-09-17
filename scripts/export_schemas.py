"""Export the ContextualChunk JSON Schema to schemas/contextual_chunk.schema.json.

Run after changing the model; ``tests/test_indexer_pipeline_api.py`` fails when
the committed file drifts from the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "packages" / "rag_common"),
    str(ROOT / "services" / "contextual_chunking_service"),
]

from contextual_chunking_service.schemas import ContextualChunk  # noqa: E402


def main() -> None:
    target = ROOT / "schemas" / "contextual_chunk.schema.json"
    target.write_text(
        json.dumps(ContextualChunk.model_json_schema(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
