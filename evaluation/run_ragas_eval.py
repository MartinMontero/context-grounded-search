#!/usr/bin/env python
"""CLI wrapper: ``python evaluation/run_ragas_eval.py --dataset ... --thresholds ...``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evaluation"), str(ROOT / "packages" / "rag_common")]

from ragas_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
