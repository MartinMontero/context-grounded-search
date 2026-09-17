"""RAGAS evaluation with CI pass/fail thresholds.

Builds ``SingleTurnSample`` -> ``EvaluationDataset`` from a fixture file, scores
``context_precision``, ``context_recall``, ``faithfulness`` and
``answer_relevancy``, and exits non-zero when any mean score is below its
threshold in ``thresholds.yaml``.

Providers
* Judge LLM: Anthropic via ``ragas.llms.llm_factory`` (instructor adapter).
  ragas 0.4.x always sends ``temperature``/``top_p``, which Claude Opus 5 and
  Sonnet 5 reject with HTTP 400 - Haiku 4.5 accepts them, hence the default.
* ``answer_relevancy`` embeddings: OpenAI ``text-embedding-3-small`` when
  ``OPENAI_API_KEY`` is set, otherwise the FastEmbed dense model (same family the
  retrieval stack uses) through a ``BaseRagasEmbedding`` adapter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ragas import EvaluationDataset, SingleTurnSample

log = logging.getLogger("ragas_eval")

REQUIRED_KEYS = ("user_input", "response", "retrieved_contexts", "reference")
METRIC_NAMES = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


# --- dataset ---------------------------------------------------------------------


def load_samples(path: Path) -> list[SingleTurnSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw["samples"] if isinstance(raw, dict) else raw
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty list of samples")
    samples: list[SingleTurnSample] = []
    for i, record in enumerate(records):
        missing = [k for k in REQUIRED_KEYS if k not in record]
        if missing:
            raise ValueError(f"{path}: sample {i} is missing {missing}")
        if not isinstance(record["retrieved_contexts"], list) or not record["retrieved_contexts"]:
            raise ValueError(f"{path}: sample {i} needs a non-empty retrieved_contexts list")
        samples.append(
            SingleTurnSample(
                user_input=str(record["user_input"]),
                response=str(record["response"]),
                retrieved_contexts=[str(c) for c in record["retrieved_contexts"]],
                reference=str(record["reference"]),
            )
        )
    return samples


def load_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset(samples=load_samples(path))


# --- providers --------------------------------------------------------------------


def build_judge(*, model: str, api_key: str | None):
    import anthropic
    from ragas.llms import llm_factory

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    return llm_factory(model, provider="anthropic", client=client, max_tokens=2048)


class FastEmbedRagasEmbeddings:
    """``BaseRagasEmbedding`` adapter over FastEmbed (no API key required)."""

    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL, cache_dir: str | None = None):
        from fastembed import TextEmbedding
        from ragas.embeddings import BaseRagasEmbedding

        # Subclass at runtime so importing this module never needs the models loaded.
        class _Adapter(BaseRagasEmbedding):
            def __init__(self) -> None:
                super().__init__()
                self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

            def embed_text(self, text: str, **kwargs: Any) -> list[float]:
                return next(iter(self.model.embed([text]))).tolist()

            async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
                return await asyncio.to_thread(self.embed_text, text)

        self.adapter = _Adapter()
        self.model_name = model_name


def build_embeddings(*, openai_api_key: str | None, embedding_model: str, fastembed_model: str):
    """Explicit embedding provider for answer_relevancy."""
    if openai_api_key:
        import openai
        from ragas.embeddings import OpenAIEmbeddings

        log.info("answer_relevancy embeddings: OpenAI %s", embedding_model)
        return OpenAIEmbeddings(client=openai.OpenAI(api_key=openai_api_key), model=embedding_model)
    log.info("answer_relevancy embeddings: FastEmbed %s (OPENAI_API_KEY not set)", fastembed_model)
    return FastEmbedRagasEmbeddings(fastembed_model).adapter


def build_metrics(llm, embeddings) -> list:
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    return [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]


# --- scoring ----------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdFailure:
    metric: str
    score: float
    threshold: float


def load_thresholds(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    thresholds = {k: float(v) for k, v in data.get("thresholds", data).items()}
    missing = [m for m in METRIC_NAMES if m not in thresholds]
    if missing:
        raise ValueError(f"{path}: thresholds missing for {missing}")
    return thresholds


def mean_scores(result) -> dict[str, float]:
    frame = result.to_pandas()
    scores: dict[str, float] = {}
    for metric in METRIC_NAMES:
        if metric not in frame:
            raise ValueError(f"metric column '{metric}' missing from RAGAS result")
        scores[metric] = float(frame[metric].astype(float).mean())
    return scores


def check_thresholds(
    scores: dict[str, float], thresholds: dict[str, float]
) -> list[ThresholdFailure]:
    failures = []
    for metric, threshold in thresholds.items():
        score = scores.get(metric)
        if score is None or score != score or score < threshold:  # NaN counts as failure
            failures.append(
                ThresholdFailure(metric, float("nan") if score is None else score, threshold)
            )
    return failures


def run_evaluation(dataset: EvaluationDataset, *, llm, embeddings, max_workers: int = 4):
    from ragas import evaluate
    from ragas.run_config import RunConfig

    return evaluate(
        dataset,
        metrics=build_metrics(llm, embeddings),
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=max_workers, timeout=180),
        raise_exceptions=False,
        show_progress=False,
    )


# --- CLI --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation with CI thresholds")
    parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/fixtures/eval_dataset.json")
    )
    parser.add_argument("--thresholds", type=Path, default=Path("evaluation/thresholds.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/ragas_report.json"))
    parser.add_argument(
        "--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    parser.add_argument(
        "--embedding-model", default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    parser.add_argument(
        "--fastembed-model", default=os.getenv("DENSE_MODEL", DEFAULT_FASTEMBED_MODEL)
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the fixture + thresholds only"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    dataset = load_dataset(args.dataset)
    thresholds = load_thresholds(args.thresholds)
    log.info("loaded %d samples; thresholds=%s", len(dataset), thresholds)
    if args.dry_run:
        return 0

    llm = build_judge(model=args.judge_model, api_key=os.getenv("ANTHROPIC_API_KEY"))
    embeddings = build_embeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model=args.embedding_model,
        fastembed_model=args.fastembed_model,
    )
    result = run_evaluation(dataset, llm=llm, embeddings=embeddings, max_workers=args.max_workers)
    scores = mean_scores(result)
    failures = check_thresholds(scores, thresholds)

    report = {
        "judge_model": args.judge_model,
        "samples": len(dataset),
        "scores": scores,
        "thresholds": thresholds,
        "passed": not failures,
        "failures": [f.__dict__ for f in failures],
        "per_sample": json.loads(result.to_pandas().to_json(orient="records")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for metric, score in scores.items():
        status = "PASS" if score >= thresholds[metric] else "FAIL"
        print(f"{status}  {metric:<18} {score:.3f}  (threshold {thresholds[metric]:.2f})")
    if failures:
        print(f"\n{len(failures)} metric(s) below threshold - see {args.output}", file=sys.stderr)
        return 1
    print(f"\nall thresholds met - report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
