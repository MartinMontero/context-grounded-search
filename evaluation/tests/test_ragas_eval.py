from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import ragas_eval
from ragas import EvaluationDataset, SingleTurnSample

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_fixture_builds_single_turn_samples_and_dataset() -> None:
    samples = ragas_eval.load_samples(FIXTURES / "eval_dataset.json")
    assert len(samples) == 10
    assert all(isinstance(s, SingleTurnSample) for s in samples)
    first = samples[0]
    assert first.user_input.startswith("How does contextual retrieval")
    assert len(first.retrieved_contexts) == 2 and first.reference and first.response
    dataset = ragas_eval.load_dataset(FIXTURES / "eval_dataset.json")
    assert isinstance(dataset, EvaluationDataset) and len(dataset) == 10
    assert set(dataset.features()) >= {"user_input", "response", "retrieved_contexts", "reference"}


def test_fixture_validation_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"user_input": "q", "response": "r", "reference": "x"}]))
    with pytest.raises(ValueError, match=r"missing \['retrieved_contexts'\]"):
        ragas_eval.load_samples(bad)
    bad.write_text(
        json.dumps(
            [{"user_input": "q", "response": "r", "reference": "x", "retrieved_contexts": []}]
        )
    )
    with pytest.raises(ValueError, match="non-empty retrieved_contexts"):
        ragas_eval.load_samples(bad)


def test_thresholds_file_and_check() -> None:
    thresholds = ragas_eval.load_thresholds(FIXTURES.parent / "thresholds.yaml")
    assert thresholds == {
        "context_precision": 0.70,
        "context_recall": 0.70,
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
    }
    passing = {
        "context_precision": 0.9,
        "context_recall": 0.8,
        "faithfulness": 0.95,
        "answer_relevancy": 0.8,
    }
    assert ragas_eval.check_thresholds(passing, thresholds) == []
    failing = {**passing, "faithfulness": 0.6, "answer_relevancy": math.nan}
    failures = ragas_eval.check_thresholds(failing, thresholds)
    assert [(f.metric, f.threshold) for f in failures] == [
        ("faithfulness", 0.8),
        ("answer_relevancy", 0.75),
    ]


def test_mean_scores_reads_metric_columns() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "context_precision": [1.0, 0.5],
            "context_recall": [1.0, 1.0],
            "faithfulness": [0.8, 0.6],
            "answer_relevancy": [0.9, 0.7],
            "user_input": ["a", "b"],
        }
    )
    result = SimpleNamespace(to_pandas=lambda: frame)
    assert ragas_eval.mean_scores(result) == {
        "context_precision": 0.75,
        "context_recall": 1.0,
        "faithfulness": 0.7,
        "answer_relevancy": 0.8,
    }
    with pytest.raises(ValueError, match="faithfulness"):
        ragas_eval.mean_scores(
            SimpleNamespace(to_pandas=lambda: frame.drop(columns=["faithfulness"]))
        )


def test_metrics_use_explicit_embeddings_for_answer_relevancy() -> None:
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    llm, embeddings = object(), object()
    metrics = ragas_eval.build_metrics(llm, embeddings)
    assert [type(m) for m in metrics] == [
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        AnswerRelevancy,
    ]
    assert all(m.llm is llm for m in metrics)
    assert metrics[3].embeddings is embeddings
    assert [m.name for m in metrics] == list(ragas_eval.METRIC_NAMES)


def test_embeddings_provider_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragas.embeddings import OpenAIEmbeddings

    openai_embeddings = ragas_eval.build_embeddings(
        openai_api_key="sk-test", embedding_model="text-embedding-3-small", fastembed_model="x"
    )
    assert isinstance(openai_embeddings, OpenAIEmbeddings)
    assert openai_embeddings.model == "text-embedding-3-small"

    created: dict[str, str] = {}

    class FakeFastEmbed:
        def __init__(self, model_name: str):
            created["model"] = model_name
            self.adapter = "fastembed-adapter"

    monkeypatch.setattr(ragas_eval, "FastEmbedRagasEmbeddings", FakeFastEmbed)
    fallback = ragas_eval.build_embeddings(
        openai_api_key=None,
        embedding_model="text-embedding-3-small",
        fastembed_model="BAAI/bge-small-en-v1.5",
    )
    assert fallback == "fastembed-adapter" and created["model"] == "BAAI/bge-small-en-v1.5"


def test_cli_dry_run_validates_without_calling_providers(capsys) -> None:
    code = ragas_eval.main(
        [
            "--dataset",
            str(FIXTURES / "eval_dataset.json"),
            "--thresholds",
            str(FIXTURES.parent / "thresholds.yaml"),
            "--dry-run",
        ]
    )
    assert code == 0
