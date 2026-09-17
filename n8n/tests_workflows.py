"""Structural validation of the exported n8n workflows (run by pytest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).parent / "workflows").glob("*.json"))
SERVICE_URLS = ("http://extraction-service:8000/", "http://chunking-service:8000/", "http://retrieval-service:8000/")


@pytest.mark.parametrize("path", WORKFLOWS, ids=[p.name for p in WORKFLOWS])
def test_workflow_is_well_formed(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {n["name"] for n in data["nodes"]}
    assert len(names) == len(data["nodes"]), "duplicate node names"
    for source, outputs in data["connections"].items():
        assert source in names
        for branch in outputs["main"]:
            for edge in branch:
                assert edge["node"] in names, f"{source} -> {edge['node']} missing"
    for node in data["nodes"]:
        if node["type"] == "n8n-nodes-base.httpRequest":
            params = node["parameters"]
            assert params["url"].startswith(SERVICE_URLS), params["url"]
            assert params["genericAuthType"] == "httpHeaderAuth"
            assert node["credentials"]["httpHeaderAuth"]["name"] == "RAG Service JWT"
            assert params["jsonBody"].startswith("={{ JSON.stringify(")
    assert data["settings"]["executionOrder"] == "v1"
    assert data["active"] is False


def test_no_workflow_declares_tags() -> None:
    """n8n imports a directory in one transaction and does not dedupe tags across files.

    Two workflows declaring the same tag collide on the unique index tag_entity(name)
    and roll the entire import back ("Successfully imported 0 workflows").
    """
    for wf in WORKFLOWS:
        data = json.loads(wf.read_text(encoding="utf-8"))
        assert "tags" not in data, f"{wf.name} declares tags; that breaks batch import"


def test_expected_workflows_present() -> None:
    assert [p.name for p in WORKFLOWS] == ["dlq-error-handler.json", "ingest-document.json", "query.json"]
