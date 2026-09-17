from __future__ import annotations

import ipaddress
import json
from email.message import EmailMessage
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from extraction_service.html_extractor import extract_html
from extraction_service.main import build_app
from extraction_service.mime_extractor import extract_mime
from extraction_service.settings import ExtractionSettings
from rag_common.auth import mint_service_token
from rag_common.errors import InvalidDocumentError

FIXTURES = Path(__file__).parent / "fixtures"
SECRET = "unit-test-secret-that-is-at-least-32-chars-long"
ARTICLE = (FIXTURES / "article.html").read_text(encoding="utf-8")


def _auth() -> dict[str, str]:
    token = mint_service_token(
        secret=SECRET, issuer="contextual-rag", audience="rag-services", subject="n8n"
    )
    return {"Authorization": f"Bearer {token}"}


def _sample_email(html: bool = True) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = "Quarterly RAG review"
    msg["From"] = "Ana <ana@example.com>"
    msg["To"] = "team@example.com, Bob <bob@example.com>"
    msg["Cc"] = "cc@example.com"
    msg["Date"] = "Tue, 03 Sep 2024 10:15:00 +0000"
    msg["Message-ID"] = "<abc123@example.com>"
    msg.set_content("Plain text body about contextual retrieval and reranking.")
    if html:
        msg.add_alternative(ARTICLE, subtype="html")
    msg.add_attachment(b"col,value\n1,2\n", maintype="text", subtype="csv", filename="data.csv")
    msg.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename="r.pdf")
    return msg.as_bytes()


# --- HTML -------------------------------------------------------------------


def test_trafilatura_extracts_article_and_metadata() -> None:
    result = extract_html(ARTICLE, url="https://example.com/post")
    assert result.title == "Contextual Retrieval Explained"
    assert result.author == "Jane Doe"
    assert result.date == "2024-05-01"
    assert "prepending chunk-specific context" in result.content
    assert "Home | About" not in result.content  # nav boilerplate removed
    assert result.word_count > 40 and not result.is_thin


def test_thin_html_uses_fallback_and_is_flagged() -> None:
    result = extract_html("<html><body><div><b>tiny</b></div></body></html>")
    assert result.content == "tiny"
    assert result.is_thin


# --- MIME -------------------------------------------------------------------


def test_mime_extraction_prefers_html_body_and_lists_attachments() -> None:
    result = extract_mime(_sample_email())
    assert result.headers.subject == "Quarterly RAG review"
    assert result.headers.sender == "ana@example.com"
    assert result.headers.to == ["team@example.com", "bob@example.com"]
    assert result.headers.cc == ["cc@example.com"]
    assert result.headers.date == "2024-09-03T10:15:00+00:00"
    assert result.headers.message_id == "<abc123@example.com>"
    assert result.body_content_type == "text/html"
    assert "prepending chunk-specific context" in result.body.content
    names = [(a.filename, a.content_type, a.size_bytes) for a in result.attachments]
    assert names == [("data.csv", "text/csv", 14), ("r.pdf", "application/pdf", 13)]
    assert result.attachments[0].text == "col,value\n1,2\n"
    assert result.attachments[1].text is None


def test_mime_plain_body_and_invalid_payloads() -> None:
    result = extract_mime(_sample_email(html=False))
    assert result.body_content_type == "text/plain"
    assert result.body.content.startswith("Plain text body")
    assert result.body.title == "Quarterly RAG review"  # subject fills in the title
    with pytest.raises(InvalidDocumentError):
        extract_mime(b"   ")
    with pytest.raises(InvalidDocumentError):
        extract_mime(b"no headers here at all")


# --- API --------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, patch_redis, fake_redis_sync):
    """App with a fake Defuddle sidecar and a fake public web behind MockTransport."""
    sidecar_calls: list[httpx.Request] = []

    def sidecar(request: httpx.Request) -> httpx.Response:
        sidecar_calls.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": "Rendered content from Defuddle " * 10,
                "wordCount": 50,
                "title": "Defuddled",
                "finalUrl": payload.get("url"),
            },
        )

    def web(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=ARTICLE)

    import extraction_service.main as main_mod

    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        if kwargs.get("follow_redirects") is False:
            return real_client(transport=httpx.MockTransport(web), **kwargs)
        return real_client(transport=httpx.MockTransport(sidecar), **kwargs)

    monkeypatch.setattr(main_mod.httpx, "AsyncClient", fake_client)

    async def resolver(host: str, port: int):
        return [ipaddress.ip_address("93.184.216.34")]

    app = build_app(ExtractionSettings(max_payload_bytes=8192))
    app.state.sidecar_calls = sidecar_calls
    with TestClient(app) as client:
        # SafeFetcher binds the system resolver at construction; swap it on the instance.
        app.state.extraction.fetcher.resolver = resolver
        yield client, app, fake_redis_sync


def test_html_endpoint_requires_auth_and_extracts(api) -> None:
    client, _, _ = api
    body = {"html": ARTICLE, "url": "https://example.com/post", "document_id": "doc-html"}
    assert client.post("/v1/extract/html", json=body).status_code == 401
    r = client.post("/v1/extract/html", json=body, headers=_auth())
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["document_id"] == "doc-html"
    assert doc["engine"] == "trafilatura" and doc["source"] == "html"
    assert doc["title"] == "Contextual Retrieval Explained"


def test_auto_engine_escalates_thin_pages_to_defuddle(api) -> None:
    client, app, _ = api
    r = client.post(
        "/v1/extract/html",
        json={"html": "<html><body><p>tiny</p></body></html>", "engine": "auto"},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert r.json()["engine"] == "defuddle"
    assert "used Defuddle" in r.json()["warnings"][0]
    sidecar_request = next(c for c in app.state.sidecar_calls if c.url.path == "/parse")
    assert sidecar_request.headers["authorization"].startswith("Bearer ")


def test_url_endpoint_fetches_and_render_uses_sidecar(api) -> None:
    client, app, _ = api
    r = client.post("/v1/extract/url", json={"url": "https://public.example/post"}, headers=_auth())
    assert r.status_code == 200, r.text
    assert r.json()["metadata"]["final_url"] == "https://public.example/post"
    assert r.json()["engine"] == "trafilatura"
    rendered = client.post(
        "/v1/extract/url",
        json={"url": "https://public.example/app", "render": True},
        headers=_auth(),
    )
    assert rendered.status_code == 200
    assert rendered.json()["engine"] == "defuddle"
    assert [c.url.path for c in app.state.sidecar_calls if c.url.path == "/render"] == ["/render"]


def test_ssrf_block_and_oversize_are_dead_lettered(api) -> None:
    client, _, redis = api
    blocked = client.post(
        "/v1/extract/url",
        json={"url": "http://127.0.0.1/admin", "document_id": "doc-x"},
        headers=_auth(),
    )
    assert blocked.status_code == 403
    assert blocked.json()["error"]["type"] == "forbidden_target"
    big = client.post(
        "/v1/extract/html",
        json={"html": "<p>" + "x" * 20000 + "</p>", "document_id": "doc-y"},
        headers=_auth(),
    )
    assert big.status_code == 413
    entries = redis.xrange("rag.dlq")
    stages = [(f["document_id"], f["stage"], f["error_type"]) for _, f in entries]
    assert stages == [
        ("doc-x", "extraction", "ForbiddenTargetError"),
        ("doc-y", "extraction", "PayloadTooLargeError"),
    ]


def test_mime_endpoint_accepts_raw_rfc822(api) -> None:
    client, _, _ = api
    r = client.post(
        "/v1/extract/mime?document_id=mail-1",
        content=_sample_email(),
        headers={**_auth(), "Content-Type": "message/rfc822"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["document_id"] == "mail-1"
    assert data["headers"]["from"] == "ana@example.com"
    assert data["body"]["source"] == "mime"
    assert [a["filename"] for a in data["attachments"]] == ["data.csv", "r.pdf"]
    upload = client.post(
        "/v1/extract/mime",
        files={"file": ("m.eml", _sample_email(), "message/rfc822")},
        headers=_auth(),
    )
    assert upload.status_code == 200
