"""Multipart MIME (RFC 822/2045) extraction with the standard library.

``email.policy.default`` gives the modern ``EmailMessage`` API: decoded
headers, ``get_body()`` with a preference list and ``iter_attachments()``.
HTML bodies are cleaned with trafilatura in recall mode (emails are short and
rarely have an <article>).
"""

from __future__ import annotations

import email
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage, MIMEPart
from email.utils import getaddresses, parsedate_to_datetime

from extraction_service.html_extractor import HtmlExtraction, extract_html
from extraction_service.schemas import Attachment, EmailHeaders
from rag_common.errors import InvalidDocumentError

_TEXT_ATTACHMENT_LIMIT = 1024 * 1024


@dataclass
class MimeExtraction:
    headers: EmailHeaders
    body: HtmlExtraction
    body_content_type: str
    attachments: list[Attachment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _addresses(msg: EmailMessage, header: str) -> list[str]:
    values = msg.get_all(header, [])
    return [addr for _, addr in getaddresses([str(v) for v in values]) if addr]


def _date(msg: EmailMessage) -> str | None:
    raw = msg.get("date")
    if not raw:
        return None
    try:
        return parsedate_to_datetime(str(raw)).isoformat()
    except (TypeError, ValueError):
        return str(raw)


def _raw_bytes(part: MIMEPart) -> bytes:
    payload = part.get_payload(decode=True)
    return payload if isinstance(payload, bytes) else b""


def _attachment(part: MIMEPart) -> Attachment:
    payload = _raw_bytes(part)
    content_type = part.get_content_type()
    text: str | None = None
    if content_type.startswith("text/") and len(payload) <= _TEXT_ATTACHMENT_LIMIT:
        try:
            text = part.get_content()
        except (LookupError, UnicodeDecodeError):
            text = payload.decode("utf-8", errors="replace")
    return Attachment(
        filename=part.get_filename(),
        content_type=content_type,
        size_bytes=len(payload),
        content_id=(part.get("content-id") or None),
        text=text,
    )


def extract_mime(raw: bytes, *, max_attachments: int = 50) -> MimeExtraction:
    if not raw.strip():
        raise InvalidDocumentError("empty MIME payload")
    msg = email.message_from_bytes(raw, policy=policy.default)
    if not isinstance(msg, EmailMessage):  # policy.default guarantees this; defensive
        raise InvalidDocumentError("could not parse message")
    if not msg.keys():
        raise InvalidDocumentError("payload has no RFC 822 headers")

    senders = _addresses(msg, "from")
    headers = EmailHeaders.model_validate(
        {
            "subject": msg.get("subject"),
            "from": senders[0] if senders else None,
            "to": _addresses(msg, "to"),
            "cc": _addresses(msg, "cc"),
            "date": _date(msg),
            "message_id": msg.get("message-id"),
        }
    )

    warnings: list[str] = []
    body_part = msg.get_body(preferencelist=("html", "plain"))
    if body_part is None:
        warnings.append("message has no text body")
        body = HtmlExtraction(content="", word_count=0)
        body_ct = "none"
    else:
        body_ct = body_part.get_content_type()
        try:
            text = body_part.get_content()
        except (LookupError, UnicodeDecodeError):
            text = _raw_bytes(body_part).decode("utf-8", errors="replace")
            warnings.append("body charset unknown; decoded as UTF-8 with replacement")
        if body_ct == "text/html":
            body = extract_html(text, favor_recall=True)
        else:
            cleaned = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines())
            body = HtmlExtraction(content=cleaned.strip(), word_count=len(cleaned.split()))
    if headers.subject and not body.title:
        body.title = headers.subject

    attachments: list[Attachment] = []
    for index, part in enumerate(msg.iter_attachments()):
        if index >= max_attachments:
            warnings.append(f"attachment list truncated at {max_attachments}")
            break
        attachments.append(_attachment(part))

    return MimeExtraction(
        headers=headers,
        body=body,
        body_content_type=body_ct,
        attachments=attachments,
        warnings=warnings + body.warnings,
    )
