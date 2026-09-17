"""Standard HTML extraction with trafilatura (lxml + lxml_html_clean)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import trafilatura
from lxml import html as lxml_html

THIN_CONTENT_WORDS = 40  # below this, ``engine=auto`` escalates to Defuddle


@dataclass
class HtmlExtraction:
    content: str
    word_count: int
    title: str | None = None
    author: str | None = None
    date: str | None = None
    description: str | None = None
    site_name: str | None = None
    language: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_thin(self) -> bool:
        return self.word_count < THIN_CONTENT_WORDS


def _fallback_text(html: str) -> str:
    """Last resort when trafilatura finds no article body: visible text via lxml."""
    try:
        root = lxml_html.fromstring(html)
    except (ValueError, lxml_html.etree.ParserError):
        return ""
    for tag in root.iter("script", "style", "noscript", "template"):
        tag.drop_tree()
    text = root.text_content()
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def extract_html(
    html: str, *, url: str | None = None, favor_recall: bool = False
) -> HtmlExtraction:
    meta = cast(
        dict[str, Any],
        trafilatura.bare_extraction(
            html,
            url=url,
            with_metadata=True,
            as_dict=True,
            include_comments=False,
            include_tables=True,
            include_links=False,
            favor_recall=favor_recall,
        )
        or {},
    )
    content = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        with_metadata=False,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_recall=favor_recall,
    )
    warnings: list[str] = []
    if not content:
        content = _fallback_text(html)
        warnings.append("trafilatura found no main content; used visible-text fallback")
    content = content.strip()
    return HtmlExtraction(
        content=content,
        word_count=len(content.split()),
        title=meta.get("title") or None,
        author=meta.get("author") or None,
        date=meta.get("date") or None,
        description=meta.get("description") or None,
        site_name=meta.get("sitename") or None,
        language=meta.get("language") or None,
        warnings=warnings,
    )
