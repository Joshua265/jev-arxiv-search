from __future__ import annotations

import json
import re
from email.utils import parsedate_to_datetime
from typing import Any


def collapse_ws(s: str | None) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def split_categories(s: str | None) -> list[str]:
    if not s:
        return []
    return s.split()


def parse_created(s: str) -> str:
    try:
        dt = parsedate_to_datetime(s)
        return dt.isoformat()
    except Exception:
        return ""


def normalize_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    arxiv_id = raw.get("id")
    title = collapse_ws(raw.get("title"))
    abstract = collapse_ws(raw.get("abstract"))
    if not arxiv_id or not title or not abstract:
        return None

    categories = split_categories(raw.get("categories"))
    primary_category = categories[0] if categories else ""

    versions: list[dict[str, str]] = raw.get("versions") or []
    submitted_at = parse_created(versions[0]["created"]) if versions else ""
    updated_at = raw.get("update_date", "")

    authors_parsed = raw.get("authors_parsed")
    if authors_parsed:
        authors_json = json.dumps(authors_parsed, ensure_ascii=False)
    else:
        authors_json = json.dumps(raw.get("authors", ""), ensure_ascii=False)

    versions_json = json.dumps(versions, ensure_ascii=False)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "primary_category": primary_category,
        "categories": categories,
        "submitted_at": submitted_at,
        "updated_at": updated_at,
        "doi": raw.get("doi") or "",
        "journal_ref": raw.get("journal-ref") or "",
        "comments": raw.get("comments") or "",
        "license": raw.get("license") or "",
        "authors_json": authors_json,
        "versions_json": versions_json,
    }
