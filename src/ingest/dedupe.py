"""Cross-source duplicate matching: one row per award, the most trusted row winning.

Ingest uses it to collapse a snapshot; discovery uses it to tell whether a feed
row is an award the curated catalog already carries.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd

from src.ingest.registry import register_sources
from src.io.snapshotting import SUPERSEDED_IDS_COLUMN
from src.text_utils import normalize_text, normalize_title_for_match

logger = logging.getLogger(__name__)

# Highest trust wins a cross-source duplicate; anything else ranks below these.
TRUST_PRECEDENCE = ("verified_local", "structured_feed", "aggregator", "unverified")
_TRUST_RANK = {value: index for index, value in enumerate(TRUST_PRECEDENCE)}
_UNKNOWN_TRUST_RANK = len(TRUST_PRECEDENCE)


def _coerce_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    return None


def normalize_url_for_dedupe(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = sorted((key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=False))
    query = urlencode(query_items)
    return urlunparse((scheme, netloc, path, "", query, ""))


def url_match_key(value: Any) -> str:
    """Return ``host + path`` for cross-source matching (``""`` when unusable).

    A bare host with no path is not an award identity — several awards can share
    a sponsor's home page — so it yields no key.
    """
    normalized = normalize_url_for_dedupe(value)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    if not parsed.netloc or parsed.path in ("", "/"):
        return ""
    return f"{parsed.netloc}{parsed.path}"


def collapse_cross_source_duplicates(
    df: pd.DataFrame, *, source_order: Sequence[str] | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one row per award across sources, the most trusted row winning.

    Two rows are the same award when they share a normalized URL (host + path)
    that no single source claims twice, a normalized ``(title, sponsor)`` pair,
    or when one record names the other's title in ``aliases``.  The URL carries
    that extra condition because sibling awards are often published on one page.
    The winner ranks first by :data:`TRUST_PRECEDENCE`,
    then by ``register_sources`` order, then by ``scholarship_id``, so the
    outcome never depends on the order the sources happened to run in.
    """
    order = (
        list(source_order)
        if source_order is not None
        else [source.name for source in register_sources()]
    )
    source_rank = {name: index for index, name in enumerate(order)}

    parent = {index: index for index in df.index}

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    rows_by_url: dict[str, list[int]] = {}
    title_groups: dict[tuple[str, str], int] = {}
    rows_by_title: dict[str, list[int]] = {}
    alias_claims: list[tuple[int, str]] = []
    has_aliases = "aliases" in df.columns

    for index in df.index:
        url_key = url_match_key(df.at[index, "source_url"])
        if url_key:
            rows_by_url.setdefault(url_key, []).append(index)

        title_key = normalize_title_for_match(df.at[index, "title"])
        if title_key:
            sponsor_key = normalize_text(df.at[index, "sponsor"])
            if (title_key, sponsor_key) in title_groups:
                union(title_groups[(title_key, sponsor_key)], index)
            else:
                title_groups[(title_key, sponsor_key)] = index
            rows_by_title.setdefault(title_key, []).append(index)

        if has_aliases:
            for alias in _coerce_list(df.at[index, "aliases"]) or []:
                alias_key = normalize_title_for_match(alias)
                if alias_key:
                    alias_claims.append((index, alias_key))

    # A URL identifies an award only while it maps to at most one row per
    # source.  A sponsor often publishes several named awards on one page --
    # AFCEA's STEM Majors page carries three, with different amounts, GPA floors
    # and membership rules -- and a source offering two rows for one URL is
    # saying exactly that, so the page's URL proves nothing about identity and
    # matching falls back to title, sponsor and aliases.
    for members in rows_by_url.values():
        sources = [str(df.at[index, "source"]) for index in members]
        if len(set(sources)) != len(sources):
            continue
        for other in members[1:]:
            union(members[0], other)

    for index, alias_key in alias_claims:
        for other in rows_by_title.get(alias_key, ()):
            if other != index:
                union(index, other)

    groups: dict[int, list[int]] = {}
    for index in df.index:
        groups.setdefault(find(index), []).append(index)

    def precedence(index: int) -> tuple[int, int, str]:
        trust = normalize_text(df.at[index, "trust"])
        return (
            _TRUST_RANK.get(trust, _UNKNOWN_TRUST_RANK),
            source_rank.get(str(df.at[index, "source"]), len(source_rank)),
            str(df.at[index, "scholarship_id"]),
        )

    superseded_ids: dict[int, list[str]] = {}
    superseded_counts: dict[str, int] = {}
    losing_rows: list[int] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        winner, *losers = sorted(members, key=precedence)
        superseded_ids[winner] = sorted(str(df.at[index, "scholarship_id"]) for index in losers)
        for index in losers:
            source_name = str(df.at[index, "source"])
            superseded_counts[source_name] = superseded_counts.get(source_name, 0) + 1
        losing_rows.extend(losers)
        logger.info(
            "Record %s from source %s superseded %d duplicate row(s): %s",
            df.at[winner, "scholarship_id"],
            df.at[winner, "source"],
            len(losers),
            ", ".join(superseded_ids[winner]),
        )

    prior_column = SUPERSEDED_IDS_COLUMN if SUPERSEDED_IDS_COLUMN in df.columns else None
    result = df.drop(index=losing_rows)
    result[SUPERSEDED_IDS_COLUMN] = [
        sorted(
            {
                *(_coerce_list(df.at[index, prior_column]) or [] if prior_column else []),
                *superseded_ids.get(index, []),
            }
        )
        for index in result.index
    ]
    return result.reset_index(drop=True), dict(sorted(superseded_counts.items()))
