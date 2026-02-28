from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from src.ingest.base import BaseSource
from src.ingest.cache import write_raw_payload
from src.normalize.canonical_id import generate_scholarship_id

logger = logging.getLogger(__name__)

_BROWSE_URL = "https://scholarshipamerica.org/students/browse-scholarships/"
_SITE_HOST = "scholarshipamerica.org"

_TAG_PATTERN = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", flags=re.IGNORECASE | re.DOTALL)
_WS_PATTERN = re.compile(r"\s+")
_MONEY_PATTERN = re.compile(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
_ISO_DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_US_DATE_PATTERN = re.compile(r"\b(\d{1,2}/\d{1,2}/20\d{2})\b")
_LONG_DATE_PATTERN = re.compile(
    r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},\s+20\d{2})\b",
    flags=re.IGNORECASE,
)
_LINKED_JSON_PATTERN = re.compile(r"https?://[^\"'\s>]+wp-json[^\"'\s<]+", flags=re.IGNORECASE)
_FWP_JSON_PATTERN = re.compile(
    r"window\.FWP_JSON\s*=\s*(\{.*?\});\s*window\.FWP_HTTP",
    flags=re.IGNORECASE | re.DOTALL,
)
_LABEL_BLOCK_PATTERN = re.compile(
    r"(Sponsor|Provider|Organization|Eligibility|Deadline|Amount|Award|Education|Institution|State|Territory|Essay(?: Prompt)?|Essay Required)\s*:?\s*",
    re.IGNORECASE,
)
_NON_DETAIL_TITLE_HINTS = ("browse scholarships", "page not found", "not found", "access denied")

_US_STATES_AND_TERRITORIES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "Guam", "U.S. Virgin Islands", "Northern Mariana Islands",
    "American Samoa",
]


@dataclass(slots=True)
class FetchStats:
    listing_urls_processed: int = 0
    detail_urls_discovered: int = 0
    detail_urls_attempted: int = 0
    detail_urls_succeeded: int = 0
    detail_urls_failed: int = 0
    cached_files_written: int = 0
    resumed_detail_urls: int = 0
    caps_hit: list[str] = field(default_factory=list)
    parse_failures: list[dict[str, str]] = field(default_factory=list)

    def add_cap(self, reason: str) -> None:
        if reason not in self.caps_hit:
            self.caps_hit.append(reason)

    def add_parse_failure(self, url: str, reason: str) -> None:
        self.parse_failures.append({"url": url, "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_urls_processed": self.listing_urls_processed,
            "detail_urls_discovered": self.detail_urls_discovered,
            "detail_urls_attempted": self.detail_urls_attempted,
            "detail_urls_succeeded": self.detail_urls_succeeded,
            "detail_urls_failed": self.detail_urls_failed,
            "cached_files_written": self.cached_files_written,
            "resumed_detail_urls": self.resumed_detail_urls,
            "caps_hit": list(self.caps_hit),
            "parse_failures": list(self.parse_failures),
        }


@dataclass(slots=True)
class _DetailFetchResult:
    detail_url: str
    record: dict[str, Any] | None = None
    cache_path: Path | None = None
    resumed_from_cache: bool = False
    skipped: bool = False
    error_reason: str | None = None


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self._current_href = value
                self._text_parts = []
                break

    def handle_data(self, data: str) -> None:
        if self._current_href is not None and data:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = " ".join(part.strip() for part in self._text_parts if part.strip())
        self.links.append((self._current_href, text))
        self._current_href = None
        self._text_parts = []


class ScholarshipAmericaLiveSource(BaseSource):
    name = "scholarship_america"
    browse_url = _BROWSE_URL

    def fetch(self, http_client: Any):  # pragma: no cover - this source uses fetch_records
        raise NotImplementedError("Use fetch_records for ScholarshipAmericaLiveSource.")

    def parse(self, raw_content: bytes, *, fetched_at: datetime) -> list[dict[str, Any]]:  # pragma: no cover
        return []

    def fetch_records(
        self,
        http_client: Any,
        *,
        raw_root: Path,
        max_listing_pages: int = 3,
        max_detail_pages: int = 200,
        max_runtime_seconds: int = 600,
        concurrency: int = 4,
        resume: bool = False,
    ) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
        fetched_at = self.utcnow()
        cache_date_prefix = fetched_at.astimezone(UTC).strftime("%Y%m%d")
        deadline_monotonic = (
            time.monotonic() + max_runtime_seconds if max_runtime_seconds > 0 else None
        )
        cached_paths: list[Path] = []
        stats = FetchStats()

        detail_urls, json_urls = self._collect_listing_urls(
            http_client=http_client,
            raw_root=raw_root,
            fetched_at=fetched_at,
            max_listing_pages=max_listing_pages,
            deadline_monotonic=deadline_monotonic,
            cached_paths=cached_paths,
            stats=stats,
        )
        detail_urls.update(
            self._collect_json_detail_urls(
                http_client=http_client,
                raw_root=raw_root,
                fetched_at=fetched_at,
                json_urls=json_urls,
                deadline_monotonic=deadline_monotonic,
                cached_paths=cached_paths,
                stats=stats,
            )
        )

        normalized_details = sorted(
            {_normalize_url(url) for url in detail_urls if _looks_like_detail_url(url, "")}
        )
        stats.detail_urls_discovered = len(normalized_details)
        if len(normalized_details) > max_detail_pages:
            normalized_details = normalized_details[:max_detail_pages]
            stats.add_cap("max_detail_pages")
            logger.info("Detail page cap reached (%d); stopping detail discovery.", max_detail_pages)

        records = self._fetch_detail_records(
            http_client=http_client,
            raw_root=raw_root,
            fetched_at=fetched_at,
            detail_urls=normalized_details,
            cache_date_prefix=cache_date_prefix,
            deadline_monotonic=deadline_monotonic,
            concurrency=concurrency,
            resume=resume,
            cached_paths=cached_paths,
            stats=stats,
        )
        records.sort(
            key=lambda row: (
                row.get("source_url") or "",
                row.get("source_id") or "",
                row.get("scholarship_id") or "",
            )
        )
        return records, cached_paths, stats.to_dict()

    def _collect_listing_urls(
        self,
        *,
        http_client: Any,
        raw_root: Path,
        fetched_at: datetime,
        max_listing_pages: int,
        deadline_monotonic: float | None,
        cached_paths: list[Path],
        stats: FetchStats,
    ) -> tuple[set[str], set[str]]:
        detail_urls: set[str] = set()
        json_urls: set[str] = set()
        queued_urls = {_normalize_url(self.browse_url)}
        queue: list[tuple[str, str]] = [(self.browse_url, "browse-scholarships")]
        saw_browse_root = False

        while queue:
            if self._runtime_expired(deadline_monotonic):
                stats.add_cap("max_runtime_seconds")
                logger.info("Runtime cap reached while collecting listing pages.")
                break
            if stats.listing_urls_processed >= max_listing_pages:
                stats.add_cap("max_listing_pages")
                logger.info("Listing page cap reached (%d); stopping listing traversal.", max_listing_pages)
                break

            listing_url, slug = queue.pop(0)
            try:
                listing_html = http_client.get_text(listing_url)
            except Exception as exc:
                reason = f"listing_fetch_failed:{type(exc).__name__}"
                stats.add_parse_failure(listing_url, reason)
                logger.warning("Failed listing fetch: %s (%s)", listing_url, reason, exc_info=True)
                continue

            cached_path = self._cache_html(
                html=listing_html,
                raw_root=raw_root,
                fetched_at=fetched_at,
                slug=slug,
            )
            cached_paths.append(cached_path)
            stats.cached_files_written += 1
            stats.listing_urls_processed += 1

            page_detail_urls, category_urls, page_json_urls = self.parse_listing_html(
                listing_html,
                base_url=listing_url,
            )
            detail_urls.update(page_detail_urls)
            json_urls.update(page_json_urls)

            normalized_listing_url = _normalize_url(listing_url)
            if not saw_browse_root and normalized_listing_url == _normalize_url(self.browse_url):
                saw_browse_root = True
                total_pages = _extract_total_pages_from_listing_html(listing_html)
                for page_number in range(2, max(total_pages, 1) + 1):
                    paged_url = _normalize_url(f"{self.browse_url}?_paged={page_number}")
                    if paged_url in queued_urls:
                        continue
                    queued_urls.add(paged_url)
                    queue.append((paged_url, f"browse-scholarships-page-{page_number}"))

            for category_url in sorted(category_urls):
                normalized = _normalize_url(category_url)
                if normalized in queued_urls:
                    continue
                queued_urls.add(normalized)
                queue.append((normalized, f"category-{_slug_from_url(normalized)}"))

        return detail_urls, json_urls

    def _collect_json_detail_urls(
        self,
        *,
        http_client: Any,
        raw_root: Path,
        fetched_at: datetime,
        json_urls: set[str],
        deadline_monotonic: float | None,
        cached_paths: list[Path],
        stats: FetchStats,
    ) -> set[str]:
        discovered: set[str] = set()
        for json_url in sorted(json_urls):
            if self._runtime_expired(deadline_monotonic):
                stats.add_cap("max_runtime_seconds")
                logger.info("Runtime cap reached while fetching discovered JSON URLs.")
                break
            try:
                json_payload = http_client.get_json(json_url)
            except Exception as exc:
                reason = f"json_fetch_failed:{type(exc).__name__}"
                stats.add_parse_failure(json_url, reason)
                logger.warning("Failed JSON discovery fetch: %s (%s)", json_url, reason, exc_info=True)
                continue

            payload_text = json.dumps(json_payload, sort_keys=True)
            cached_path = write_raw_payload(
                source_name=self.name,
                payload=payload_text.encode("utf-8"),
                extension="json",
                raw_root=raw_root,
                timestamp=fetched_at,
                slug=f"discovered-json-{_slug_from_url(json_url)}",
            )
            cached_paths.append(cached_path)
            stats.cached_files_written += 1
            discovered.update(self._extract_urls_from_json(json_payload))
        return discovered

    def _fetch_detail_records(
        self,
        *,
        http_client: Any,
        raw_root: Path,
        fetched_at: datetime,
        detail_urls: list[str],
        cache_date_prefix: str,
        deadline_monotonic: float | None,
        concurrency: int,
        resume: bool,
        cached_paths: list[Path],
        stats: FetchStats,
    ) -> list[dict[str, Any]]:
        if not detail_urls:
            return []

        records: list[dict[str, Any]] = []
        max_workers = max(1, concurrency)
        url_iter = iter(detail_urls)
        pending: dict[concurrent.futures.Future[_DetailFetchResult], str] = {}
        stop_submitting = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                while not stop_submitting and len(pending) < max_workers:
                    if self._runtime_expired(deadline_monotonic):
                        stats.add_cap("max_runtime_seconds")
                        logger.info("Runtime cap reached before submitting all detail fetches.")
                        stop_submitting = True
                        break
                    try:
                        detail_url = next(url_iter)
                    except StopIteration:
                        stop_submitting = True
                        break
                    stats.detail_urls_attempted += 1
                    future = executor.submit(
                        self._fetch_single_detail,
                        http_client=http_client,
                        raw_root=raw_root,
                        fetched_at=fetched_at,
                        detail_url=detail_url,
                        cache_date_prefix=cache_date_prefix,
                        resume=resume,
                    )
                    pending[future] = detail_url

                if not pending:
                    break

                done, _ = concurrent.futures.wait(
                    pending,
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    continue

                for future in sorted(done, key=lambda item: pending[item]):
                    detail_url = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        stats.detail_urls_failed += 1
                        reason = f"detail_worker_failed:{type(exc).__name__}"
                        stats.add_parse_failure(detail_url, reason)
                        logger.warning("Failed detail worker: %s (%s)", detail_url, reason, exc_info=True)
                        continue

                    if result.cache_path is not None:
                        cached_paths.append(result.cache_path)
                        if not result.resumed_from_cache:
                            stats.cached_files_written += 1
                    if result.resumed_from_cache:
                        stats.resumed_detail_urls += 1
                    if result.record is not None:
                        stats.detail_urls_succeeded += 1
                        records.append(result.record)
                        continue
                    if result.skipped:
                        continue
                    stats.detail_urls_failed += 1
                    if result.error_reason:
                        stats.add_parse_failure(result.detail_url, result.error_reason)
                        logger.warning(
                            "Skipping detail page: %s (%s)",
                            result.detail_url,
                            result.error_reason,
                        )

        return records

    def _fetch_single_detail(
        self,
        *,
        http_client: Any,
        raw_root: Path,
        fetched_at: datetime,
        detail_url: str,
        cache_date_prefix: str,
        resume: bool,
    ) -> _DetailFetchResult:
        cache_slug = _detail_cache_slug(detail_url)
        cache_path: Path | None = None
        detail_html: str
        resumed_from_cache = False

        if resume:
            cache_path = self._find_resume_cache(
                raw_root=raw_root,
                cache_date_prefix=cache_date_prefix,
                slug=cache_slug,
            )
            if cache_path is not None:
                try:
                    detail_html = cache_path.read_text(encoding="utf-8")
                    resumed_from_cache = True
                except OSError as exc:
                    return _DetailFetchResult(
                        detail_url=detail_url,
                        cache_path=cache_path,
                        error_reason=f"resume_cache_read_failed:{type(exc).__name__}",
                    )
            else:
                detail_html = ""
        else:
            detail_html = ""

        if not detail_html:
            try:
                detail_html = http_client.get_text(detail_url)
            except Exception as exc:
                return _DetailFetchResult(
                    detail_url=detail_url,
                    error_reason=f"detail_fetch_failed:{type(exc).__name__}",
                )
            cache_path = self._cache_html(
                html=detail_html,
                raw_root=raw_root,
                fetched_at=fetched_at,
                slug=cache_slug,
            )

        try:
            record = self.parse_detail_html(
                detail_html,
                detail_url=detail_url,
                fetched_at=fetched_at,
            )
        except Exception as exc:
            return _DetailFetchResult(
                detail_url=detail_url,
                cache_path=cache_path,
                resumed_from_cache=resumed_from_cache,
                error_reason=f"detail_parse_failed:{type(exc).__name__}",
            )

        if record is None:
            return _DetailFetchResult(
                detail_url=detail_url,
                cache_path=cache_path,
                resumed_from_cache=resumed_from_cache,
                error_reason="invalid_or_non_scholarship_page",
            )

        return _DetailFetchResult(
            detail_url=detail_url,
            record=record,
            cache_path=cache_path,
            resumed_from_cache=resumed_from_cache,
        )

    def _runtime_expired(self, deadline_monotonic: float | None) -> bool:
        return deadline_monotonic is not None and time.monotonic() >= deadline_monotonic

    def _find_resume_cache(
        self,
        *,
        raw_root: Path,
        cache_date_prefix: str,
        slug: str,
    ) -> Path | None:
        target_dir = raw_root / self.name
        if not target_dir.exists():
            return None
        matches = sorted(target_dir.glob(f"{cache_date_prefix}*_{slug}.html"))
        return matches[-1] if matches else None

    def parse_listing_html(
        self,
        html: str,
        *,
        base_url: str,
    ) -> tuple[set[str], set[str], set[str]]:
        collector = _AnchorCollector()
        collector.feed(html)

        detail_urls: set[str] = set()
        category_urls: set[str] = set()
        for href, text in collector.links:
            absolute = _normalize_url(urljoin(base_url, href))
            if not _is_http_url(absolute):
                continue
            if _looks_like_detail_url(absolute, text):
                detail_urls.add(absolute)
            elif _looks_like_category_url(absolute):
                category_urls.add(absolute)

        json_urls = {_normalize_url(url) for url in _LINKED_JSON_PATTERN.findall(html) if _is_http_url(url)}
        return detail_urls, category_urls, json_urls

    def parse_detail_html(
        self,
        html: str,
        *,
        detail_url: str,
        fetched_at: datetime,
    ) -> dict[str, Any] | None:
        normalized_url = _normalize_url(detail_url)
        if not _looks_like_detail_url(normalized_url, ""):
            return None

        text = _to_text(html)
        if not text:
            return None

        title = _extract_title(html)
        if any(hint in title.lower() for hint in _NON_DETAIL_TITLE_HINTS):
            return None

        sponsor = _extract_field_value(text, ["sponsor", "provider", "organization"]) or "Scholarship America"

        description = _extract_description(html)
        eligibility_text = _extract_field_value(text, ["eligibility", "requirements"]) or _extract_eligibility_text(text)
        deadline = _extract_deadline(text)
        amount_min, amount_max = _extract_amount_range(text)
        education_level = _extract_field_value(text, ["education", "institution", "school"]) or None

        states_allowed = _extract_states(text)
        essay_required, essay_prompt = _extract_essay_fields(text)
        source_id = _slug_from_url(normalized_url)
        resolved_title = title or source_id.replace("-", " ").strip().title() or normalized_url

        scholarship_id = generate_scholarship_id(
            title=resolved_title,
            sponsor=sponsor,
            amount_min=amount_min,
            amount_max=amount_max,
            deadline=deadline,
            source_url=normalized_url,
        )

        return {
            "scholarship_id": scholarship_id,
            "source": self.name,
            "source_id": source_id,
            "source_url": normalized_url,
            "title": resolved_title,
            "sponsor": sponsor,
            "description": description or None,
            "eligibility_text": eligibility_text or None,
            "deadline": deadline,
            "amount_min": amount_min,
            "amount_max": amount_max,
            "is_recurring": None,
            "states_allowed": states_allowed,
            "majors_allowed": None,
            "min_gpa": None,
            "citizenship": None,
            "education_level": education_level,
            "essay_required": essay_required,
            "essay_prompt": essay_prompt,
            "keywords": [source_id] if source_id else None,
            "first_seen_at": fetched_at,
            "last_seen_at": fetched_at,
        }

    def _extract_urls_from_json(self, payload: Any) -> set[str]:
        urls: set[str] = set()

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    _walk(value)
                return
            if isinstance(node, list):
                for value in node:
                    _walk(value)
                return
            if isinstance(node, str):
                normalized = _normalize_url(node)
                if _looks_like_detail_url(normalized, ""):
                    urls.add(normalized)

        _walk(payload)
        return urls

    def _cache_html(self, *, html: str, raw_root: Path, fetched_at: datetime, slug: str) -> Path:
        return write_raw_payload(
            source_name=self.name,
            payload=html.encode("utf-8"),
            extension="html",
            raw_root=raw_root,
            timestamp=fetched_at,
            slug=slug,
        )


def _detail_cache_slug(url: str) -> str:
    normalized = _normalize_url(url)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"detail-{_slug_from_url(normalized)}-{digest}"


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    scheme = "https" if parsed.netloc else parsed.scheme
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = sorted((key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=False))
    query = urlencode(query_items)
    return urlunparse((scheme, netloc, path, "", query, ""))


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        return "home"
    return re.sub(r"[^a-z0-9]+", "-", path_parts[-1].lower()).strip("-") or "detail"


def _is_internal(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == _SITE_HOST or host.endswith(f".{_SITE_HOST}")


def _looks_like_category_url(url: str) -> bool:
    if not _is_internal(url):
        return False
    path = urlparse(url).path.lower()
    return path.startswith("/students/") and "scholarship" in path


def _looks_like_detail_url(url: str, anchor_text: str) -> bool:
    if not _is_internal(url):
        return False

    path = urlparse(url).path.lower()
    if any(path.endswith(suffix) for suffix in (".jpg", ".png", ".pdf", ".zip")):
        return False
    if "browse-scholarships" in path:
        return False
    if path.startswith("/students/") and "scholarship" in path:
        return False
    if "/scholarship/" in path or "/scholarships/" in path:
        return True

    lowered_text = anchor_text.strip().lower()
    if lowered_text in {"apply", "learn more", "view scholarship", "read more"}:
        return True
    if "scholarship" in path and len(path.split("/")) > 3:
        return True
    return False


def _to_text(html: str) -> str:
    without_script = _SCRIPT_STYLE_PATTERN.sub(" ", html)
    cleaned = _TAG_PATTERN.sub(" ", without_script)
    return _WS_PATTERN.sub(" ", unescape(cleaned)).strip()


def _extract_title(html: str) -> str:
    og_match = re.search(
        r"<meta[^>]*property=['\"]og:title['\"][^>]*content=['\"](.*?)['\"][^>]*>",
        html,
        flags=re.IGNORECASE,
    )
    if og_match:
        return _WS_PATTERN.sub(" ", unescape(og_match.group(1))).strip()

    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
    if h1_match:
        return _WS_PATTERN.sub(" ", _to_text(h1_match.group(1))).strip()

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        title = _WS_PATTERN.sub(" ", _to_text(title_match.group(1))).strip()
        return title.replace("| Scholarship America", "").strip()

    return ""


def _extract_description(html: str) -> str:
    meta_match = re.search(
        r"<meta[^>]*name=['\"]description['\"][^>]*content=['\"](.*?)['\"][^>]*>",
        html,
        flags=re.IGNORECASE,
    )
    if meta_match:
        candidate = _WS_PATTERN.sub(" ", unescape(meta_match.group(1))).strip()
        if candidate:
            return candidate

    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    candidates = [_WS_PATTERN.sub(" ", _to_text(chunk)).strip() for chunk in paragraphs]
    candidates = [entry for entry in candidates if len(entry) >= 30]
    return candidates[0] if candidates else ""


def _extract_field_value(text: str, field_names: list[str]) -> str:
    lowered = text.lower()
    for field in field_names:
        key = f"{field.lower()}:"
        start = lowered.find(key)
        if start == -1:
            continue
        raw = text[start + len(key) :]
        split = _LABEL_BLOCK_PATTERN.split(raw, maxsplit=1)
        candidate = split[0].strip(" .;:\n\t")
        if candidate:
            return candidate
    return ""


def _extract_eligibility_text(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    picks = [line.strip() for line in sentences if "eligib" in line.lower() or "require" in line.lower()]
    picks = [line for line in picks if len(line) >= 30]
    if not picks:
        return ""
    return " ".join(picks[:3])


def _extract_deadline(text: str) -> str | None:
    iso = _ISO_DATE_PATTERN.search(text)
    if iso:
        return iso.group(1)

    us_date = _US_DATE_PATTERN.search(text)
    if us_date:
        try:
            return datetime.strptime(us_date.group(1), "%m/%d/%Y").date().isoformat()
        except ValueError:
            pass

    long_date = _LONG_DATE_PATTERN.search(text)
    if long_date:
        raw = long_date.group(1)
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _extract_amount_range(text: str) -> tuple[float | None, float | None]:
    matches = [float(chunk.replace(",", "")) for chunk in _MONEY_PATTERN.findall(text)]
    if not matches:
        return None, None
    return min(matches), max(matches)


def _extract_states(text: str) -> list[str] | None:
    lowered = text.lower()
    found = [name for name in _US_STATES_AND_TERRITORIES if name.lower() in lowered]
    if not found:
        return None
    return sorted(set(found))


def _extract_essay_fields(text: str) -> tuple[bool | None, str | None]:
    lowered = text.lower()
    essay_required: bool | None = None
    essay_prompt: str | None = None

    required_match = re.search(r"essay required\s*:\s*([^.;\n]+)", lowered)
    if required_match:
        answer = required_match.group(1).strip()
        if answer.startswith("yes"):
            essay_required = True
        elif answer.startswith("no"):
            essay_required = False

    prompt_match = re.search(r"essay prompt\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if prompt_match:
        essay_prompt = prompt_match.group(1).strip(" .;:") or None

    if essay_required is None:
        if "essay required" in lowered and "not required" in lowered:
            essay_required = False
        elif "essay" in lowered:
            essay_required = True

    return essay_required, essay_prompt


def _extract_total_pages_from_listing_html(html: str) -> int:
    match = _FWP_JSON_PATTERN.search(html)
    if not match:
        return 1
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return 1

    pager = (
        payload.get("preload_data", {})
        .get("settings", {})
        .get("pager", {})
    )
    raw_total_pages = pager.get("total_pages")
    try:
        total_pages = int(raw_total_pages)
    except (TypeError, ValueError):
        return 1
    return max(total_pages, 1)
