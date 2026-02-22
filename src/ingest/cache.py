from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path


_SLUG_SAFE_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = _SLUG_SAFE_PATTERN.sub("-", lowered).strip("-")
    return slug or "payload"


def write_raw_payload(
    *,
    source_name: str,
    payload: bytes,
    extension: str,
    raw_root: Path,
    timestamp: datetime | None = None,
    slug: str | None = None,
) -> Path:
    resolved_ts = timestamp or datetime.now(tz=UTC)
    safe_extension = extension.lstrip(".")
    stamp = resolved_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")

    target_dir = raw_root / source_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if slug:
        filename = f"{stamp}_{_slugify(slug)}.{safe_extension}"
    else:
        filename = f"{stamp}.{safe_extension}"

    output_path = target_dir / filename
    output_path.write_bytes(payload)
    return output_path
