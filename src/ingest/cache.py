from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def write_raw_payload(
    *,
    source_name: str,
    payload: bytes,
    extension: str,
    raw_root: Path,
    timestamp: datetime | None = None,
) -> Path:
    resolved_ts = timestamp or datetime.now(tz=UTC)
    safe_extension = extension.lstrip(".")
    stamp = resolved_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")

    target_dir = raw_root / source_name
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{stamp}.{safe_extension}"
    output_path.write_bytes(payload)
    return output_path
