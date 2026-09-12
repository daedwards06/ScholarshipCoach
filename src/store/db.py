"""The family database: one SQLite file, no server, standard library only.

The profile is a JSON file (:mod:`src.profile.store`) because it is one
document per student.  A tracker, essays, recommenders and outcomes are
relational and grow without bound, so they live here instead, in
``data/private/coach.db`` -- under the same git-ignored directory as the
profiles, because every row of it is personal data.

Schema changes are numbered SQL files under ``migrations/``.  Opening a
connection applies whichever ones the file has not seen yet and records them
in ``schema_migrations``, so a database created by an older build catches up
on the next open and re-opening an up-to-date one is a no-op.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PRIVATE_DIR = ROOT_DIR / "data" / "private"
DEFAULT_DB_PATH = PRIVATE_DIR / "coach.db"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

MIGRATIONS_TABLE = "schema_migrations"

_MIGRATION_NAME_RE = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")


class MigrationError(RuntimeError):
    """A migration file is misnamed or could not be applied."""


def utc_now() -> str:
    """Return the current UTC time as a sortable ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def migration_files(migrations_dir: Path | None = None) -> list[Path]:
    """Return the migration files in apply order, rejecting misnamed ones."""
    directory = migrations_dir or MIGRATIONS_DIR
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("*.sql"), key=lambda path: path.name)
    for path in paths:
        if not _MIGRATION_NAME_RE.match(path.name):
            raise MigrationError(
                f"Migration {path.name!r} must be named NNNN_snake_case.sql "
                "(the number sets apply order)."
            )
    return paths


def applied_migrations(conn: sqlite3.Connection) -> list[str]:
    """Return the migration names this database has already applied."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} ("
        "    name TEXT PRIMARY KEY,"
        "    applied_at TEXT NOT NULL"
        ")"
    )
    rows = conn.execute(f"SELECT name FROM {MIGRATIONS_TABLE} ORDER BY name").fetchall()
    return [str(row[0]) for row in rows]


def apply_migrations(
    conn: sqlite3.Connection, migrations_dir: Path | None = None
) -> list[str]:
    """Apply every unapplied migration and return the names newly applied."""
    already = set(applied_migrations(conn))
    newly: list[str] = []
    for path in migration_files(migrations_dir):
        if path.name in already:
            continue
        try:
            conn.executescript(path.read_text(encoding="utf-8-sig"))
        except sqlite3.Error as exc:
            conn.rollback()
            raise MigrationError(f"Migration {path.name!r} failed: {exc}") from exc
        conn.execute(
            f"INSERT INTO {MIGRATIONS_TABLE} (name, applied_at) VALUES (?, ?)",
            (path.name, utc_now()),
        )
        conn.commit()
        newly.append(path.name)
    return newly


def connect(
    db_path: Path | str | None = None, migrations_dir: Path | None = None
) -> sqlite3.Connection:
    """Open ``db_path`` (creating it if needed) with migrations applied."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Off by default in SQLite, and every cascade in the schema depends on it.
    conn.execute("PRAGMA foreign_keys = ON")
    apply_migrations(conn, migrations_dir)
    return conn


@contextmanager
def open_db(
    db_path: Path | str | None = None, migrations_dir: Path | None = None
) -> Iterator[sqlite3.Connection]:
    """Open a connection for the duration of a ``with`` block, then close it."""
    conn = connect(db_path, migrations_dir)
    try:
        yield conn
    finally:
        conn.close()
