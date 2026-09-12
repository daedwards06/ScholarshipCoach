"""Work the catalog confirm queue: list, show, confirm, and reject proposals.

Everything automated proposes; a person confirms.  This is that person's tool.

    python scripts/catalog_inbox.py list
    python scripts/catalog_inbox.py show prefill-nc-tech-foundation
    python scripts/catalog_inbox.py confirm prefill-nc-tech-foundation --set trust=verified_local
    python scripts/catalog_inbox.py reject feed-some-award --reason "duplicate of an existing record"

``confirm`` writes to ``data/catalog/records/`` only if the merged record passes
the catalog schema; otherwise it reports the violations and leaves the proposal
in the queue.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.catalog.inbox import (
    INBOX_DIR,
    Proposal,
    ProposalError,
    confirm,
    list_proposals,
    load_proposal,
    reject,
)
from src.normalize.catalog_schema import RECORDS_DIR


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Work the curated catalog confirm queue.")
    parser.add_argument("--inbox-dir", type=Path, default=INBOX_DIR)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List every pending proposal.")

    show = subparsers.add_parser("show", help="Print one proposal as JSON.")
    show.add_argument("proposal_id")

    confirm_parser = subparsers.add_parser(
        "confirm", help="Validate a proposal and write it into the catalog."
    )
    confirm_parser.add_argument("proposal_id")
    confirm_parser.add_argument(
        "--set",
        dest="edits",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Override a field. VALUE is parsed as JSON when possible, else kept as text.",
    )
    confirm_parser.add_argument(
        "--edits", dest="edits_file", type=Path, help="JSON object of field overrides."
    )

    reject_parser = subparsers.add_parser("reject", help="Archive a proposal with a reason.")
    reject_parser.add_argument("proposal_id")
    reject_parser.add_argument("--reason", required=True)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        args.command = "help"
    return args


def _parse_edit(assignment: str) -> tuple[str, Any]:
    field, separator, raw = assignment.partition("=")
    if not separator or not field.strip():
        raise ProposalError(f"--set expects FIELD=VALUE, got {assignment!r}")
    try:
        return field.strip(), json.loads(raw)
    except json.JSONDecodeError:
        return field.strip(), raw


def _collect_edits(args: argparse.Namespace) -> dict[str, Any]:
    edits: dict[str, Any] = {}
    if args.edits_file is not None:
        try:
            payload = json.loads(args.edits_file.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise ProposalError(f"{args.edits_file}: unreadable edits file ({exc})") from exc
        if not isinstance(payload, dict):
            raise ProposalError(f"{args.edits_file}: expected a JSON object of field overrides")
        edits.update(payload)

    for assignment in args.edits:
        field, value = _parse_edit(assignment)
        edits[field] = value
    return edits


def _summarize(proposal: Proposal) -> str:
    change = f"{len(proposal.diff)} field(s) changed" if proposal.diff else "new record"
    title = proposal.title or proposal.catalog_id or "<untitled>"
    return f"  {proposal.proposal_id:<44} {proposal.kind:<9} {proposal.created_on:<12} {title} ({change})"


def _run_list(args: argparse.Namespace) -> int:
    proposals = list_proposals(args.inbox_dir)
    if not proposals:
        print(f"No proposals in {args.inbox_dir}")
        return 0

    print(f"{len(proposals)} proposal(s) in {args.inbox_dir}:")
    for proposal in proposals:
        print(_summarize(proposal))
    return 0


def _run_show(args: argparse.Namespace) -> int:
    proposal = load_proposal(args.proposal_id, args.inbox_dir)
    print(json.dumps(proposal.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_confirm(args: argparse.Namespace) -> int:
    path = confirm(
        args.proposal_id,
        _collect_edits(args),
        inbox_dir=args.inbox_dir,
        records_dir=args.records_dir,
    )
    print(f"Confirmed {args.proposal_id} -> {path}")
    return 0


def _run_reject(args: argparse.Namespace) -> int:
    path = reject(args.proposal_id, args.reason, inbox_dir=args.inbox_dir)
    print(f"Rejected {args.proposal_id} -> {path}")
    return 0


_COMMANDS = {
    "list": _run_list,
    "show": _run_show,
    "confirm": _run_confirm,
    "reject": _run_reject,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        return 1
    try:
        return handler(args)
    except ProposalError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
