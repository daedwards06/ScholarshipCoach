"""Generative-LLM layer for ScholarshipCoach.

The LLM lives strictly at the ingest boundary (Stage 0): a provider-agnostic
chat client whose outputs are validated and content-hash cached so everything
downstream of the snapshot stays deterministic and offline.
"""
from __future__ import annotations

from src.llm.client import LlmClient, LlmError, client_from_env
from src.llm.extraction import (
    EXTRACTION_FIELDS,
    EXTRACTION_PROMPT_VERSION,
    extract_fields,
    parse_extraction,
)

__all__ = [
    "EXTRACTION_FIELDS",
    "EXTRACTION_PROMPT_VERSION",
    "LlmClient",
    "LlmError",
    "client_from_env",
    "extract_fields",
    "parse_extraction",
]
