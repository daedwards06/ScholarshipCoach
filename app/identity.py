"""Who is using the app, from the headers ``tailscale serve`` adds.

``tailscale serve`` passes ``Tailscale-User-Login`` and ``Tailscale-User-Name``
on requests from devices owned by a signed-in tailnet user.  A ``[family]``
table in ``.streamlit/secrets.toml`` (server only, never committed) maps each
login to ``student`` or ``parent``.  No header, or a login the table does not
name, means no identity and the family PIN decides, as before.

The decision functions are pure; only the ``*_from_*`` readers touch Streamlit.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import streamlit as st

Role = Literal["student", "parent"]

LOGIN_HEADER = "Tailscale-User-Login"
NAME_HEADER = "Tailscale-User-Name"
IDENTITY_HEADERS = (LOGIN_HEADER, NAME_HEADER)
FAMILY_SECRET = "family"
ROLES: tuple[Role, ...] = ("student", "parent")


@dataclass(frozen=True)
class Identity:
    login: str
    name: str
    role: Role


def _header(headers: Mapping[str, Any] | None, name: str) -> str:
    if not headers:
        return ""
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value or "").strip()
    return ""


def header_presence(headers: Mapping[str, Any] | None) -> dict[str, bool]:
    """Which identity headers arrived, by name only — never their values."""
    return {name: bool(_header(headers, name)) for name in IDENTITY_HEADERS}


def role_for_headers(
    headers: Mapping[str, Any] | None, family_map: Mapping[str, Any] | None
) -> Identity | None:
    """Return the mapped identity for a request, or None to fall back to the PIN."""
    login = _header(headers, LOGIN_HEADER).casefold()
    if not login or not family_map:
        return None
    roles = {str(key).strip().casefold(): str(value).strip().casefold() for key, value in family_map.items()}
    role = roles.get(login)
    if role not in ROLES:
        return None
    name = _header(headers, NAME_HEADER) or login
    return Identity(login=login, name=name, role=cast(Role, role))


def family_map_from_secrets() -> dict[str, Any]:
    """Read ``[family]`` from ``st.secrets``; no secrets file or no table is ``{}``."""
    try:
        table = st.secrets.get(FAMILY_SECRET)
    except Exception:
        # Streamlit raises rather than returning empty when there is no secrets.toml.
        return {}
    return dict(table) if isinstance(table, Mapping) else {}


def headers_from_context() -> Mapping[str, Any]:
    try:
        return cast(Mapping[str, Any], st.context.headers)
    except Exception:
        return {}


def current_identity() -> Identity | None:
    return role_for_headers(headers_from_context(), family_map_from_secrets())


def activity_key(identity: Identity | None, mode: str) -> str:
    """Whom an opened page counts for: the login, else the role the mode stands for."""
    if identity is not None:
        return identity.login
    return "student" if mode == "student" else "parent"
