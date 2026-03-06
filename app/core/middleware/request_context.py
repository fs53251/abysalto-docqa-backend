"""
Request-scoped context helpers.
    - Provide request_id (later user_id/session_id) as contextvars
    - Allow logging and error responses to include request_id consistently
"""

from __future__ import annotations

import contextvars

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
identity_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "identity", default="-"
)


def reset_request_context() -> None:
    request_id_var.set("-")
    identity_var.set("-")


def get_request_id() -> str:
    return request_id_var.get()


def set_request_id(value: str) -> None:
    request_id_var.set(value)


def get_identity() -> str:
    """
    Returns user/session identity when available.
    """
    return identity_var.get()


def set_identity(value: str) -> None:
    identity_var.set(value)
