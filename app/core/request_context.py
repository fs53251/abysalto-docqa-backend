from __future__ import annotations

import contextvars

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="-",
)
_identity_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "identity",
    default="-",
)


def reset_request_context() -> None:
    _request_id_var.set("-")
    _identity_var.set("-")


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(value: str) -> None:
    _request_id_var.set(value)


def get_identity() -> str:
    return _identity_var.get()


def set_identity(value: str) -> None:
    _identity_var.set(value)
