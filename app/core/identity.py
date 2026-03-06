from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RequestIdentity:
    kind: Literal["user", "session"]
    user_id: uuid.UUID | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "user":
            if self.user_id is None or self.session_id is not None:
                raise ValueError("INVALID_USER_IDENTITY")
            return

        if self.kind == "session":
            if self.user_id is not None or not self.session_id:
                raise ValueError("INVALID_SESSION_IDENTITY")
            return

        raise ValueError("INVALID_IDENTITY_KIND")

    @property
    def log_identity(self) -> str:
        if self.kind == "user":
            return f"user:{self.user_id}"
        return f"sess:{self.session_id}"

    @classmethod
    def for_user(cls, user_id: uuid.UUID | str) -> "RequestIdentity":
        parsed = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
        return cls(kind="user", user_id=parsed, session_id=None)

    @classmethod
    def for_session(cls, session_id: str) -> "RequestIdentity":
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("SESSION_ID_REQUIRED")
        return cls(kind="session", user_id=None, session_id=normalized)
