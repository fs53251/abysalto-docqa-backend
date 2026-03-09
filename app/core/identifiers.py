from __future__ import annotations

import uuid

PUBLIC_DOCUMENT_ID_LENGTH = 32

# UUIDv4 identifier is a randomly generated unique ID.
# hexadecimal chars (0-9, a-f)


def generate_document_id() -> uuid.UUID:
    """
    Generate a UUIDv4 document identifier stored in the database.
    """
    return uuid.uuid4()


def document_public_id(value: uuid.UUID) -> str:
    """
    Return the public representation of a document UUID.
    """
    if not isinstance(value, uuid.UUID):
        raise TypeError("DOCUMENT_ID_MUST_BE_UUID")
    if value.version != 4:
        raise ValueError("DOCUMENT_ID_MUST_BE_UUID4")

    return value.hex


def parse_document_public_id(value: str) -> uuid.UUID:
    """
    Parse a public doc_id.

    Public doc_ids are lowercase UUIDv4 hex strings without dashes.
    Example: 8d7cf10cb6954f2daa7317c2a85fbc2f
    """
    if not isinstance(value, str):
        raise ValueError("INVALID_DOC_ID")

    normalized = value.strip().lower()
    if len(normalized) != PUBLIC_DOCUMENT_ID_LENGTH:
        raise ValueError("INVALID_DOC_ID")

    try:
        parsed = uuid.UUID(hex=normalized)
    except ValueError as exc:
        raise ValueError("INVALID_DOC_ID") from exc

    if parsed.version != 4 or parsed.hex != normalized:
        raise ValueError("INVALID_DOC_ID")

    return parsed


def is_document_public_id(value: str) -> bool:
    try:
        parse_document_public_id(value)
    except ValueError:
        return False
    return True
