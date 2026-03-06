from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User


def normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(
    db: Session, *, email: str, password_hash: str, is_active: bool = True
) -> User:
    user = User(
        email=normalize_email(email),
        password_hash=password_hash,
        is_active=is_active,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_user_by_email(db: Session, *, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == normalize_email(email))
    return db.execute(stmt).scalars().first()


def get_user(db: Session, *, user_id: uuid.UUID) -> Optional[User]:
    stmt = select(User).where(User.id == user_id)
    return db.execute(stmt).scalars().first()
