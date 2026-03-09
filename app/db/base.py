from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase

# DeclarativeBase:
#   1) table mapping orm
#   2) ORM metadata collection
#   3) model class registration


class Base(DeclarativeBase):
    """
    Root parent for all database models.
    """

    pass
