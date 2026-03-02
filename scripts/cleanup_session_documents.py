from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is importable when running as a script:
#   poetry run python scripts/cleanup_session_documents.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import and_, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.models import Document  # noqa: E402
from app.db.session import get_sessionmaker, init_db_dev_failsafe  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cleanup_session_documents")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def main(ttl_days: int, dry_run: bool) -> int:
    init_db_dev_failsafe()
    SessionLocal = get_sessionmaker()

    cutoff = _utcnow() - timedelta(days=ttl_days)

    with SessionLocal() as db:
        stmt = select(Document).where(
            and_(
                Document.owner_session_id.is_not(None),
                Document.owner_user_id.is_(None),
                Document.created_at < cutoff,
            )
        )
        docs = list(db.execute(stmt).scalars().all())

        logger.info(
            "Found %d expired session documents (ttl_days=%d)", len(docs), ttl_days
        )

        deleted = 0
        for doc in docs:
            logger.info(
                "Expired doc: id=%s filename=%s stored_path=%s",
                doc.id,
                doc.filename,
                doc.stored_path,
            )

            if not dry_run:
                # Try to delete stored file/directory if present
                if doc.stored_path:
                    p = Path(doc.stored_path)
                    try:
                        if p.is_file():
                            p.unlink(missing_ok=True)
                        # If it sits under uploads/{doc_id}/original, delete doc folder too
                        doc_dir = (
                            p.parent.parent if p.parent.name == "original" else p.parent
                        )
                        if doc_dir.exists():
                            for child in sorted(doc_dir.rglob("*"), reverse=True):
                                if child.is_file():
                                    child.unlink(missing_ok=True)
                                else:
                                    child.rmdir()
                            doc_dir.rmdir()
                    except Exception as e:
                        logger.warning(
                            "Failed to delete files for doc=%s: %s", doc.id, e
                        )

                db.delete(doc)
                deleted += 1

        if not dry_run:
            db.commit()

    logger.info("Deleted %d documents", deleted)
    return deleted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl-days", type=int, default=settings.SESSION_TTL_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main(ttl_days=args.ttl_days, dry_run=args.dry_run)
