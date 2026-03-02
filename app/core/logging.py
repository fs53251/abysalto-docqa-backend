import logging
import sys

from app.core.config import settings
from app.core.request_context import get_identity, get_request_id

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
    
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        if not hasattr(record, "identity"):
            record.identity = get_identity()
        return record

    logging.setLogRecordFactory(record_factory)

    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | rid=%(request_id)s | "
            "id=%(identity)s | %(message)s"
        ),
    )