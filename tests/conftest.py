from pathlib import Path

import pytest

from app.core.config import settings


@pytest.fixture()
def temp_data_dir(tmp_path: Path):
    """
    Uses a temporary DATA_DIR for tests and restores the original
    value after execution.
    """
    old = settings.DATA_DIR
    settings.DATA_DIR = str(tmp_path)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    yield tmp_path
    settings.DATA_DIR = old
