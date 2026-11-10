import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture(scope="session")
def recognizer():
    from airwriting.config import CHAR_MODEL_PATH
    from airwriting.recognizer import Recognizer
    if not CHAR_MODEL_PATH.exists():
        pytest.skip("models/char_cnn.npz not found - run train_model.py first")
    return Recognizer(CHAR_MODEL_PATH)
