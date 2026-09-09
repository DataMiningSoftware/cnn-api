import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flowers import FLOWER_CLASSES


def test_has_102_classes():
    assert len(FLOWER_CLASSES) == 102


def test_classes_are_unique():
    assert len(set(FLOWER_CLASSES)) == len(FLOWER_CLASSES)


def test_classes_are_nonempty_strings():
    assert all(isinstance(c, str) and c.strip() for c in FLOWER_CLASSES)
