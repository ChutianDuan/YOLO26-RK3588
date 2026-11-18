"""Workspace paths and read-only relocation of historical artifact references."""
from pathlib import Path
import json

TEST_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TEST_ROOT.parent
SOURCE = WORKSPACE / 'yolo_run'
DEFAULT_BUILD = TEST_ROOT / 'build-rk3588' / 'bin'
DEFAULT_DATASET = TEST_ROOT / 'rknn_video_compare_20261018_002d290d'


def resolve_artifact(value):
    """Only known relocated prefixes are rewritten; raw records stay untouched."""
    path = Path(value)
    for old, new in ((WORKSPACE / 'yolo_run/validation', TEST_ROOT / 'validation'),
                     (WORKSPACE / 'Camera', TEST_ROOT / 'camera')):
        try:
            return new / path.relative_to(old)
        except ValueError:
            pass
    return path


def relocate_records(value):
    if isinstance(value, list):
        return [relocate_records(v) for v in value]
    if isinstance(value, dict):
        return {k: str(resolve_artifact(v)) if k in ('output', 'config') and isinstance(v, str)
                else relocate_records(v) for k, v in value.items()}
    return value


def load_history(path):
    return relocate_records(json.loads(resolve_artifact(path).read_text()))
