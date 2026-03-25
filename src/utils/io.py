"""I/O utility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclass/Path objects to JSON-serializable types."""
    if hasattr(obj, "__dict__") and not isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
