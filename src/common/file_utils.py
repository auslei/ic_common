"""
Common file utility functions.
"""
from pathlib import Path

def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
