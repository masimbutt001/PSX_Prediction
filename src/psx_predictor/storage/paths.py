"""Directory and path resolution utilities for local analytical storage."""

from pathlib import Path
from typing import Optional

STANDARD_DATA_DIRS = [
    "raw/prices",
    "raw/news",
    "raw/announcements",
    "raw/macro",
    "processed/prices",
    "processed/corporate_actions",
    "features/technical",
    "features/combined",
    "predictions",
    "models",
    "reports",
]


def get_storage_paths(base_dir: Optional[Path] = None) -> dict[str, Path]:
    """Resolve standard storage paths relative to the analytical base directory.

    Args:
        base_dir: Root analytical directory. Defaults to Path("data").

    Returns:
        Dictionary mapping directory key identifiers to Path objects.
    """
    root = base_dir or Path("data")
    paths: dict[str, Path] = {"root": root}

    for sub in STANDARD_DATA_DIRS:
        key = sub.replace("/", "_")
        paths[key] = root / sub

    return paths


def ensure_directories(base_dir: Optional[Path] = None) -> dict[str, Path]:
    """Ensure all required standard storage directories exist on disk.

    Args:
        base_dir: Root analytical directory. Defaults to Path("data").

    Returns:
        Dictionary mapping directory keys to resolved existing Path objects.
    """
    paths = get_storage_paths(base_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
