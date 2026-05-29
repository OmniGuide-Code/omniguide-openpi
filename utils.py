"""Shared utility functions for the OmniGuide guidance pipeline."""

import json
import os
from pathlib import Path


def save_json(data, path: str | Path) -> None:
    """Write data to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str | Path) -> dict:
    """Load and return a JSON file."""
    with open(path) as f:
        return json.load(f)
