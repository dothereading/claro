"""Shared loading of repo-relative resource files.

The one place that knows where the repo root is and how to read YAML / text,
so callers don't re-derive `Path(__file__).resolve().parents[N]` and re-spell
`yaml.safe_load(path.read_text())` at every site that touches a config or
prompt file.

Pass a path relative to the repo root (e.g. ``"config/reward.yaml"``) and it
resolves against :data:`REPO_ROOT`; pass an absolute path and it is used
as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# This module lives at <repo>/claro/resources.py, so the repo root is two
# parents up.
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: str | Path) -> Path:
    """Absolute paths pass through; relative paths resolve against REPO_ROOT."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_text(path: str | Path) -> str:
    """Read a UTF-8 text file (repo-relative or absolute)."""
    return resolve(path).read_text(encoding="utf-8")


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file (repo-relative or absolute) into Python objects."""
    return yaml.safe_load(load_text(path))
