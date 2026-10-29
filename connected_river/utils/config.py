import os
from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    """Return the folder containing the notebook and supporting code."""

    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the active data workspace, or the repository by default."""

    configured = os.environ.get("CONNECTED_RIVER_WORKSPACE")
    return Path(configured).expanduser().resolve() if configured else repository_root()


# Keep YAML reading in one place so empty files behave the same way in notebooks,
# tests and command-line runs.
def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        workspace_path = project_root() / path
        repository_path = repository_root() / path
        path = workspace_path if workspace_path.exists() else repository_path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_profile(name: str) -> dict[str, Any]:
    """Load a named workflow profile from ``config/``."""
    if name.endswith((".yaml", ".yml")):
        path = Path("config") / name
    else:
        path = Path("config") / f"{name}.yaml"
    return load_yaml(path)


# Leave absolute paths alone, but resolve project-relative paths consistently.
# This also lets a user keep large downloads outside the repository.
def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path
