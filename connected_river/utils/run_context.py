"""Run folders and source details used to reproduce a build."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from connected_river.utils.config import project_root, repository_root, resolve_path
from connected_river.utils.manifest import path_sha256, source_tree_sha256

AUTHORED_RELEASE_INPUTS = (
    "config/connected_river.yaml",
    "config/source_catalog.yaml",
    "notebooks/01_scientific_matching_decisions.ipynb",
    "notebooks/02_network_matching_workflow.ipynb",
    "output/scientific_matching_method/WATER_QUALITY_MATCHING_METHOD.md",
    "output/scientific_matching_method/decision_matrix.csv",
    "THIRD_PARTY_DATA.md",
    "requirements-release.lock",
)

RELEASE_LOCK = "requirements-release.lock"


def canonical_json_value(value: Any) -> Any:
    """Return the stable JSON representation used by persisted run context."""
    return json.loads(json.dumps(value, default=str))


def authored_release_input_manifest(root: str | Path) -> list[dict[str, str]]:
    """Hash authored methods/configuration copied into a public release."""
    root_path = Path(root)
    return [
        {"path": relative_path, "sha256": digest}
        for relative_path in AUTHORED_RELEASE_INPUTS
        if (path := root_path / relative_path).is_file()
        if (digest := path_sha256(path)) is not None
    ]


def code_revision() -> str | None:
    """Return the current Git revision when this checkout has one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


# Save the starting settings before any long-running work begins. A resumed run
# can then check that it is still using the same inputs and configuration.
def create_run_context(profile: str, config: dict[str, Any], root: str | Path = "runs") -> dict[str, Any]:
    """Create a run directory and save the settings used to start it."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = resolve_path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    root_path = project_root()
    lock_path = root_path / RELEASE_LOCK
    context = {
        "run_id": run_id,
        "profile": profile,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "code_revision": code_revision(),
        # Some distributed project folders are not Git checkouts. The source
        # tree digest still identifies the exact Python implementation used.
        "source_tree_sha256": source_tree_sha256(
            repository_root() / "connected_river"
        ),
        "authored_release_inputs": authored_release_input_manifest(root_path),
        "locked_environment": {
            "path": RELEASE_LOCK,
            "sha256": path_sha256(lock_path),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        # YAML dates are loaded as ``datetime.date`` objects. Convert them to
        # same JSON-compatible representation that will be read back later so
        # an unchanged configuration compares equal at promotion time.
        "config": canonical_json_value(config),
    }
    (run_dir / "run_context.json").write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    return {"run_id": run_id, "run_dir": str(run_dir), **context}
