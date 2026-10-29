from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, get_ident
from typing import Any

try:  # POSIX production platforms (Linux/macOS).
    import fcntl
except ImportError:  # pragma: no cover - best-effort fallback on Windows.
    fcntl = None

from connected_river.utils.config import project_root, resolve_path

_MANIFEST_LOCK = Lock()


def portable_path(path: str | Path, *, root: str | Path | None = None) -> str:
    """Return a shareable path without including the local computer directory.

    Project files are saved relative to the project root. For files outside it,
    only the filename is kept; the hash is used to identify the file.
    """
    resolved = resolve_path(path).resolve()
    base = Path(root).resolve() if root is not None else project_root().resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.name


def file_sha256(path: str | Path) -> str:
    """Return a SHA256 digest for a single file."""
    path = resolve_path(path)
    if path.is_dir():
        raise IsADirectoryError(f"file_sha256 expects a file, got directory: {path}")

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def path_sha256(path: str | Path) -> str | None:
    """Return a stable SHA256 digest for a file or directory tree."""
    path = resolve_path(path)

    if not path.exists():
        return None

    if path.is_file():
        return file_sha256(path)

    if path.is_dir():
        h = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = child.relative_to(path).as_posix().encode("utf-8")
            h.update(rel)
            h.update(b"\0")
            with child.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            h.update(b"\0")
        return h.hexdigest()

    return None


def source_tree_sha256(path: str | Path) -> str | None:
    """Hash authored source files while ignoring interpreter cache residue."""
    source = resolve_path(path)
    if not source.exists():
        return None
    if not source.is_dir():
        return file_sha256(source)
    digest = hashlib.sha256()
    files = sorted(
        child for child in source.rglob("*")
        if child.is_file()
        and child.name != ".DS_Store"
        and child.suffix not in {".pyc", ".pyo"}
        and "__pycache__" not in child.parts
    )
    for child in files:
        digest.update(child.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def write_manifest_entry(
    manifest_path: str | Path,
    *,
    dataset: str,
    source_url: str,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    manifest_path = resolve_path(manifest_path)
    output_path = resolve_path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "dataset": dataset,
        "source_url": source_url,
        "output_path": portable_path(output_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": path_sha256(output_path),
        "metadata": metadata or {},
    }

    # Several downloads can finish at once. Lock the manifest while one result
    # is being added, and use a separate temporary file for each writer.
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    with _MANIFEST_LOCK, lock_path.open("a+", encoding="utf-8") as lock_stream:
        if fcntl is not None:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        temporary = manifest_path.with_name(
            f".{manifest_path.name}.{os.getpid()}.{get_ident()}.part"
        )
        try:
            existing: list[dict[str, Any]] = []
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing.append(entry)
            temporary.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
            if fcntl is not None:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return manifest_path
