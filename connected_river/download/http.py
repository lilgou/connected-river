"""Shared HTTP helpers for connected-river data acquisition."""

from __future__ import annotations

import hashlib
import re
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from connected_river.utils.config import resolve_path
from connected_river.utils.io import ensure_dir
from connected_river.utils.manifest import write_manifest_entry

DEFAULT_HEADERS = {"User-Agent": "ConnectedRiver/0.1 (+https://environment.data.gov.uk/)"}


def session_with_retries(total: int = 6, backoff_factor: float = 1.0) -> requests.Session:
    """Return a requests session that retries temporary download failures."""
    session = requests.Session()
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(DEFAULT_HEADERS)
    return session


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Stream downloads to a temporary file first. A partial response should never
# replace an existing good copy if the connection drops halfway through.
def download_file(
    url: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    timeout: int = 180,
    dataset: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Download a public file with retries and replace it only when complete."""
    out = resolve_path(output_path).resolve()
    ensure_dir(out.parent)
    if out.exists() and not overwrite:
        return out
    tmp = out.with_name(f".{out.name}.{uuid4().hex}.part")
    try:
        with session_with_retries() as s:
            with s.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tmp.open("xb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    if dataset:
        write_manifest_entry(
            "outputs/logs/download_manifest.json",
            dataset=dataset,
            source_url=url,
            output_path=out,
            metadata={**(metadata or {}), "bytes": out.stat().st_size, "sha256": file_sha256(out)},
        )
    return out


# Check member paths before extracting a zip. This also stops a badly formed
# archive from scattering files outside its data folder.
def extract_zip(
    zip_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    max_members: int = 100_000,
    max_uncompressed_bytes: int = 20_000_000_000,
) -> list[Path]:
    """Safely extract a publisher ZIP and return its file paths.

    Member paths are validated before anything is written. Symlinks,
    encrypted entries, traversal paths and implausibly large archives are
    rejected so a compromised upstream archive cannot write outside the
    requested raw-data directory.
    """
    zpath = resolve_path(zip_path)
    out_dir = ensure_dir(output_dir)
    destination_root = out_dir.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(zpath) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if len(members) > max_members:
            raise ValueError(f"ZIP contains {len(members):,} files; limit is {max_members:,}")
        total_size = sum(info.file_size for info in members)
        if total_size > max_uncompressed_bytes:
            raise ValueError(
                f"ZIP expands to {total_size:,} bytes; limit is {max_uncompressed_bytes:,}"
            )
        validated: list[tuple[zipfile.ZipInfo, Path]] = []
        seen: set[Path] = set()
        for info in members:
            member = Path(info.filename)
            unix_type = (info.external_attr >> 16) & 0o170000
            if info.flag_bits & 0x1:
                raise ValueError(f"Encrypted ZIP member is not supported: {info.filename}")
            if unix_type == stat.S_IFLNK:
                raise ValueError(f"ZIP symlink is not allowed: {info.filename}")
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member path: {info.filename}")
            target = (destination_root / member).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise ValueError(f"ZIP member escapes output directory: {info.filename}")
            if target in seen:
                raise ValueError(f"Duplicate ZIP member path: {info.filename}")
            seen.add(target)
            validated.append((info, target))

        for info, target in validated:
            if target.exists() and not overwrite:
                extracted.append(target)
                continue
            ensure_dir(target.parent)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.part")
            try:
                with zf.open(info) as src, temporary.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted.append(target)
    return extracted


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "download"


def read_tabular(path: str | Path, *, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read common public-data tabular formats into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, dtype=str)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0, dtype=str)
    if suffix in {".json", ".geojson"}:
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular file type: {path}")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make public-data columns easier to join across files."""
    out = df.copy()
    out.columns = [re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(c)).strip("_").lower()) for c in out.columns]
    return out
