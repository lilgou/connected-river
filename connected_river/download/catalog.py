"""Versioned source catalogue, generic acquisition and health reporting."""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import pandas as pd
import yaml

from connected_river.download.http import download_file, session_with_retries
from connected_river.utils.config import project_root


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    theme: str
    kind: str
    refresh: str
    default: bool
    required_for_release: bool
    options: dict[str, object]


# Read source details from YAML rather than spreading URLs and filenames across
# the download scripts. This is the main place to update a source later.
def load_catalog(path: str | Path = "config/source_catalog.yaml") -> dict[str, Source]:
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        source_id: Source(
            source_id,
            item["title"],
            item["theme"],
            item["kind"],
            item["refresh"],
            bool(item.get("default", False)),
            bool(item.get("required_for_release", False)),
            item,
        )
        for source_id, item in data["sources"].items()
    }


def _filename_from_url(url: str, fallback: str) -> str:
    query_name = parse_qs(urlparse(url).query).get("fileName", [None])[0]
    candidate = query_name if query_name else Path(urlparse(url).path).name or fallback
    return _validated_filename(candidate)


def _validated_filename(value: object) -> str:
    """Return a safe, single-component filename or reject the value."""
    name = str(value).strip()
    for _ in range(8):
        decoded = unquote(name)
        if decoded == name:
            break
        name = decoded
    else:
        raise ValueError(f"Unsafe download filename: {value!r}")

    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ":" in name
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or len(posix.parts) != 1
        or len(windows.parts) != 1
    ):
        raise ValueError(f"Unsafe download filename: {value!r}")
    return name


# Some Defra pages link to several yearly files. Discovery keeps that changing
# list out of the code while still checking each filename before using it.
def discover_defra_files(
    dataset_id: str,
    include_pattern: str | None = None,
    *,
    timeout: int = 90,
    retries: int = 6,
) -> list[tuple[str, str]]:
    page = f"https://environment.data.gov.uk/dataset/{dataset_id}"
    with session_with_retries(total=retries) as session:
        response = session.get(page, timeout=timeout)
        response.raise_for_status()
    body = html_lib.unescape(response.text)
    links: list[tuple[str, str]] = []
    next_data = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', body, flags=re.I | re.S)
    if next_data:
        try:
            payload = json.loads(next_data.group(1))
            def walk(value: object) -> None:
                if isinstance(value, dict):
                    uri = value.get("fileURI") or value.get("fileUri") or value.get("downloadUrl")
                    if uri:
                        uri = str(uri).replace(
                            "https://api.agrimetrics.co.uk/file-management/data-sets/",
                            "https://environment.data.gov.uk/file-management-open/data-sets/",
                        )
                        filename = _validated_filename(
                            value.get("name")
                            or value.get("id")
                            or _filename_from_url(str(uri), "download")
                        )
                        if not include_pattern or re.search(include_pattern, filename):
                            links.append((str(uri), filename))
                    for child in value.values():
                        walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)
            walk(payload)
        except json.JSONDecodeError:
            pass
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', body, flags=re.I)
    for href in hrefs:
        url = urljoin(page, href)
        if "/api/file/download" not in url and "/file-management-open/" not in url and "/file-management/data-sets/" not in url:
            continue
        filename = _filename_from_url(url, "download")
        if include_pattern and not re.search(include_pattern, filename):
            continue
        links.append((url, filename))
    return list(dict.fromkeys(links))


# Use the same acquisition route for direct files and discovered file lists, so
# both end up with the same cache checks and source record.
def acquire_source(source: Source, raw_root: str | Path = "data/raw/catalog", *, overwrite: bool = False) -> list[Path]:
    raw_root = Path(raw_root)
    if not raw_root.is_absolute():
        raw_root = project_root() / raw_root
    output_dir = raw_root / source.source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if source.kind in {"direct", "api_snapshot"}:
        url = str(source.options["url"])
        filename = _validated_filename(
            source.options.get("filename") or _filename_from_url(url, source.source_id)
        )
        return [download_file(url, output_dir / filename, overwrite=overwrite, dataset=source.source_id, metadata={"theme": source.theme, "refresh": source.refresh})]
    if source.kind == "defra_dataset":
        files = discover_defra_files(str(source.options["dataset_id"]), str(source.options.get("include_pattern") or "") or None)
        if not files:
            raise RuntimeError(f"No matching downloadable files discovered for {source.source_id}")
        return [
            download_file(
                url,
                output_dir / _validated_filename(filename),
                overwrite=overwrite,
                dataset=source.source_id,
                metadata={
                    "theme": source.theme,
                    "dataset_id": source.options["dataset_id"],
                },
            )
            for url, filename in files
        ]
    if source.kind == "reference":
        raise RuntimeError(f"{source.source_id} is catalogue-only: {source.options.get('note', 'no bulk download is exposed')}")
    raise ValueError(f"Unsupported source kind: {source.kind}")


def _content_canary(
    payload: bytes,
    *,
    content_type: str,
    filename: str,
    required_columns: list[str] | None = None,
) -> dict[str, object]:
    """Validate a small response sample without downloading a full national asset."""

    if not payload:
        raise ValueError("source returned no content bytes")
    lowered = content_type.lower()
    suffix = Path(filename).suffix.lower()
    columns: list[str] = []
    temporal_values: list[str] = []
    year_detection_method = "not_available_for_format"
    if suffix == ".csv" or "csv" in lowered:
        text = payload.decode("utf-8-sig", errors="replace")
        reader = csv.reader(StringIO(text))
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV canary has no header row")
        columns = [
            str(column).strip()
            for column in header
        ]
        if not columns or not any(columns):
            raise ValueError("CSV canary has no columns")
        temporal_indexes = [
            index
            for index, column in enumerate(columns)
            if any(token in column.casefold() for token in ("year", "date", "time", "period"))
        ]
        if temporal_indexes:
            year_detection_method = "declared_csv_temporal_columns"
            for row_number, row in enumerate(reader):
                if row_number >= 100:
                    break
                temporal_values.extend(
                    row[index] for index in temporal_indexes if index < len(row)
                )
        detected_format = "csv"
    elif suffix == ".zip" or "zip" in lowered:
        if not payload.startswith(b"PK"):
            raise ValueError("ZIP canary does not have a ZIP signature")
        detected_format = "zip"
    elif suffix in {".json", ".geojson"} or "json" in lowered:
        if payload.lstrip()[:1] not in {b"{", b"["}:
            raise ValueError("JSON canary does not begin with an object or array")
        detected_format = "json"
    elif "html" in lowered or suffix in {".htm", ".html"}:
        sample = payload[:4096].lower()
        if b"<html" not in sample and b"<!doctype html" not in sample:
            raise ValueError("HTML canary has no document signature")
        detected_format = "html"
        temporal_values = [payload[:65536].decode("utf-8", errors="ignore")]
        year_detection_method = "bounded_html_text"
    else:
        detected_format = suffix.lstrip(".") or "binary"

    required_columns = [str(value) for value in (required_columns or [])]
    missing = sorted(set(required_columns) - set(columns)) if required_columns else []
    if missing:
        raise ValueError("canary schema missing required columns: " + ", ".join(missing))
    upper_year = datetime.now(timezone.utc).year + 2
    years = [
        int(value)
        for text_value in temporal_values
        for value in re.findall(r"(?:19|20)\d{2}", text_value)
        if 1900 <= int(value) <= upper_year
    ]
    return {
        "content_verified": True,
        "sample_bytes": len(payload),
        "detected_format": detected_format,
        "schema_columns": " | ".join(columns),
        "discovered_latest_year": max(years) if years else None,
        "year_detection_method": year_detection_method,
    }


# A quick probe catches moved pages, login screens and unexpected file formats
# before a long national run gets as far as data processing.
def probe_source(
    source: Source,
    *,
    deep: bool = False,
    target_year: int | None = None,
) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    try:
        canary = {
            "content_verified": False,
            "sample_bytes": 0,
            "detected_format": None,
            "schema_columns": "",
            "discovered_latest_year": None,
            "year_detection_method": "not_checked",
        }
        last_modified = None
        if source.options.get("local_path"):
            local = project_root() / str(source.options["local_path"])
            status = "healthy" if local.exists() and local.stat().st_size else "unavailable"
            detail = f"registered local source; {local.stat().st_size:,} bytes" if status == "healthy" else f"local source missing: {local}"
            if deep and status == "healthy":
                with local.open("rb") as stream:
                    payload = stream.read(65536)
                canary = _content_canary(
                    payload,
                    content_type="",
                    filename=local.name,
                    required_columns=source.options.get("required_columns"),
                )
        elif source.kind == "defra_dataset":
            discovery_timeout = 60 if source.required_for_release else 30
            discovery_retries = 5 if source.required_for_release else 2
            files = discover_defra_files(
                str(source.options["dataset_id"]),
                str(source.options.get("include_pattern") or "") or None,
                timeout=discovery_timeout,
                retries=discovery_retries,
            )
            status, detail = ("healthy", f"{len(files)} matching files") if files else ("degraded", "dataset page reachable but no matching files")
            if deep and files:
                url, filename = files[0]
                with session_with_retries(total=2, backoff_factor=0.2) as session:
                    response = session.get(
                        url, stream=True, timeout=45, headers={"Range": "bytes=0-65535"}
                    )
                    response.raise_for_status()
                    payload = next(response.iter_content(chunk_size=65536), b"")
                    last_modified = response.headers.get("last-modified")
                    canary = _content_canary(
                        payload,
                        content_type=response.headers.get("content-type", ""),
                        filename=filename,
                        required_columns=source.options.get("required_columns"),
                    )
        else:
            with session_with_retries(total=2, backoff_factor=0.2) as session:
                response = session.request(
                    str(source.options.get("probe_method", "GET")), str(source.options["url"]),
                    stream=True, timeout=30,
                    headers={"Accept": str(source.options.get("probe_accept", "*/*"))},
                )
                response.raise_for_status()
                detail, status = f"HTTP {response.status_code}; {response.headers.get('content-type', 'unknown')}", "healthy"
                last_modified = response.headers.get("last-modified")
                if deep:
                    payload = next(response.iter_content(chunk_size=65536), b"")
                    canary = _content_canary(
                        payload,
                        content_type=response.headers.get("content-type", ""),
                        filename=str(
                            source.options.get("filename")
                            or _filename_from_url(str(source.options["url"]), source.source_id)
                        ),
                        required_columns=source.options.get("required_columns"),
                    )
    except Exception as exc:
        status, detail = "unavailable", f"{type(exc).__name__}: {exc}"
        canary = {
            "content_verified": False,
            "sample_bytes": 0,
            "detected_format": None,
            "schema_columns": "",
            "discovered_latest_year": None,
            "year_detection_method": "check_failed",
        }
        last_modified = None
    latest_year = canary["discovered_latest_year"]
    target_year_status = (
        "not_requested" if target_year is None
        else "observed_in_canary" if latest_year is not None and int(latest_year) >= int(target_year)
        else "not_observable_from_canary"
    )
    return {
        "source_id": source.source_id,
        "title": source.title,
        "theme": source.theme,
        "required_for_release": source.required_for_release,
        "status": status,
        "detail": detail,
        "check_depth": "content_canary" if deep else "reachability",
        **canary,
        "last_modified": last_modified,
        "target_year": target_year,
        "target_year_status": target_year_status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": (datetime.now(timezone.utc) - started).total_seconds(),
    }


# Return one row per configured source so the notebook can show what is ready,
# optional or currently unavailable without hiding failed checks.
def source_health(
    catalog: dict[str, Source],
    output_dir: str | Path = "outputs/source_health",
    *,
    workers: int = 8,
    deep: bool = False,
    target_year: int | None = None,
) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(catalog)))) as executor:
        result = pd.DataFrame(
            executor.map(
                lambda source: probe_source(source, deep=deep, target_year=target_year),
                catalog.values(),
            )
        )
    output_dir = project_root() / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "source_health.csv", index=False)
    (output_dir / "source_health.json").write_text(result.to_json(orient="records", indent=2), encoding="utf-8")
    return result


def required_source_failures(health: pd.DataFrame) -> pd.DataFrame:
    """Return required release sources that are not currently healthy."""

    required = {"source_id", "required_for_release", "status"}
    missing = sorted(required - set(health.columns))
    if missing:
        raise KeyError(f"Source-health table is missing required columns: {missing}")
    return health.loc[
        health["required_for_release"].fillna(False).astype(bool)
        & ~health["status"].astype("string").eq("healthy")
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and acquire connected-river sources.")
    parser.add_argument("command", choices=["list", "health", "download"])
    parser.add_argument("sources", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Read and validate a bounded content sample, including declared CSV columns.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        help="Record whether that year is visible in the bounded canary; not every source embeds years.",
    )
    parser.add_argument(
        "--fail-required",
        action="store_true",
        help="Exit non-zero when any source declared required_for_release is not healthy",
    )
    parser.add_argument("--output-dir", default="outputs/source_health")
    args = parser.parse_args()
    catalog = load_catalog()
    selected = catalog if args.all else {key: value for key, value in catalog.items() if key in args.sources or (not args.sources and value.default)}
    if args.command == "list":
        print(pd.DataFrame([{
            "source_id": s.source_id,
            "theme": s.theme,
            "refresh": s.refresh,
            "default": s.default,
            "required_for_release": s.required_for_release,
            "title": s.title,
        } for s in catalog.values()]).to_string(index=False))
    elif args.command == "health":
        health = source_health(
            selected,
            output_dir=args.output_dir,
            deep=args.deep,
            target_year=args.target_year,
        )
        print(health.to_string(index=False))
        failures = required_source_failures(health)
        if args.fail_required and not failures.empty:
            print(
                "Required sources unavailable: "
                + ", ".join(failures["source_id"].astype(str)),
            )
            raise SystemExit(1)
    else:
        for source in selected.values():
            print(source.source_id, *acquire_source(source, overwrite=args.overwrite), sep="\n  ")


if __name__ == "__main__":
    main()
