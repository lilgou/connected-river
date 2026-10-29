"""Notebook-facing helpers for preparing and running the complete workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

IRN_ACCESS_PAGE = "https://www.ceh.ac.uk/data/15000-watercourse-network"
HYDROMETRIC_AREAS_ACCESS_PAGE = (
    "https://catalogue.ceh.ac.uk/documents/1957166d-7523-44f4-b279-aa5314163237"
)
@dataclass(frozen=True)
class WorkflowRun:
    """Paths and provenance returned by a workflow run."""

    workspace: Path
    release_directory: Path
    manifest: dict[str, object]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_input(path: str | Path) -> str:
    """Checksum a file or a directory tree."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        return sha256_file(source)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def _copy_repository_inputs(repository: Path, workspace: Path) -> None:
    """Copy the authored configuration and release notes into a workspace."""

    files = (
        "THIRD_PARTY_DATA.md",
        "requirements-release.lock",
        "config/connected_river.yaml",
        "config/source_catalog.yaml",
        "output/scientific_matching_method/WATER_QUALITY_MATCHING_METHOD.md",
        "output/scientific_matching_method/decision_matrix.csv",
    )
    for relative in files:
        source = repository / relative
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    notebook_directory = workspace / "notebooks"
    notebook_directory.mkdir(parents=True, exist_ok=True)
    for notebook_name in (
        "The_Connected_River_Part_1_Building.ipynb",
        "The_Connected_River_Part_2_Exploring.ipynb",
    ):
        shutil.copy2(repository / notebook_name, notebook_directory / notebook_name)


def prepare_workspace(
    repository_root: str | Path,
    workspace_directory: str | Path,
) -> Path:
    """Create the data workspace used by downloads and generated outputs."""

    repository = Path(repository_root).resolve()
    workspace = Path(workspace_directory).expanduser()
    if not workspace.is_absolute():
        workspace = repository / workspace
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    _copy_repository_inputs(repository, workspace)
    os.environ["CONNECTED_RIVER_WORKSPACE"] = str(workspace)
    return workspace


def download_file(url: str, destination: str | Path) -> Path:
    """Download once and reuse a completed local file."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    return destination


def safe_extract_zip(archive_path: str | Path, destination: str | Path) -> Path:
    """Extract a ZIP after rejecting paths outside the destination."""

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe ZIP path: {member.filename}")
        archive.extractall(destination)
    return destination


def adapt_custom_network(
    source: str | Path,
    destination: str | Path,
    columns: dict[str, str],
) -> Path:
    """Map a directed edge table to the fields used by the matching workflow."""

    frame = pd.read_csv(source, low_memory=False)
    required = ["edge_id", "from_node", "to_node", "length_m", "geometry_wkt"]
    missing = [name for name in required if columns.get(name) not in frame]
    if missing:
        raise KeyError("Custom network is missing mapped fields: " + ", ".join(missing))

    def optional(name: str, default: object) -> object:
        source_name = columns.get(name)
        return frame[source_name] if source_name and source_name in frame else default

    adapted = pd.DataFrame(
        {
            "OBJECTID": frame[columns["edge_id"]],
            "RIVERS_ID": frame[columns["edge_id"]],
            "FNODE_FULL": frame[columns["from_node"]],
            "TNODE_FULL": frame[columns["to_node"]],
            "FLOWDIR": "With digitized",
            "TRACEABLE": 1,
            "ENABLED": 1,
            "STRAHLER": optional("strahler", pd.NA),
            "SHREVE": optional("shreve", pd.NA),
            "OS_NAME": optional("river_name", pd.NA),
            "HA": optional("area_id", pd.NA),
            "LENGTH": frame[columns["length_m"]],
            "SHAPE_LEN": frame[columns["length_m"]],
            "geometry": frame[columns["geometry_wkt"]],
        }
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_csv(destination, index=False)
    return destination


def _install_hydrological_areas(
    workspace: Path,
    network_directory: Path,
    separate_input: str | Path,
) -> Path:
    destination = workspace / "data/raw/river_structure/hydrological_areas"
    destination.mkdir(parents=True, exist_ok=True)
    search_roots = [network_directory]

    if separate_input:
        separate = Path(separate_input).expanduser().resolve()
        if not separate.exists():
            raise FileNotFoundError(separate)
        if separate.is_file() and separate.suffix.lower() == ".zip":
            unpacked = workspace / "data/raw/river_structure/_area_bundle"
            safe_extract_zip(separate, unpacked)
            search_roots.insert(0, unpacked)
        elif separate.is_dir():
            search_roots.insert(0, separate)
        else:
            search_roots.insert(0, separate.parent)

    shapefile = next(
        (
            candidate
            for root in search_roots
            for candidate in root.rglob("hyd_areas.shp")
        ),
        None,
    )
    if shapefile is None:
        raise FileNotFoundError(
            "Download the Hydrometric Areas with Coastline package from "
            f"{HYDROMETRIC_AREAS_ACCESS_PAGE} and set HYDROLOGICAL_AREAS_INPUT "
            "to its ZIP or folder."
        )
    for sidecar in shapefile.parent.glob(f"{shapefile.stem}.*"):
        shutil.copy2(sidecar, destination / sidecar.name)
    return destination / "hyd_areas.shp"


def _prepare_network(
    workspace: Path,
    network_mode: str,
    network_input: str | Path,
    hydrological_areas_input: str | Path,
    custom_network_columns: dict[str, str],
) -> tuple[Path, Path]:
    source = Path(network_input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    network_directory = workspace / "data/raw/river_structure"
    network_directory.mkdir(parents=True, exist_ok=True)
    irn_csv = network_directory / "IRN_river_line.csv"

    if network_mode == "custom_irn_schema":
        if source.suffix.lower() != ".csv":
            raise ValueError("The custom-network adapter currently expects one CSV")
        adapt_custom_network(source, irn_csv, custom_network_columns)
    elif network_mode == "ukceh_irn":
        if source.suffix.lower() == ".zip":
            safe_extract_zip(source, network_directory)
            matches = list(network_directory.rglob("IRN_river_line.csv"))
            if not matches:
                raise FileNotFoundError("The network ZIP contains no IRN_river_line.csv")
            if matches[0].resolve() != irn_csv.resolve():
                shutil.copy2(matches[0], irn_csv)
        else:
            shutil.copy2(source, irn_csv)
    else:
        raise ValueError("NETWORK_MODE must be 'ukceh_irn' or 'custom_irn_schema'")

    hydrological_areas = _install_hydrological_areas(
        workspace,
        network_directory,
        hydrological_areas_input,
    )
    return irn_csv, hydrological_areas


# This is the notebook-friendly entry point. It deals with the run folder and
# passes the chosen settings into the longer release workflow.
def build_or_resume(
    *,
    repository_root: str | Path,
    workspace_directory: str | Path,
    run_mode: str,
    existing_release_directory: str | Path,
    target_year: int,
    refresh_recent_years: int,
    network_mode: str,
    network_input: str | Path,
    hydrological_areas_input: str | Path,
    custom_network_columns: dict[str, str],
    matching_options: dict[str, object],
) -> WorkflowRun:
    """Prepare inputs, run the national build, or reopen an existing release."""

    if run_mode not in {"build_england", "use_existing"}:
        raise ValueError("RUN_MODE must be 'build_england' or 'use_existing'")
    window = int(matching_options["antecedent_window_months"])
    if not 1 <= window <= 12:
        raise ValueError("The antecedent window must be between 1 and 12 months")

    repository = Path(repository_root).resolve()
    workspace = prepare_workspace(repository, workspace_directory)
    hydrological_areas: Path | None = None

    if run_mode == "build_england":
        if not network_input:
            raise FileNotFoundError(
                "Set NETWORK_INPUT to the authorised IRN CSV or ZIP obtained from "
                + IRN_ACCESS_PAGE
            )
        _, hydrological_areas = _prepare_network(
            workspace,
            network_mode,
            network_input,
            hydrological_areas_input,
            custom_network_columns,
        )

        profile_path = workspace / "config/connected_river.yaml"
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        profile.setdefault("matching", {}).update(matching_options)
        profile.setdefault("run", {})["target_year"] = int(target_year)
        profile_path.write_text(
            yaml.safe_dump(profile, sort_keys=False),
            encoding="utf-8",
        )

        from connected_river.pipeline.annual_update import annual_update

        run_directory = annual_update(
            target_year,
            refresh_recent_years=refresh_recent_years,
            run_root=workspace / "runs",
        )
        release_directory = (run_directory / "release").resolve()
    else:
        if not existing_release_directory:
            raise FileNotFoundError(
                "Set EXISTING_RELEASE_DIRECTORY when RUN_MODE='use_existing'"
            )
        release_directory = Path(existing_release_directory).expanduser().resolve()
        if not release_directory.is_dir():
            raise FileNotFoundError(release_directory)

    manifest: dict[str, object] = {
        "run_mode": run_mode,
        "release_directory": str(release_directory),
        "target_year": int(target_year),
        "matching_options": matching_options,
        "network_mode": network_mode,
        "network_source_sha256": (
            sha256_file(network_input)
            if run_mode == "build_england" and network_input
            else None
        ),
        "hydrological_areas": str(hydrological_areas) if hydrological_areas else None,
        "hydrometric_areas_source": HYDROMETRIC_AREAS_ACCESS_PAGE,
        "hydrometric_areas_input_sha256": (
            sha256_input(hydrological_areas_input)
            if run_mode == "build_england" and hydrological_areas_input
            else None
        ),
        "irn_access_page": IRN_ACCESS_PAGE,
    }
    (workspace / "latest_method_run.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return WorkflowRun(workspace, release_directory, manifest)


def network_contract_table(column_mapping: dict[str, str]) -> pd.DataFrame:
    """Describe the minimum fields expected from another directed network."""

    rows = [
        ("edge_id", True, "unique and stable"),
        ("from_node", True, "upstream endpoint"),
        ("to_node", True, "downstream endpoint"),
        ("length_m", True, "finite and greater than zero"),
        ("geometry_wkt", True, "valid line in the declared CRS"),
        ("strahler", False, "positive where supplied"),
        ("shreve", False, "positive where supplied"),
        ("river_name", False, "descriptive only"),
        ("area_id", False, "regional grouping only"),
    ]
    table = pd.DataFrame(rows, columns=["pipeline field", "required", "check"])
    table["configured source column"] = table["pipeline field"].map(column_mapping)
    table["configured"] = table["configured source column"].notna()
    return table
