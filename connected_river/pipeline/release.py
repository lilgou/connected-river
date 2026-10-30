"""Build a restartable connected-river release from declared inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from connected_river.chemistry_registry import determinand_registry_records
from connected_river.features.analysis_ready_product import (
    build_analysis_ready_product,
    build_data_dictionary,
)
from connected_river.features.chemistry_features import (
    build_antecedent_chemistry_features_from_path,
)
from connected_river.features.habitat_context import build_habitat_context_ledger
from connected_river.features.integrated_product import build_connected_river_product
from connected_river.features.trends import build_repeated_site_trends
from connected_river.features.waterbody_dossier import build_waterbody_evidence_dossier
from connected_river.ingest.catchment_planning import build_catchment_planning_index
from connected_river.ingest.ecology_bulk import build_ecology_index
from connected_river.ingest.edm import build_edm_index
from connected_river.ingest.river_habitat import build_river_habitat_index
from connected_river.ingest.rsn import build_rsn_index
from connected_river.mapping.catchment_explorer import (
    assign_hydrological_areas,
    build_catchment_map_geometry,
    build_hydrological_area_map_geometry,
)
from connected_river.mapping.river_routes import (
    ROUTE_ASSET_NAME,
    build_river_connection_route_asset,
)
from connected_river.pipeline.context_integration import (
    build_context_integrated_analysis,
)
from connected_river.pipeline.contracts import validate_site_year, write_qa
from connected_river.pipeline.network_integration import (
    NetworkIntegrationSettings,
    build_network_integrated_evidence,
)
from connected_river.pipeline.source_coverage import build_source_temporal_coverage
from connected_river.utils.config import load_profile, project_root, repository_root
from connected_river.utils.manifest import (
    path_sha256,
    portable_path,
    source_tree_sha256,
)
from connected_river.utils.run_context import (
    authored_release_input_manifest,
    canonical_json_value,
    create_run_context,
)
from connected_river.validation.chemistry_coverage_ledger import (
    write_chemistry_coverage_outputs,
)

_OPTIONAL_RELEASE_ASSETS = (
    ("data/processed/mapping/catchment_map_geometry.geojson", "catchment_map_geometry.geojson"),
    (
        "data/processed/mapping/hydrological_area_map_geometry.geojson",
        "hydrological_area_map_geometry.geojson",
    ),
    (
        "notebooks/The_Connected_River_Part_1_Building.ipynb",
        "the-connected-river-part-1-building.ipynb",
    ),
    (
        "notebooks/The_Connected_River_Part_2_Exploring.ipynb",
        "the-connected-river-part-2-exploring.ipynb",
    ),
    (
        "output/scientific_matching_method/WATER_QUALITY_MATCHING_METHOD.md",
        "WATER_QUALITY_MATCHING_METHOD.md",
    ),
    (
        "output/scientific_matching_method/decision_matrix.csv",
        "water-quality-matching-decision-matrix.csv",
    ),
    ("THIRD_PARTY_DATA.md", "THIRD_PARTY_DATA.md"),
    ("config/source_catalog.yaml", "source_catalog.yaml"),
    ("config/connected_river.yaml", "release_profile.yaml"),
    ("requirements-release.lock", "requirements-release.lock"),
)

_PROMOTION_REQUIRED_ASSETS = (
    "connected_river_analysis_ready.csv.zip",
    "connected_river_data_dictionary.csv",
    "target_determinand_coverage.csv",
    "target_determinand_acquisition.csv",
    "connected_river_integrated_data.parquet",
    "storm_overflow_asset_year.parquet",
    "edm_source_inventory.parquet",
    "analysis_ready_qa.json",
    "deep_release_qa.json",
    "download_manifest.json",
    "waterbody_evidence_dossier.parquet",
    "wfd_river_objectives.parquet",
    "source_catalog.yaml",
    "release_profile.yaml",
    "run_context.json",
    "requirements-release.lock",
    "habitat_context_ledger.parquet",
    "chemistry_coverage_ledger.parquet",
    "chemistry_coverage_summary.csv",
    "2018-onward-chemistry-coverage.md",
    "tributary_flow_proxy_validation.json",
    "tributary_flow_proxy_nrfa_gauges.parquet",
    "selected_match_route_edm.parquet",
    "WATER_QUALITY_MATCHING_METHOD.md",
    "water-quality-matching-decision-matrix.csv",
    "01-building-the-connected-river.ipynb",
    "02-exploring-the-connected-river.ipynb",
    "THIRD_PARTY_DATA.md",
    "README.md",
    ROUTE_ASSET_NAME,
)

_RELEASE_ASSET_ROLES = {
    "connected_river_analysis_ready.csv.zip": "primary compressed analysis-ready dataset",
    "connected_river_data_dictionary.csv": "field-level data dictionary",
    "target_determinand_coverage.csv": "target-determinand registry and observed coverage",
    "target_determinand_acquisition.csv": (
        "exact-code archive-year inventory and 2018-onward completeness"
    ),
    "connected_river_integrated_data.parquet": "lossless specialist dataset",
    "storm_overflow_asset_year.parquet": "annual EDM asset and spill evidence",
    "edm_source_inventory.parquet": "annual EDM ingestion and spatial coverage QA",
    "analysis_ready_qa.json": "analysis-ready dataset validation",
    "deep_release_qa.json": "machine-readable release QA",
    "supporting_release_qa.json": "checks for supporting release files",
    "download_manifest.json": "saved download details",
    "run_context.json": "saved build settings and source digest",
    "requirements-release.lock": "fully resolved release environment lock",
    "habitat_context_ledger.parquet": "date-bounded RHS replacement-context sidecar",
    "chemistry_coverage_ledger.parquet": "one mutually exclusive chemistry outcome per automated survey",
    "chemistry_coverage_summary.csv": "chemistry outcomes by programme, geography, year and evidence support",
    "2018-onward-chemistry-coverage.md": "2018-onward automated-product chemistry coverage interpretation",
    "source_catalog.yaml": "saved source list",
    "release_profile.yaml": "saved release settings",
    "tributary_flow_proxy_validation.json": "gauged validation of the tributary flow-influence proxy",
    "tributary_flow_proxy_nrfa_gauges.parquet": "gauge-level IRN flow-proxy validation evidence",
    "catchment_map_geometry.geojson": "release catchment geometry",
    "hydrological_area_map_geometry.geojson": "release hydrological-area geometry",
    "hydrological_area_assignments.parquet": "release-record to hydrological-area map",
    ROUTE_ASSET_NAME: "exact biology-to-chemistry routes with bounded neighbouring IRN structure",
    "WATER_QUALITY_MATCHING_METHOD.md": "matching method and decisions",
    "water-quality-matching-decision-matrix.csv": "machine-readable matching decision matrix",
    "01-building-the-connected-river.ipynb": "versioned workflow notebook source at build time",
    "02-exploring-the-connected-river.ipynb": "versioned analysis notebook source at build time",
    "THIRD_PARTY_DATA.md": "third-party data redistribution cautions",
    "wfd_river_objectives.parquet": "WFD river objectives and alternative-objective context",
    "wfd_protected_areas.parquet": "protected areas linked to WFD water bodies",
    "waterbody_evidence_dossier.parquet": "one monitoring and regulatory summary per water body",
    "README.md": "release-specific methods and limitations",
}


def _write_json_safely(path: Path, payload: Any) -> Path:
    """Write JSON through a temporary file before replacing the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_csv_zip(csv_path: Path) -> Path:
    """Create a repeatable ZIP and replace the final file only when complete."""
    zip_path = csv_path.with_suffix(csv_path.suffix + ".zip")
    temporary = zip_path.with_name(f".{zip_path.name}.{uuid4().hex}.part")
    member = ZipInfo(csv_path.name, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = 0o644 << 16
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            with csv_path.open("rb") as source, archive.open(
                member, "w", force_zip64=True
            ) as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        temporary.replace(zip_path)
    finally:
        temporary.unlink(missing_ok=True)
    return zip_path


def _write_target_determinand_acquisition(
    inventory_path: Path,
    release_dir: Path,
    years: range | list[int] | tuple[int, ...],
    *,
    required_from_year: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Write a code-by-year download summary from the raw-partition inventory.

    Duplicate caches are reported but never summed:
    ``largest_valid_cache_rows`` is the largest valid cache for a code/year,
    which avoids presenting duplicate downloads as additional monitoring
    observations. The row count is labelled as a cache property because cached
    partitions can have different raw API scopes before the common river and
    material filter is applied during ingestion.

    By default every requested code/year pair remains mandatory. Supplying
    ``required_from_year`` keeps earlier missing rows visible in the summary but
    enforces completeness from that year onward.
    """
    inventory = pd.read_parquet(inventory_path).copy()
    parsed = inventory["file"].astype("string").str.extract(
        r"(?P<year>(?:19|20)\d{2})[_-].*[_-](?P<determinand_code>\d{4})\.(?:csv|parquet)$",
    )
    inventory = pd.concat([inventory, parsed], axis=1)
    inventory["year"] = pd.to_numeric(inventory["year"], errors="coerce").astype("Int64")
    inventory["determinand_code"] = inventory["determinand_code"].astype("string")
    inventory["rows"] = pd.to_numeric(inventory["rows"], errors="coerce").fillna(0)

    registry = pd.DataFrame(determinand_registry_records())
    rows: list[dict[str, object]] = []
    for year in sorted({int(value) for value in years}):
        for item in registry.to_dict("records"):
            code = str(item["determinand_code"]).zfill(4)
            year_matches = inventory["year"].eq(year).fillna(False)
            code_matches = inventory["determinand_code"].eq(code).fillna(False)
            found = inventory[year_matches & code_matches]
            statuses = set(found["status"].astype(str))
            loaded = found[found["status"].eq("loaded").fillna(False)]
            if not loaded.empty:
                status = "observations_loaded"
                largest_cache_rows = int(loaded["rows"].max())
            elif "queried zero observations" in statuses:
                status = "queried_zero_observations"
                largest_cache_rows = 0
            else:
                status = "missing_or_invalid"
                largest_cache_rows = 0
            cache_files = sorted({portable_path(path) for path in found["file"].dropna()})
            rows.append({
                "year": year,
                "determinand_code": code,
                "determinand_label": item["determinand_label"],
                "feature_key": item["feature_key"],
                "source_unit": item["source_unit"],
                "standard_unit": item["standard_unit"],
                "acquisition_status": status,
                "largest_valid_cache_rows": largest_cache_rows,
                "cache_file_count": len(cache_files),
                "duplicate_cache_file_count": max(0, len(cache_files) - 1),
                "cache_files": ";".join(cache_files),
            })
    coverage = pd.DataFrame(rows)
    missing = coverage[coverage["acquisition_status"].eq("missing_or_invalid")]
    required_missing = (
        missing
        if required_from_year is None
        else missing[missing["year"].ge(int(required_from_year))]
    )
    if not required_missing.empty:
        preview = ", ".join(
            f"{row.year}/{row.determinand_code}"
            for row in required_missing.head(12).itertuples()
        )
        raise RuntimeError(
            "Target-determinand download summary contains "
            f"{len(required_missing)} required missing or invalid partitions "
            f"(first: {preview})"
        )
    output_path = release_dir / "target_determinand_acquisition.csv"
    coverage.to_csv(output_path, index=False)
    return coverage, output_path


# Write both a compact CSV for general use and a fuller Parquet file for people
# who want the detailed matching and route information.
def _write_public_data_products(
    joined_product: pd.DataFrame,
    release_dir: Path,
    acquisition_coverage: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write the supported ZIP, dictionary and their QA record.

    The uncompressed CSV is an intermediate only. Keeping another 100+ MB
    copy beside its byte-identical ZIP adds no scientific or operational value.
    """
    release_dir.mkdir(parents=True, exist_ok=True)
    # Keep the supported ZIP as the only full CSV distribution product.
    (release_dir / "connected_river_integrated_data.csv").unlink(missing_ok=True)

    analysis_ready = build_analysis_ready_product(joined_product)
    analysis_ready_csv = release_dir / "connected_river_analysis_ready.csv"
    analysis_ready.to_csv(analysis_ready_csv, index=False)
    analysis_ready_zip = _write_csv_zip(analysis_ready_csv)
    analysis_ready_csv.unlink()

    dictionary_csv = release_dir / "connected_river_data_dictionary.csv"
    dictionary = build_data_dictionary(analysis_ready)
    dictionary.to_csv(dictionary_csv, index=False)
    coverage = pd.DataFrame(determinand_registry_records())
    origin = joined_product.get(
        "data_origin", pd.Series(pd.NA, index=joined_product.index, dtype="string"),
    ).astype("string")
    sample_year = pd.to_numeric(joined_product.get("sample_year"), errors="coerce")
    for row_index, item in coverage.iterrows():
        count_column = f"target_exact_{item['feature_key']}_observation_count_6mo"
        counts = pd.to_numeric(
            joined_product.get(count_column, pd.Series(0, index=joined_product.index)),
            errors="coerce",
        ).fillna(0)
        automated_observed = counts.gt(0) & origin.eq("connected_river_pipeline")
        automated_years = sample_year[automated_observed].dropna()
        coverage.loc[row_index, "surveys_with_observations"] = int(
            automated_observed.sum()
        )
        # Survey windows overlap by design, so this is a sum of observation
        # uses across windows, not a count of unique source observations.
        coverage.loc[row_index, "survey_window_observation_uses"] = int(
            counts[origin.eq("connected_river_pipeline")].sum()
        )
        coverage.loc[row_index, "first_observed_year"] = (
            int(automated_years.min()) if not automated_years.empty else pd.NA
        )
        coverage.loc[row_index, "latest_observed_year"] = (
            int(automated_years.max()) if not automated_years.empty else pd.NA
        )
        if acquisition_coverage is not None:
            acquired = acquisition_coverage[
                acquisition_coverage["determinand_code"].astype("string").str.zfill(4)
                .eq(str(item["determinand_code"]).zfill(4))
            ]
            observed = acquired[acquired["acquisition_status"].eq("observations_loaded")]
            coverage.loc[row_index, "acquisition_years_expected"] = len(acquired)
            coverage.loc[row_index, "acquisition_years_with_observations"] = len(observed)
            coverage.loc[row_index, "acquisition_years_queried_zero"] = int(
                acquired["acquisition_status"].eq("queried_zero_observations").sum()
            )
            coverage.loc[row_index, "acquisition_largest_cache_rows_total"] = int(
                pd.to_numeric(
                    observed["largest_valid_cache_rows"], errors="coerce",
                ).fillna(0).sum()
            )
            coverage.loc[row_index, "acquisition_first_observed_year"] = (
                int(observed["year"].min()) if not observed.empty else pd.NA
            )
            coverage.loc[row_index, "acquisition_latest_observed_year"] = (
                int(observed["year"].max()) if not observed.empty else pd.NA
            )
    coverage_csv = release_dir / "target_determinand_coverage.csv"
    coverage.to_csv(coverage_csv, index=False)
    analysis_ready_qa = {
        "status": "passed",
        "full_rows": len(joined_product),
        "analysis_ready_rows": len(analysis_ready),
        "analysis_ready_columns": len(analysis_ready.columns),
        "all_null_columns": [
            column for column in analysis_ready.columns if analysis_ready[column].isna().all()
        ],
        "record_ids_identical": bool(
            analysis_ready["integration_record_id"].equals(
                joined_product["integration_record_id"]
            )
        ),
        "dictionary_columns_identical": bool(
            dictionary["column"].tolist() == list(analysis_ready.columns)
        ),
    }
    if not (
        analysis_ready_qa["full_rows"] == analysis_ready_qa["analysis_ready_rows"]
        and not analysis_ready_qa["all_null_columns"]
        and analysis_ready_qa["record_ids_identical"]
        and analysis_ready_qa["dictionary_columns_identical"]
    ):
        analysis_ready_qa["status"] = "failed"
        raise ValueError("Analysis-ready release QA failed")
    qa_path = release_dir / "analysis_ready_qa.json"
    _write_json_safely(qa_path, analysis_ready_qa)
    return {
        "analysis_ready_zip": analysis_ready_zip,
        "dictionary": dictionary_csv,
        "determinand_coverage": coverage_csv,
        "qa": qa_path,
    }


def _recorded_project_path(value: object, root: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.name
    return path.as_posix().lstrip("./")


def _build_consumed_input_manifest(
    root: Path, run_dir: Path, release_dir: Path,
) -> Path:
    """Record the raw and downloaded files used by completed stages."""
    acquisition_path = root / "outputs/logs/download_manifest.json"
    acquisitions: dict[tuple[str, str | None], dict[str, Any]] = {}
    if acquisition_path.is_file():
        try:
            global_entries = json.loads(acquisition_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            global_entries = []
        for entry in global_entries if isinstance(global_entries, list) else []:
            output = _recorded_project_path(entry.get("output_path", ""), root)
            key = (output, entry.get("sha256"))
            # The last successful acquisition is the most useful metadata for a
            # duplicated cache entry; the content hash still fixes its identity.
            acquisitions[key] = entry

    used: dict[tuple[str, str | None], dict[str, Any]] = {}
    for stage_path in sorted((run_dir / "stages").glob("*.json")):
        try:
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if stage.get("status") != "complete":
            continue
        stage_name = str(stage.get("stage") or stage_path.stem)
        for item in stage.get("inputs", []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            path = _recorded_project_path(item["path"], root)
            if not path.startswith("data/raw/"):
                continue
            key = (path, item.get("sha256"))
            entry = used.setdefault(key, {
                "dataset": None,
                "source_url": None,
                "output_path": path,
                "sha256": item.get("sha256"),
                "downloaded_at_utc": None,
                "metadata": {},
                "used_by_stages": [],
            })
            if stage_name not in entry["used_by_stages"]:
                entry["used_by_stages"].append(stage_name)
            acquired = acquisitions.get(key)
            if acquired is None:
                acquired = next(
                    (value for (candidate_path, _), value in acquisitions.items()
                     if candidate_path == path),
                    None,
                )
            if acquired:
                entry.update({
                    "dataset": acquired.get("dataset"),
                    "source_url": acquired.get("source_url"),
                    "downloaded_at_utc": acquired.get("downloaded_at_utc"),
                    "metadata": acquired.get("metadata") or {},
                })
            raw_input = root / path
            if not raw_input.is_dir():
                continue
            prefix = path.rstrip("/") + "/"
            # Stage records hash a raw directory as a whole. Preserve that
            # compact integrity check, and also record every current downloaded
            # file beneath it so the release retains query metadata and an
            # independently verifiable file hash. Stale or page-order duplicate
            # acquisition events cannot enter because only the hash currently
            # present on disk is accepted.
            for (candidate_path, candidate_hash), candidate_acquisition in acquisitions.items():
                if not candidate_hash or not candidate_path.startswith(prefix):
                    continue
                candidate_file = root / candidate_path
                if not candidate_file.is_file() or path_sha256(candidate_file) != candidate_hash:
                    continue
                candidate_key = (candidate_path, candidate_hash)
                candidate_entry = used.setdefault(candidate_key, {
                    "dataset": candidate_acquisition.get("dataset"),
                    "source_url": candidate_acquisition.get("source_url"),
                    "output_path": candidate_path,
                    "sha256": candidate_hash,
                    "downloaded_at_utc": candidate_acquisition.get("downloaded_at_utc"),
                    "metadata": candidate_acquisition.get("metadata") or {},
                    "used_by_stages": [],
                })
                if stage_name not in candidate_entry["used_by_stages"]:
                    candidate_entry["used_by_stages"].append(stage_name)
    entries = sorted(used.values(), key=lambda item: (item["output_path"], item["sha256"] or ""))
    return _write_json_safely(release_dir / "download_manifest.json", entries)


def _freeze_optional_release_assets(
    root: Path, release_dir: Path, *, run_dir: Path | None = None,
) -> list[Path]:
    """Copy the methods, notebooks and source records available for the release."""
    copied: list[Path] = []
    for source_name, destination_name in _OPTIONAL_RELEASE_ASSETS:
        source = root / source_name
        destination = release_dir / destination_name
        if source.is_file():
            shutil.copy2(source, destination)
            copied.append(destination)
        else:
            destination.unlink(missing_ok=True)
    if run_dir is not None:
        copied.append(_build_consumed_input_manifest(root, run_dir, release_dir))
        context = run_dir / "run_context.json"
        if context.is_file():
            destination = release_dir / "run_context.json"
            shutil.copy2(context, destination)
            copied.append(destination)
    return copied


def _write_hydrological_assignments(
    full: pd.DataFrame, root: Path, release_dir: Path,
) -> Path | None:
    """Save map-area membership so the raw shapefile is not needed later."""
    shapefile = root / "data/raw/river_structure/hydrological_areas/hyd_areas.shp"
    required = {"integration_record_id", "easting", "northing"}
    output = release_dir / "hydrological_area_assignments.parquet"
    if not shapefile.is_file() or not required.issubset(full.columns):
        output.unlink(missing_ok=True)
        return None
    assigned = assign_hydrological_areas(
        full[["integration_record_id", "easting", "northing"]], shapefile,
    )
    assigned[["integration_record_id", "map_area_id", "catchment_name"]].to_parquet(
        output, index=False,
    )
    return output


def _run_deep_release_audit(run_dir: Path) -> dict[str, Any]:
    """Audit the current-source specialist product and persist the result."""
    release_dir = run_dir / "release"
    product_path = release_dir / "connected_river_integrated_data.parquet"
    product = pd.read_parquet(product_path, columns=[
        "integration_record_id", "data_origin", "sample_date", "sample_year",
    ])
    temporal = json.loads(
        (release_dir / "public_temporal_coverage_qa.json").read_text(encoding="utf-8")
    )
    compact = json.loads(
        (release_dir / "analysis_ready_qa.json").read_text(encoding="utf-8")
    )
    dates = pd.to_datetime(product["sample_date"], errors="coerce")
    years = pd.to_numeric(product["sample_year"], errors="coerce")
    checks = [
        {
            "name": "unique_integration_record_id",
            "passed": bool(product["integration_record_id"].notna().all() and product["integration_record_id"].is_unique),
            "severity": "error",
            "detail": "Every released survey has one non-null stable identifier.",
        },
        {
            "name": "current_pipeline_rows_only",
            "passed": bool(product["data_origin"].eq("connected_river_pipeline").all()),
            "severity": "error",
            "detail": "The release contains only rows acquired and processed by this pipeline.",
        },
        {
            "name": "sample_date_year_consistent",
            "passed": bool(dates.notna().all() and dates.dt.year.eq(years).all()),
            "severity": "error",
            "detail": "Sample dates and recorded sample years agree.",
        },
        {
            "name": "requested_years_complete",
            "passed": not temporal.get("missing_years"),
            "severity": "error",
            "detail": "No requested biology year is missing from the joined product.",
        },
        {
            "name": "analysis_ready_export_passed",
            "passed": compact.get("status") == "passed",
            "severity": "error",
            "detail": "The compact CSV and specialist Parquet have matching record identities.",
        },
    ]
    deep_qa = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "run_id": run_dir.name,
        "rows": len(product),
        "columns": len(pq.read_schema(product_path).names),
        "release_sha256": path_sha256(product_path),
        "checks": checks,
    }
    release_qa_path = run_dir / "release/deep_release_qa.json"
    run_qa_path = run_dir / "qa/deep_release_qa.json"
    _write_json_safely(release_qa_path, deep_qa)
    _write_json_safely(run_qa_path, deep_qa)
    if deep_qa["status"] != "passed":
        failures = [
            check["name"] for check in deep_qa["checks"]
            if check["severity"] == "error" and not check["passed"]
        ]
        raise ValueError("Public release QA failed: " + ", ".join(failures))
    return deep_qa


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _promote_release(
    *,
    root: Path,
    run_dir: Path,
    deep_qa: dict[str, Any],
    promotion_gates: dict[str, bool] | None = None,
) -> Path:
    """Publish outputs only after the completed run passes its checks."""
    if deep_qa.get("status") != "passed":
        raise ValueError("Outputs cannot be published until the detailed QA passes")
    success_path = run_dir / "SUCCESS"
    if not success_path.is_file():
        raise RuntimeError("Outputs cannot be published before the run is complete")
    if promotion_gates is None:
        gate_path = run_dir / "qa/promotion_gates.json"
        if not gate_path.is_file():
            raise RuntimeError("Outputs cannot be published without saved validation checks")
        promotion_gates = json.loads(gate_path.read_text(encoding="utf-8"))
    failed_gates = sorted(name for name, passed in promotion_gates.items() if not passed)
    if failed_gates:
        raise ValueError("Output validation failed: " + ", ".join(failed_gates))
    release_dir = run_dir / "release"
    missing = [
        filename for filename in _PROMOTION_REQUIRED_ASSETS
        if not (release_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required output files are missing: " + ", ".join(missing)
        )
    persisted_qa_path = release_dir / "deep_release_qa.json"
    try:
        persisted_qa = json.loads(persisted_qa_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Saved output QA is not valid JSON") from exc
    if persisted_qa != deep_qa:
        raise ValueError("The supplied QA does not match the saved output audit")
    if persisted_qa.get("run_id") not in {None, run_dir.name}:
        raise ValueError("The saved output QA belongs to a different run")
    public_path = release_dir / "connected_river_integrated_data.parquet"
    public_sha256 = path_sha256(public_path)
    if not persisted_qa.get("release_sha256") or persisted_qa["release_sha256"] != public_sha256:
        raise ValueError("The output Parquet no longer matches the saved output audit")

    context_path = run_dir / "run_context.json"
    context = (
        json.loads(context_path.read_text(encoding="utf-8"))
        if context_path.is_file()
        else {}
    )
    assets = []
    for path in sorted(item for item in release_dir.iterdir() if item.is_file()):
        assets.append({
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": path_sha256(path),
            "role": _RELEASE_ASSET_ROLES.get(path.name, "supporting release asset"),
        })
    promoted_at = success_path.read_text(encoding="utf-8").strip()
    manifest = {
        "schema_version": "1.0",
        "release_id": run_dir.name,
        "created_at": promoted_at or datetime.now(timezone.utc).isoformat(),
        "publication_status": "promoted",
        "source_directory": _relative_to_root(release_dir, root),
        "profile": context.get("profile"),
        "code_revision": context.get("code_revision"),
        "locked_environment": context.get("locked_environment"),
        "build_source_tree_sha256": context.get("source_tree_sha256"),
        "promotion_source_tree_sha256": source_tree_sha256(repository_root() / "connected_river"),
        "qa": {
            "status": deep_qa["status"],
            "rows": deep_qa.get("rows"),
            "columns": deep_qa.get("columns"),
            "release_sha256": deep_qa.get("release_sha256"),
        },
        "promotion_gates": promotion_gates,
        "assets": assets,
    }
    manifests_dir = root / "releases"
    versioned_path = _write_json_safely(manifests_dir / f"{run_dir.name}.json", manifest)
    _write_json_safely(manifests_dir / "current.json", manifest)
    return versioned_path


def _evaluate_candidate_gates(
    *, root: Path, run_dir: Path, deep_qa: dict[str, Any], config: dict[str, Any],
) -> dict[str, bool]:
    """Run the final output checks without changing the published release."""
    release_dir = run_dir / "release"
    context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    specialist_path = release_dir / "connected_river_integrated_data.parquet"
    rows = pd.read_parquet(
        specialist_path,
        columns=["integration_record_id", "data_origin"],
    )
    automated_rows = int(rows["data_origin"].eq("connected_river_pipeline").sum())
    try:
        decisions = pd.read_parquet(
            release_dir / "biology_chemistry_match_decisions.parquet",
            columns=[
                "survey_date", "candidate_support_window_end",
                "candidate_support_temporal_method",
            ],
        )
        leakage_safe = bool(
            pd.to_datetime(decisions["candidate_support_window_end"], errors="coerce")
            .le(pd.to_datetime(decisions["survey_date"], errors="coerce"))
            .fillna(False).all()
            and decisions["candidate_support_temporal_method"].eq(
                "six_months_through_first_survey"
                if int(config.get("matching", {}).get("antecedent_window_months", 6)) == 6
                else (
                    f"{int(config.get('matching', {}).get('antecedent_window_months', 6))}"
                    "_calendar_months_through_first_survey"
                )
            ).all()
        )
    except (OSError, KeyError, ValueError):
        leakage_safe = False
    try:
        determinands = pd.read_csv(release_dir / "target_determinand_coverage.csv")
        all_determinand_codes = determinands["determinand_code"].astype("string").nunique() == 41
    except (OSError, KeyError, ValueError):
        all_determinand_codes = False
    try:
        coverage = pd.read_parquet(
            release_dir / "chemistry_coverage_ledger.parquet",
            columns=["integration_record_id", "chemistry_coverage_outcome"],
        )
        exhaustive_coverage = (
            len(coverage) == automated_rows
            and coverage["integration_record_id"].is_unique
            and coverage["chemistry_coverage_outcome"].notna().all()
        )
    except (OSError, KeyError, ValueError):
        exhaustive_coverage = False
    matching = config.get("matching", {})
    try:
        configured_tributary_penalty_points = float(matching.get("tributary_penalty_points", 0))
        if not np.isfinite(configured_tributary_penalty_points) or configured_tributary_penalty_points < 0:
            configured_tributary_penalty_points = 0.0
    except (TypeError, ValueError):
        configured_tributary_penalty_points = 0.0
    tributary_penalty_auditable = configured_tributary_penalty_points >= 0
    lock = context.get("locked_environment") or {}
    revision_and_lock = bool(
        context.get("code_revision")
        and lock.get("sha256")
        and lock.get("sha256") == path_sha256(root / str(lock.get("path")))
    )

    required_assets = all(
        (release_dir / filename).is_file() for filename in _PROMOTION_REQUIRED_ASSETS
    )
    gates = {
        "no_future_evidence_affects_earlier_survey": leakage_safe,
        "all_41_determinands_separate": bool(all_determinand_codes),
        "tributary_penalty_validated_or_experimental": tributary_penalty_auditable,
        "every_automated_survey_has_one_coverage_outcome": bool(exhaustive_coverage),
        "deep_release_qa_passed": deep_qa.get("status") == "passed",
        "required_candidate_assets_present": required_assets,
        "real_revision_and_locked_environment": revision_and_lock,
    }
    _write_json_safely(run_dir / "qa/promotion_gates.json", gates)
    return gates


def _write_candidate_manifest(
    *, root: Path, run_dir: Path, deep_qa: dict[str, Any], promotion_gates: dict[str, bool],
) -> Path:
    """Save a candidate manifest without changing current.json."""
    release_dir = run_dir / "release"
    context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    assets = [
        {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": path_sha256(path),
            "role": _RELEASE_ASSET_ROLES.get(path.name, "supporting release asset"),
        }
        for path in sorted(release_dir.iterdir()) if path.is_file()
    ]
    manifest = {
        "schema_version": "1.1",
        "release_id": run_dir.name,
        "created_at": (run_dir / "SUCCESS").read_text(encoding="utf-8").strip(),
        "publication_status": "candidate",
        "promotion_eligible": all(promotion_gates.values()),
        "source_directory": _relative_to_root(release_dir, root),
        "profile": context.get("profile"),
        "code_revision": context.get("code_revision"),
        "locked_environment": context.get("locked_environment"),
        "build_source_tree_sha256": context.get("source_tree_sha256"),
        "candidate_source_tree_sha256": source_tree_sha256(repository_root() / "connected_river"),
        "qa": {
            "status": deep_qa.get("status"), "rows": deep_qa.get("rows"),
            "columns": deep_qa.get("columns"),
            "release_sha256": deep_qa.get("release_sha256"),
        },
        "promotion_gates": promotion_gates,
        "assets": assets,
    }
    return _write_json_safely(
        root / "releases/candidates" / f"{run_dir.name}.json", manifest
    )


def _record(stage_path: Path, *, name: str, status: str, inputs: list[Path], outputs: list[Path]) -> None:
    stage_path.write_text(json.dumps({
        "stage": name, "status": status, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"path": portable_path(p), "sha256": path_sha256(p)} for p in inputs],
        "outputs": [{"path": portable_path(p), "sha256": path_sha256(p)} for p in outputs],
    }, indent=2), encoding="utf-8")


def _recorded_path(path: object, root: Path) -> Path:
    recorded = Path(str(path))
    return recorded if recorded.is_absolute() else root / recorded


def _completed(stage_file: Path, outputs: dict[str, Path]) -> bool:
    """Return true only when the stage record and current content hashes agree."""
    if not stage_file.exists() or not all(path.exists() for path in outputs.values()):
        return False
    try:
        record = json.loads(stage_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if record.get("status") != "complete":
        return False
    root = project_root()
    for kind in ("inputs", "outputs"):
        entries = record.get(kind)
        if not isinstance(entries, list):
            return False
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("path"):
                return False
            path = _recorded_path(entry["path"], root)
            if path_sha256(path) != entry.get("sha256"):
                return False
    return True


def _assert_run_context_current(
    run_dir: Path,
    *,
    profile: str,
    config: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Prevent a resumed or long-running build from mixing authored versions."""
    context_path = run_dir / "run_context.json"
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Run context is unreadable: {context_path}") from exc
    mismatches: list[str] = []
    if context.get("profile") != profile:
        mismatches.append("profile")
    if context.get("config") != canonical_json_value(config):
        mismatches.append("configuration")
    if context.get("source_tree_sha256") != source_tree_sha256(repository_root() / "connected_river"):
        mismatches.append("source tree")
    frozen_authored = context.get("authored_release_inputs")
    if frozen_authored != authored_release_input_manifest(root):
        mismatches.append("authored methods/configuration files")
    if mismatches:
        raise RuntimeError(
            "Cannot continue a run after its saved "
            + ", ".join(mismatches)
            + " changed; start a fresh audited run"
        )
    return context


def _resolve_discharge_csv(root: Path) -> Path:
    """Return the acquired consented-discharge table used for path attribution."""
    portable = root / "data/raw/discharge_permits/consented_sewage_discharges.csv"
    if portable.is_file():
        return portable
    legacy = root / (
        "data/raw/discharges/stream/"
        "arcgis_storm_overflow_Consented_Discharge_-_Sewage_"
        "Consented_Discharges_to_Controlled_Waters_0.csv"
    )
    if legacy.is_file():
        return legacy
    candidates = sorted(
        (root / "data/raw/discharge_permits/extracted").rglob("*.csv"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No consented-discharge CSV is available. Run the annual updater first."
        )
    return candidates[0]


@dataclass(frozen=True)
class _ProductStageOutputs:
    joined_product: pd.DataFrame
    product_path: Path
    habitat_context_path: Path
    trends_path: Path
    ecology_qa_path: Path
    source_coverage_path: Path
    public_coverage_qa_path: Path
    expected_public_years: list[int]


# Download and clean each source separately. A failed source can then be fixed
# and resumed without rebuilding everything that came before it.
def _build_source_stages(
    *,
    root: Path,
    data_dir: Path,
    stage_dir: Path,
    years: list[int],
    raw_ecology_dir: str | Path | None,
) -> tuple[
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
    Path,
]:
    """Build and validate the ecology, pressure and planning source indices."""
    raw_dir = Path(raw_ecology_dir) if raw_ecology_dir else root / "data/raw/full_river_health_sources/ea_ecology_fish"
    ecology_output_dir = data_dir / "ecology"
    ecology_outputs = {name: ecology_output_dir / f"ecology_{name}.parquet" for name in ("sites", "samples", "metrics", "taxa", "taxonomy", "site_year", "source_inventory")}
    if not _completed(stage_dir / "ecology.json", ecology_outputs):
        ecology_outputs = build_ecology_index(raw_dir, ecology_output_dir)
        _record(stage_dir / "ecology.json", name="ecology", status="complete", inputs=sorted(raw_dir.glob("*.zip")), outputs=list(ecology_outputs.values()))

    edm_raw = root / "data/raw/storm_overflows/edm"
    edm_output_dir = data_dir / "edm"
    edm_outputs = {"asset_year": edm_output_dir / "edm_asset_year.parquet", "inventory": edm_output_dir / "edm_source_inventory.parquet"}
    if not _completed(stage_dir / "edm.json", edm_outputs):
        edm_outputs = build_edm_index(edm_raw, edm_output_dir, years)
        _record(stage_dir / "edm.json", name="edm", status="complete", inputs=sorted(edm_raw.glob("*.zip")), outputs=list(edm_outputs.values()))
    edm = pd.read_parquet(edm_outputs["asset_year"])
    edm_inventory = pd.read_parquet(edm_outputs["inventory"])
    acquired_edm_years = {
        int(path.stem.split("_")[1])
        for path in edm_raw.glob("EDM_*_Storm_Overflow_Annual_Return.zip")
    }
    acquired_inventory = edm_inventory.loc[
        pd.to_numeric(edm_inventory["year"], errors="coerce").isin(
            acquired_edm_years
        )
    ]
    invalid_edm_years = acquired_inventory.loc[
        ~acquired_inventory["status"].eq("loaded")
        | pd.to_numeric(
            acquired_inventory["assets_with_network_location"], errors="coerce"
        ).lt(100)
    ]
    if len(acquired_inventory) != len(acquired_edm_years) or not invalid_edm_years.empty:
        raise ValueError(
            "EDM annual-return QA failed for acquired years: "
            + ", ".join(map(str, invalid_edm_years.get("year", [])))
        )
    invalid_edm = edm["spill_duration_hours"].dropna().lt(0) | edm["spill_duration_hours"].dropna().gt(8784)
    if invalid_edm.any():
        raise ValueError(f"EDM duration QA failed: {int(invalid_edm.sum())} values outside 0-8,784 hours")

    catalog_raw = root / "data/raw/catalog"
    rsn_candidates = [
        path for path in (catalog_raw / "river_surveillance_network").glob("*.zip")
        if not path.name.endswith(".gpkg.zip")
    ]
    rsn_zip = max(rsn_candidates, key=lambda path: (path.stat().st_mtime_ns, path.name), default=None)
    if rsn_zip is None:
        raise FileNotFoundError("NCEA River Surveillance Network ZIP is missing; run the source catalogue downloader")
    rsn_outputs = build_rsn_index(rsn_zip, data_dir / "rsn")
    _record(stage_dir / "rsn.json", name="rsn", status="complete", inputs=[rsn_zip], outputs=list(rsn_outputs.values()))

    wfd_outputs = build_catchment_planning_index(catalog_raw, data_dir / "catchment_planning")
    _record(stage_dir / "catchment_planning.json", name="catchment_planning", status="complete", inputs=[catalog_raw / "catchment_classifications/classifications.csv", catalog_raw / "catchment_objectives/objectives.csv", catalog_raw / "catchment_rnags/rnags.csv", catalog_raw / "catchment_protected_areas/protected_areas.csv"], outputs=list(wfd_outputs.values()))

    map_dir = root / "data/processed/mapping"
    catchment_geometry_source = catalog_raw / "catchment_geometry/water_bodies.geojson"
    if catchment_geometry_source.is_file():
        build_catchment_map_geometry(
            catchment_geometry_source, map_dir / "catchment_map_geometry.geojson",
        )
    hydrology_source = root / "data/raw/river_structure/hydrological_areas/hyd_areas.shp"
    if hydrology_source.is_file():
        build_hydrological_area_map_geometry(
            hydrology_source, map_dir / "hydrological_area_map_geometry.geojson",
        )

    rhs_zip = max(
        (catalog_raw / "river_habitat_surveys").glob("*.zip"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    rhs_output = build_river_habitat_index(rhs_zip, data_dir / "river_habitat/river_habitat_surveys.parquet")
    _record(stage_dir / "river_habitat.json", name="river_habitat", status="complete", inputs=[rhs_zip], outputs=[rhs_output])

    return ecology_outputs, edm_outputs, rsn_outputs, wfd_outputs, rhs_output
# This is the main spatial step: sites are placed on the river network and the
# configurable matching recipe is applied.
def _build_network_stage(
    *,
    root: Path,
    data_dir: Path,
    stage_dir: Path,
    years: list[int],
    matching_config: dict[str, Any],
    ecology_outputs: dict[str, Path],
    edm_outputs: dict[str, Path],
    wfd_outputs: dict[str, Path],
    exact_determinand_complete_from_year: int | None,
    enforce_target_determinand_completeness: bool,
) -> tuple[dict[str, Path], Path]:
    """Build or reopen the river-network matching stage."""
    network_dir = data_dir / "network_integration"
    discharge_csv = _resolve_discharge_csv(root)
    network_expected = {
        "ranked_matches": network_dir / "ranked_biology_chemistry_matches.parquet",
        "selected_matches": network_dir / "selected_biology_chemistry_matches.parquet",
        "integrated": network_dir / "river_connected_site_year_evidence.parquet",
        "path_edm": network_dir / "selected_match_route_edm.parquet",
        "qa": network_dir / "network_integration_qa.json",
        "sensitivity": network_dir / "match_distance_sensitivity.parquet",
        "graph": network_dir / "network/irn_graph.pkl",
        "biology_snaps": network_dir / "biology_site_irn_snaps.parquet",
        "wq_snaps": network_dir / "water_quality_site_irn_snaps.parquet",
        "observations": network_dir / "water_quality/wq_observations.parquet",
        "features": network_dir / "water_quality/wq_site_year_features.parquet",
        "inventory": network_dir / "water_quality/wq_source_inventory.parquet",
    }
    if not _completed(stage_dir / "network_integration.json", network_expected):
        network_settings = NetworkIntegrationSettings(
            biology_site_year_path=ecology_outputs["site_year"], edm_asset_year_path=edm_outputs["asset_year"],
            edm_snap_path=None, water_quality_raw_dir=root / "data/raw/water_quality",
            discharge_csv=discharge_csv,
            irn_raw_csv=root / "data/raw/river_structure/IRN_river_line.csv", output_dir=network_dir,
            years=years, wfd_status_path=wfd_outputs["status_summary"],
            wfd_pressure_path=wfd_outputs["pressure_summary"],
            max_snap_distance_m=float(matching_config.get("max_snap_distance_m", 1000)),
            max_euclidean_distance_m=float(matching_config.get("max_euclidean_distance_m", 15000)),
            max_network_distance_m=float(matching_config.get("max_network_distance_m", 10000)),
            tributary_transport_decay_scale_m=float(
                matching_config.get("tributary_transport_decay_scale_m", 10000)
            ) if matching_config.get("tributary_transport_decay_scale_m") is not None else None,
            tributary_penalty_points=float(
                matching_config.get("tributary_penalty_points", 0)
            ),
            tributary_flow_proxy_exponent=float(
                matching_config.get("tributary_flow_proxy_exponent", 0.72)
            ),
            major_tributary_flow_fraction=float(
                matching_config.get("major_tributary_flow_fraction", 0.25)
            ),
            side_tributary_upstream_cutoff_m=float(
                matching_config.get("side_tributary_upstream_cutoff_m", 10000)
            ),
            candidate_direction=str(
                matching_config.get("candidate_direction", "either_direction")
            ),
            candidate_support_window_months=int(
                matching_config.get("antecedent_window_months", 6)
            ),
            min_wq_sampling_dates=int(
                matching_config.get("min_wq_samples_per_year", 3)
            ),
            min_wq_target_determinands=int(
                matching_config.get("min_wq_target_determinands", 3)
            ),
            max_acceptable_snap_distance_m=float(
                matching_config.get("max_acceptable_snap_distance_m", 250)
            ),
            allow_proxy_minor_tributaries=bool(
                matching_config.get("allow_proxy_minor_tributaries", False)
            ),
            allow_downstream_wq=bool(
                matching_config.get("allow_downstream_wq", True)
            ),
            target_determinand_complete_from_year=exact_determinand_complete_from_year,
            enforce_target_determinand_completeness=bool(
                enforce_target_determinand_completeness
            ),
        )
        network_outputs = build_network_integrated_evidence(network_settings)
        _record(
            stage_dir / "network_integration.json", name="network_integration", status="complete",
            inputs=[ecology_outputs["site_year"], edm_outputs["asset_year"], discharge_csv, root / "data/raw/water_quality", root / "data/raw/river_structure/IRN_river_line.csv"],
            outputs=list(network_outputs.values()),
        )
    else:
        network_outputs = network_expected

    return network_outputs, discharge_csv
# Add the supporting flow, habitat, discharge and regulatory information after
# the core biology-to-chemistry matches have been made.
def _build_context_stages(
    *,
    root: Path,
    data_dir: Path,
    stage_dir: Path,
    years: list[int],
    matching_config: dict[str, Any],
    ecology_outputs: dict[str, Path],
    network_outputs: dict[str, Path],
    discharge_csv: Path,
    refresh_flow: bool,
) -> tuple[Path, dict[str, Path]]:
    """Build survey chemistry features and the wider flow/context evidence."""
    chemistry_survey_path = data_dir / "chemistry/antecedent_chemistry_survey_evidence.parquet"
    chemistry_survey_path.parent.mkdir(parents=True, exist_ok=True)
    if not chemistry_survey_path.exists():
        chemistry_survey = build_antecedent_chemistry_features_from_path(
            pd.read_parquet(ecology_outputs["samples"]),
            pd.read_parquet(network_outputs["selected_matches"]),
            network_outputs["observations"],
            window_months=int(matching_config.get("antecedent_window_months", 6)),
            assemblages=None,
            years=years,
        )
        chemistry_survey.to_parquet(chemistry_survey_path, index=False)
    _record(
        stage_dir / "chemistry_survey.json", name="chemistry_survey", status="complete",
        inputs=[ecology_outputs["samples"], network_outputs["selected_matches"], network_outputs["observations"]],
        outputs=[chemistry_survey_path],
    )

    context_dir = data_dir / "context_integration"
    context_expected = {
        "analysis": context_dir / "river_health_analysis_site_year.parquet",
        "flow_features": context_dir / "biology_site_year_nrfa_flow_features.parquet",
        "ranked_gauges": context_dir / "ranked_biology_nrfa_gauges.parquet",
        "selected_gauges": context_dir / "selected_biology_nrfa_gauges_by_year.parquet",
        "daily_flows": root / "data/raw/nrfa/nrfa_gdf_matched_stations.parquet",
        "match_decisions": context_dir / "biology_chemistry_match_decisions.parquet",
        "discharge_context": context_dir / "selected_match_discharge_context.parquet",
        "flow_proxy_validation": context_dir / "tributary_flow_proxy_validation.json",
        "flow_proxy_gauges": context_dir / "tributary_flow_proxy_nrfa_gauges.parquet",
        "qa": context_dir / "context_integration_qa.json",
    }
    if refresh_flow or not _completed(stage_dir / "context_integration.json", context_expected):
        context_outputs = build_context_integrated_analysis(
            integrated_path=network_outputs["integrated"], selected_match_path=network_outputs["selected_matches"],
            sensitivity_path=network_outputs["sensitivity"], biology_snap_path=network_outputs["biology_snaps"],
            graph_path=network_outputs["graph"], irn_raw_csv=root / "data/raw/river_structure/IRN_river_line.csv",
            nrfa_raw_dir=root / "data/raw/nrfa",
            discharge_csv=discharge_csv, output_dir=context_dir, years=years, refresh_flow=refresh_flow,
            min_wq_sampling_dates=int(matching_config.get("min_wq_samples_per_year", 3)),
            min_wq_target_determinands=int(matching_config.get("min_wq_target_determinands", 3)),
            max_acceptable_snap_distance_m=float(
                matching_config.get("max_acceptable_snap_distance_m", 250)
            ),
            tributary_flow_proxy_exponent=float(
                matching_config.get("tributary_flow_proxy_exponent", 0.72)
            ),
            side_tributary_upstream_cutoff_m=float(
                matching_config.get("side_tributary_upstream_cutoff_m", 10000)
            ),
            allow_proxy_minor_tributaries=bool(
                matching_config.get("allow_proxy_minor_tributaries", False)
            ),
            allow_downstream_wq=bool(
                matching_config.get("allow_downstream_wq", True)
            ),
        )
        _record(
            stage_dir / "context_integration.json", name="context_integration", status="complete",
            inputs=[network_outputs["integrated"], network_outputs["selected_matches"], root / "data/raw/nrfa", discharge_csv],
            outputs=list(context_outputs.values()),
        )
    else:
        context_outputs = context_expected

    return chemistry_survey_path, context_outputs
# Bring the cleaned sources and network results together before writing the two
# user-facing versions of the dataset.
def _build_product_stage(
    *,
    root: Path,
    run_dir: Path,
    data_dir: Path,
    stage_dir: Path,
    years: list[int],
    run_config: dict[str, Any],
    flow_config: dict[str, Any],
    public_extension_start_year: int | None,
    chemistry_survey_path: Path,
    context_outputs: dict[str, Path],
    ecology_outputs: dict[str, Path],
    network_outputs: dict[str, Path],
    edm_outputs: dict[str, Path],
    rsn_outputs: dict[str, Path],
    rhs_output: Path,
) -> _ProductStageOutputs:
    """Build the integrated product and its temporal/source QA tables."""
    configured_extension_start = int(
        public_extension_start_year
        if public_extension_start_year is not None
        else run_config.get("public_extension_start_year", 2018)
    )
    extension_start_year = configured_extension_start
    if extension_start_year not in years:
        raise ValueError(
            "public_extension_start_year must be included in the requested years: "
            f"{extension_start_year} not in {min(years)}-{max(years)}"
        )
    chemistry_survey_all = pd.read_parquet(chemistry_survey_path)
    product_dir = data_dir / "connected_river_product"
    product_dir.mkdir(parents=True, exist_ok=True)
    product_path = product_dir / "connected_river_integrated_data.parquet"
    selected_gauges = pd.read_parquet(context_outputs["selected_gauges"])
    selected_nrfa_ids = sorted(
        pd.to_numeric(selected_gauges["nrfa_station_id"], errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    reference_start = pd.Timestamp(flow_config.get("reference_start", "1974-05-20"))
    reference_end = pd.Timestamp(flow_config.get("reference_end", "2019-03-29"))
    flow_read_end = max(reference_end, pd.Timestamp(f"{max(years)}-12-31"))
    product_daily_flows = pd.read_parquet(
        context_outputs["daily_flows"],
        filters=[
            ("nrfa_station_id", "in", selected_nrfa_ids),
            ("date", ">=", reference_start),
            ("date", "<=", flow_read_end),
        ],
        columns=["nrfa_station_id", "date", "flow_m3s", "quality_flag"],
    )
    joined_product = build_connected_river_product(
        chemistry_survey_all,
        selected_gauges,
        product_daily_flows,
        pd.read_parquet(context_outputs["analysis"]),
        automated_start_year=extension_start_year,
        flow_reference_start=reference_start,
        flow_reference_end=reference_end,
        minimum_flow_reference_coverage=float(
            flow_config.get("minimum_reference_coverage", 0.5)
        ),
    )
    habitat_context_path = product_dir / "habitat_context_ledger.parquet"
    build_habitat_context_ledger(
        joined_product,
        pd.read_parquet(rhs_output),
    ).to_parquet(habitat_context_path, index=False)
    expected_public_years = [int(value) for value in sorted(
        set(pd.to_numeric(chemistry_survey_all["sample_year"], errors="coerce").dropna().astype(int))
        & {int(year) for year in years if int(year) >= extension_start_year}
    )]
    automated_rows = joined_product
    observed_public_years = [int(value) for value in sorted(
        pd.to_numeric(automated_rows["sample_year"], errors="coerce").dropna().astype(int).unique()
    )]
    missing_public_years = sorted(set(expected_public_years) - set(observed_public_years))
    public_coverage_qa_path = run_dir / "qa/public_temporal_coverage.json"
    public_coverage_qa_path.parent.mkdir(parents=True, exist_ok=True)
    public_coverage_qa_path.write_text(json.dumps({
        "extension_start_year": extension_start_year,
        "expected_years_with_biology": expected_public_years,
        "observed_automated_product_years": observed_public_years,
        "missing_years": missing_public_years,
        "automated_rows_by_year": {
            str(int(year)): int(count)
            for year, count in automated_rows.groupby("sample_year").size().items()
        },
    }, indent=2), encoding="utf-8")
    if missing_public_years:
        raise ValueError(
            "Public temporal coverage QA failed; biological survey years were lost: "
            f"{missing_public_years}"
        )
    joined_product.to_parquet(product_path, index=False)
    _record(
        stage_dir / "connected_river_product.json", name="connected_river_product", status="complete",
        inputs=[chemistry_survey_path, context_outputs["selected_gauges"], context_outputs["daily_flows"], context_outputs["analysis"]],
        outputs=[
            product_path, habitat_context_path,
            public_coverage_qa_path,
        ],
    )

    site_year = pd.read_parquet(ecology_outputs["site_year"])
    trends_path = build_repeated_site_trends(site_year, data_dir / "trends/repeated_site_observation_trends.parquet")
    _record(stage_dir / "trends.json", name="trends", status="complete", inputs=[ecology_outputs["site_year"]], outputs=[trends_path])
    qa_path = write_qa(validate_site_year(site_year), run_dir / "qa/ecology_spine_qa.json")
    _record(stage_dir / "qa.json", name="qa", status="complete", inputs=[ecology_outputs["site_year"]], outputs=[qa_path])

    coverage_path = build_source_temporal_coverage(
        ecology_samples_path=ecology_outputs["samples"],
        water_quality_observations_path=network_outputs["observations"],
        nrfa_daily_flows_path=context_outputs["daily_flows"],
        edm_asset_year_path=edm_outputs["asset_year"],
        rsn_site_year_path=rsn_outputs["site_year_evidence"],
        output_path=data_dir / "source_coverage/source_temporal_coverage.csv",
    )

    return _ProductStageOutputs(
        joined_product=joined_product,
        product_path=product_path,
        habitat_context_path=habitat_context_path,
        trends_path=trends_path,
        ecology_qa_path=qa_path,
        source_coverage_path=coverage_path,
        public_coverage_qa_path=public_coverage_qa_path,
        expected_public_years=expected_public_years,
    )
def _write_release_stage(
    *,
    root: Path,
    run_dir: Path,
    stage_dir: Path,
    release_dir: Path,
    years: list[int],
    run_config: dict[str, Any],
    exact_determinand_complete_from_year: int | None,
    ecology_outputs: dict[str, Path],
    edm_outputs: dict[str, Path],
    rsn_outputs: dict[str, Path],
    wfd_outputs: dict[str, Path],
    rhs_output: Path,
    network_outputs: dict[str, Path],
    context_outputs: dict[str, Path],
    products: _ProductStageOutputs,
) -> dict[str, Any]:
    """Assemble the documented release directory and run its deep QA."""
    joined_product = products.joined_product
    product_path = products.product_path
    habitat_context_path = products.habitat_context_path
    trends_path = products.trends_path
    qa_path = products.ecology_qa_path
    coverage_path = products.source_coverage_path
    public_coverage_qa_path = products.public_coverage_qa_path
    expected_public_years = products.expected_public_years

    canonical = release_dir / "river_health_site_year_evidence.parquet"
    shutil.copy2(ecology_outputs["site_year"], canonical)
    shutil.copy2(edm_outputs["asset_year"], release_dir / "storm_overflow_asset_year.parquet")
    shutil.copy2(edm_outputs["inventory"], release_dir / "edm_source_inventory.parquet")
    shutil.copy2(rsn_outputs["site_year_evidence"], release_dir / "rsn_integrated_site_year_evidence.parquet")
    shutil.copy2(wfd_outputs["latest_classifications"], release_dir / "wfd_latest_river_classifications.parquet")
    shutil.copy2(wfd_outputs["status_summary"], release_dir / "wfd_river_status_summary.parquet")
    shutil.copy2(wfd_outputs["pressure_summary"], release_dir / "wfd_river_pressure_summary.parquet")
    shutil.copy2(wfd_outputs["objectives"], release_dir / "wfd_river_objectives.parquet")
    shutil.copy2(wfd_outputs["protected_areas"], release_dir / "wfd_protected_areas.parquet")
    waterbody_dossier = build_waterbody_evidence_dossier(
        joined_product,
        status=pd.read_parquet(wfd_outputs["status_summary"]),
        pressures=pd.read_parquet(wfd_outputs["pressure_summary"]),
        protected_areas=pd.read_parquet(wfd_outputs["protected_areas"]),
        objectives=pd.read_parquet(wfd_outputs["objectives"]),
        rsn_evidence=pd.read_parquet(rsn_outputs["site_year_evidence"]),
    )
    waterbody_dossier.to_parquet(
        release_dir / "waterbody_evidence_dossier.parquet", index=False
    )
    shutil.copy2(rhs_output, release_dir / "river_habitat_surveys.parquet")
    shutil.copy2(network_outputs["integrated"], release_dir / "river_connected_site_year_evidence.parquet")
    shutil.copy2(network_outputs["ranked_matches"], release_dir / "ranked_biology_chemistry_matches.parquet")
    shutil.copy2(network_outputs["selected_matches"], release_dir / "selected_biology_chemistry_matches.parquet")
    shutil.copy2(network_outputs["path_edm"], release_dir / "selected_match_route_edm.parquet")
    shutil.copy2(network_outputs["qa"], release_dir / "network_integration_qa.json")
    build_river_connection_route_asset(
        ranked_path=network_outputs["ranked_matches"],
        biology_snap_path=network_outputs["biology_snaps"],
        water_quality_snap_path=network_outputs["wq_snaps"],
        graph_path=network_outputs["graph"],
        irn_raw_csv=root / "data/raw/river_structure/IRN_river_line.csv",
        output_path=release_dir / ROUTE_ASSET_NAME,
        years=expected_public_years,
    )
    shutil.copy2(context_outputs["analysis"], release_dir / "river_health_analysis_site_year.parquet")
    shutil.copy2(product_path, release_dir / "connected_river_integrated_data.parquet")
    shutil.copy2(habitat_context_path, release_dir / "habitat_context_ledger.parquet")
    acquisition_coverage, _ = _write_target_determinand_acquisition(
        network_outputs["inventory"],
        release_dir,
        range(int(run_config.get("online_matching_start_year", min(years))), max(years) + 1),
        required_from_year=exact_determinand_complete_from_year,
    )
    _write_public_data_products(
        joined_product, release_dir, acquisition_coverage=acquisition_coverage,
    )
    write_chemistry_coverage_outputs(
        release_dir / "connected_river_integrated_data.parquet",
        release_dir,
        biology_snaps_path=network_outputs["biology_snaps"],
    )
    _write_hydrological_assignments(joined_product, root, release_dir)
    shutil.copy2(context_outputs["flow_features"], release_dir / "biology_site_year_nrfa_flow_features.parquet")
    shutil.copy2(context_outputs["ranked_gauges"], release_dir / "ranked_biology_nrfa_gauges.parquet")
    shutil.copy2(context_outputs["selected_gauges"], release_dir / "selected_biology_nrfa_gauges_by_year.parquet")
    shutil.copy2(context_outputs["match_decisions"], release_dir / "biology_chemistry_match_decisions.parquet")
    shutil.copy2(context_outputs["discharge_context"], release_dir / "selected_match_discharge_context.parquet")
    shutil.copy2(
        context_outputs["flow_proxy_validation"],
        release_dir / "tributary_flow_proxy_validation.json",
    )
    shutil.copy2(
        context_outputs["flow_proxy_gauges"],
        release_dir / "tributary_flow_proxy_nrfa_gauges.parquet",
    )
    shutil.copy2(context_outputs["qa"], release_dir / "context_integration_qa.json")
    shutil.copy2(trends_path, release_dir / "repeated_site_observation_trends.parquet")
    shutil.copy2(coverage_path, release_dir / "source_temporal_coverage.csv")
    shutil.copy2(public_coverage_qa_path, release_dir / "public_temporal_coverage_qa.json")
    _freeze_optional_release_assets(root, release_dir, run_dir=run_dir)
    (release_dir / "README.md").write_text(
        "# Connected River release\n\n"
        "This release contains only records downloaded and processed by this pipeline. "
        "The cited study motivated the research problem and default six-month "
        "window, but its deposited data are not included.\n\n"
        "## Main products\n\n"
        "- `connected_river_analysis_ready.csv.zip`: compact survey-level table for reuse.\n"
        "- `connected_river_integrated_data.parquet`: lossless specialist table with biology, chemistry, flow, route and matching diagnostics.\n"
        "- `connected_river_data_dictionary.csv`: field definitions, units and lineage.\n"
        "- `ranked_biology_chemistry_matches.parquet`: every candidate route and score.\n"
        "- `selected_biology_chemistry_matches.parquet`: selected routes and evidence gates.\n"
        "- `target_determinand_coverage.csv`: observed target-chemistry coverage.\n"
        "- `analysis_ready_qa.json`, `deep_release_qa.json` and `public_temporal_coverage_qa.json`: release checks.\n\n"
        "Rows that fail or require review retain their decision details, while their chemistry is excluded "
        "from accepted analytical fields. Network connections provide local river context; "
        "they do not prove identical water or causal exposure.\n\n"
        "## Redistribution\n\n"
        "Project QA does not override source licences. Review `THIRD_PARTY_DATA.md` and "
        "the saved source manifest before publishing spatial derivatives.\n",
        encoding="utf-8",
    )
    deep_qa = _run_deep_release_audit(run_dir)
    _record(
        stage_dir / "release.json", name="release", status="complete",
        inputs=[ecology_outputs["site_year"], edm_outputs["asset_year"], rsn_outputs["site_year_evidence"], wfd_outputs["latest_classifications"], qa_path],
        outputs=sorted(release_dir.iterdir()),
    )
    return deep_qa
# Run the stages in order and reuse completed ones when their inputs have not
# changed. This is what makes a long national build practical to resume.
def build_release(
    *, profile: str = "connected_river", run_root: str | Path = "runs",
    raw_ecology_dir: str | Path | None = None, resume_run: str | Path | None = None,
    years: list[int] | tuple[int, ...] | None = None,
    public_extension_start_year: int | None = None,
    refresh_flow: bool = False,
    promote: bool = False,
) -> Path:
    root = project_root()
    config = load_profile(profile)
    run_config = config.setdefault("run", {})
    matching_config = config.setdefault("matching", {})
    flow_config = config.setdefault("flow", {})
    if years is None:
        online_start = int(run_config.get("online_matching_start_year", 2018))
        target_year = int(run_config.get("target_year", datetime.now().year - 1))
        years = list(range(online_start, target_year + 1))
    else:
        years = [int(year) for year in years]
    if not years:
        raise ValueError("At least one online matching year is required")
    exact_determinand_complete_from_year = run_config.get(
        "exact_determinand_complete_from_year"
    )
    enforce_target_determinand_completeness = run_config.get(
        "enforce_target_determinand_completeness", False
    )
    if exact_determinand_complete_from_year is not None:
        exact_determinand_complete_from_year = int(
            exact_determinand_complete_from_year
        )
    # Record the expanded year set in run_context.json, rather than leaving the
    # exact release boundary implicit in two configuration endpoints.
    run_config["years"] = years
    if public_extension_start_year is not None:
        run_config["public_extension_start_year"] = int(public_extension_start_year)
    if resume_run:
        candidate = Path(resume_run)
        run_dir = candidate if candidate.is_absolute() else root / Path(run_root) / candidate
        if not (run_dir / "run_context.json").exists():
            raise FileNotFoundError(f"Cannot resume: run context not found in {run_dir}")
        if (run_dir / "SUCCESS").exists():
            if promote:
                _assert_run_context_current(
                    run_dir,
                    profile=profile,
                    config=config,
                    root=root,
                )
                deep_qa_path = run_dir / "qa/deep_release_qa.json"
                try:
                    deep_qa = json.loads(deep_qa_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    raise RuntimeError(
                        f"Completed run has no readable deep QA: {deep_qa_path}"
                    ) from exc
                promotion_gates = _evaluate_candidate_gates(
                    root=root,
                    run_dir=run_dir,
                    deep_qa=deep_qa,
                    config=config,
                )
                _write_candidate_manifest(
                    root=root,
                    run_dir=run_dir,
                    deep_qa=deep_qa,
                    promotion_gates=promotion_gates,
                )
                _promote_release(
                    root=root,
                    run_dir=run_dir,
                    deep_qa=deep_qa,
                    promotion_gates=promotion_gates,
                )
            return run_dir
        _assert_run_context_current(
            run_dir,
            profile=profile,
            config=config,
            root=root,
        )
    else:
        context = create_run_context(profile, config, root=run_root)
        run_dir = Path(context["run_dir"])
    stage_dir, data_dir, release_dir = run_dir / "stages", run_dir / "data", run_dir / "release"
    for directory in (stage_dir, data_dir, release_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (
        ecology_outputs,
        edm_outputs,
        rsn_outputs,
        wfd_outputs,
        rhs_output,
    ) = _build_source_stages(
        root=root,
        data_dir=data_dir,
        stage_dir=stage_dir,
        years=years,
        raw_ecology_dir=raw_ecology_dir,
    )

    network_outputs, discharge_csv = _build_network_stage(
        root=root,
        data_dir=data_dir,
        stage_dir=stage_dir,
        years=years,
        matching_config=matching_config,
        ecology_outputs=ecology_outputs,
        edm_outputs=edm_outputs,
        wfd_outputs=wfd_outputs,
        exact_determinand_complete_from_year=exact_determinand_complete_from_year,
        enforce_target_determinand_completeness=bool(
            enforce_target_determinand_completeness
        ),
    )

    chemistry_survey_path, context_outputs = _build_context_stages(
        root=root,
        data_dir=data_dir,
        stage_dir=stage_dir,
        years=years,
        matching_config=matching_config,
        ecology_outputs=ecology_outputs,
        network_outputs=network_outputs,
        discharge_csv=discharge_csv,
        refresh_flow=refresh_flow,
    )

    products = _build_product_stage(
        root=root,
        run_dir=run_dir,
        data_dir=data_dir,
        stage_dir=stage_dir,
        years=years,
        run_config=run_config,
        flow_config=flow_config,
        public_extension_start_year=public_extension_start_year,
        chemistry_survey_path=chemistry_survey_path,
        context_outputs=context_outputs,
        ecology_outputs=ecology_outputs,
        network_outputs=network_outputs,
        edm_outputs=edm_outputs,
        rsn_outputs=rsn_outputs,
        rhs_output=rhs_output,
    )
    deep_qa = _write_release_stage(
        root=root,
        run_dir=run_dir,
        stage_dir=stage_dir,
        release_dir=release_dir,
        years=years,
        run_config=run_config,
        exact_determinand_complete_from_year=exact_determinand_complete_from_year,
        ecology_outputs=ecology_outputs,
        edm_outputs=edm_outputs,
        rsn_outputs=rsn_outputs,
        wfd_outputs=wfd_outputs,
        rhs_output=rhs_output,
        network_outputs=network_outputs,
        context_outputs=context_outputs,
        products=products,
    )
    # Check that code, methods and configuration have not changed during a
    # long-running build before publishing its outputs.
    _assert_run_context_current(
        run_dir,
        profile=profile,
        config=config,
        root=root,
    )
    (run_dir / "SUCCESS").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    promotion_gates = _evaluate_candidate_gates(
        root=root, run_dir=run_dir, deep_qa=deep_qa, config=config,
    )
    _write_candidate_manifest(
        root=root, run_dir=run_dir, deep_qa=deep_qa,
        promotion_gates=promotion_gates,
    )
    if promote:
        _promote_release(
            root=root, run_dir=run_dir, deep_qa=deep_qa,
            promotion_gates=promotion_gates,
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a connected-river release.")
    parser.add_argument("--profile", default="connected_river")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--raw-ecology-dir")
    parser.add_argument("--resume-run", help="Run id or path of an incomplete run to resume")
    parser.add_argument("--years", type=int, nargs="+", help="Override profile years, including a newly available current year")
    parser.add_argument("--public-extension-start-year", type=int)
    parser.add_argument("--refresh-flow", action="store_true", help="Renew the annual quality-controlled NRFA records before matching")
    parser.add_argument(
        "--promote", action="store_true",
        help="Publish the completed outputs only when all saved checks pass",
    )
    args = parser.parse_args()
    print(build_release(
        profile=args.profile, run_root=args.run_root,
        raw_ecology_dir=args.raw_ecology_dir, resume_run=args.resume_run,
        years=args.years,
        public_extension_start_year=args.public_extension_start_year,
        refresh_flow=args.refresh_flow,
        promote=args.promote,
    ))


if __name__ == "__main__":
    main()
