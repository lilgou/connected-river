"""Write and validate the compact outputs presented in the notebook."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from connected_river.workflow import sha256_file


@dataclass(frozen=True)
class CompletedDataset:
    """Validated joined products and small tables used for presentation."""

    paths: dict[str, Path]
    qa: dict[str, object]
    parquet_file: pq.ParquetFile
    summary: pd.DataFrame
    preview: pd.DataFrame
    dictionary: pd.DataFrame
    manifest_path: Path


# Save the full candidate table as well as the winners. The extra rows are what
# make it possible to understand or rerun a station choice later.
def write_matching_outputs(
    *,
    workspace: str | Path,
    scored_candidates: pd.DataFrame,
    selected_matches: pd.DataFrame,
    filter_audit: dict[str, object],
    matching_options: dict[str, object],
    route_filters: list[str],
    custom_recipe: dict[str, object] | None,
    method_run: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    """Write candidate and selected-match files, with checksums."""

    output_directory = Path(workspace) / "user_matching_outputs"
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "ranked_candidate_routes.parquet": output_directory
        / "ranked_candidate_routes.parquet",
        "selected_matches.parquet": output_directory / "selected_matches.parquet",
        "selected_matches_summary.csv": output_directory
        / "selected_matches_summary.csv",
        "filter_audit.json": output_directory / "filter_audit.json",
        "matching_recipe.json": output_directory / "matching_recipe.json",
    }

    scored_candidates.to_parquet(paths["ranked_candidate_routes.parquet"], index=False)
    selected_matches.to_parquet(paths["selected_matches.parquet"], index=False)
    summary_columns = [
        column
        for column in (
            "bio_site_id",
            "year",
            "wq_site_id",
            "match_score",
            "network_distance_m",
            "network_direction",
            "candidate_gate_decision",
            "candidate_rejection_reasons",
            "candidate_review_reasons",
            "wq_n_sampling_dates",
            "wq_n_target_determinands",
            "wq_has_metals",
            "path_permitted_sewage_outlets",
            "side_tributary_permitted_sewage_outlets",
        )
        if column in selected_matches
    ]
    selected_matches[summary_columns].to_csv(
        paths["selected_matches_summary.csv"],
        index=False,
    )
    paths["filter_audit.json"].write_text(
        json.dumps(filter_audit, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["matching_recipe.json"].write_text(
        json.dumps(
            {
                "matching_options": matching_options,
                "candidate_route_filters": route_filters,
                "custom_match_recipe": custom_recipe,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        **method_run,
        "candidate_rows": len(scored_candidates),
        "selected_rows": len(selected_matches),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths.values()
        },
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_directory, manifest


# Gather the small set of checks a user is likely to want after a run: file
# paths, row counts, date coverage and whether the expected outputs exist.
def inspect_completed_dataset(
    release_directory: str | Path,
    output_directory: str | Path,
) -> CompletedDataset:
    """Check the joined output and return presentation-sized summaries."""

    release = Path(release_directory)
    outputs = Path(output_directory)
    paths = {
        "Analysis-ready CSV (recommended)": release
        / "connected_river_analysis_ready.csv.zip",
        "Specialist joined Parquet": release
        / "connected_river_integrated_data.parquet",
        "Data dictionary": release / "connected_river_data_dictionary.csv",
        "Output checks": release / "analysis_ready_qa.json",
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("The build did not produce: " + ", ".join(missing))

    qa = json.loads(paths["Output checks"].read_text(encoding="utf-8"))
    if qa.get("status") != "passed":
        raise RuntimeError("Analysis-ready output checks did not pass")
    if not qa.get("record_ids_identical"):
        raise RuntimeError("Compact and specialist record identities differ")
    if not qa.get("dictionary_columns_identical"):
        raise RuntimeError("The data dictionary does not cover the compact dataset")

    specialist = pq.ParquetFile(paths["Specialist joined Parquet"])
    if specialist.metadata.num_rows != int(qa["full_rows"]):
        raise RuntimeError("Specialist Parquet row count disagrees with the stored checks")
    with zipfile.ZipFile(paths["Analysis-ready CSV (recommended)"]) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
    if csv_members != ["connected_river_analysis_ready.csv"]:
        raise RuntimeError("Analysis-ready ZIP does not contain the expected CSV")

    summary = pd.DataFrame(
        [
            {
                "product": "Analysis-ready CSV",
                "rows": int(qa["analysis_ready_rows"]),
                "columns": int(qa["analysis_ready_columns"]),
                "file size (MB)": paths[
                    "Analysis-ready CSV (recommended)"
                ].stat().st_size
                / 1_000_000,
                "file": paths["Analysis-ready CSV (recommended)"].name,
            },
            {
                "product": "Specialist joined Parquet",
                "rows": specialist.metadata.num_rows,
                "columns": specialist.metadata.num_columns,
                "file size (MB)": paths[
                    "Specialist joined Parquet"
                ].stat().st_size
                / 1_000_000,
                "file": paths["Specialist joined Parquet"].name,
            },
        ]
    )
    preview_columns = [
        column
        for column in (
            "data_origin",
            "assemblage",
            "site_id",
            "sample_id",
            "sample_date",
            "sample_year",
            "taxon_richness",
            "wq_site_id",
            "match_decision",
            "match_use_recommended",
            "chemistry_included_in_release",
            "chemistry_release_status",
        )
        if column in specialist.schema_arrow.names
    ]
    all_preview_rows = pd.read_parquet(
        paths["Specialist joined Parquet"],
        columns=preview_columns,
    )
    preview = (
        all_preview_rows.loc[
            all_preview_rows["data_origin"].eq("connected_river_pipeline")
            & all_preview_rows["chemistry_included_in_release"].fillna(False)
        ]
        .groupby("assemblage", sort=True, observed=True)
        .head(1)
        .drop(columns="data_origin")
        .reset_index(drop=True)
    )
    data_dictionary = pd.read_csv(paths["Data dictionary"])
    manifest = {
        "release_directory": str(release),
        "qa_status": qa["status"],
        "rows": specialist.metadata.num_rows,
        "products": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for label, path in paths.items()
        },
    }
    manifest_path = outputs / "completed_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return CompletedDataset(
        paths,
        qa,
        specialist,
        summary,
        preview,
        data_dictionary,
        manifest_path,
    )
