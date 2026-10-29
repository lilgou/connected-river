"""Build the river-connected biology, chemistry and sewage evidence product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from connected_river.ingest.discharge_context import build_discharge_context
from connected_river.ingest.water_quality import build_water_quality_index
from connected_river.matching.annual_matching import scientific_sensitivity_summary
from connected_river.matching.irn_matching import (
    attribute_edm_to_selected_paths,
    attribute_static_discharges_to_selected_paths,
    build_directed_candidates,
)
from connected_river.matching.tributary_influence import graph_has_tributary_attributes
from connected_river.spatial.grid_reference import ngr_to_easting_northing
from connected_river.spatial.irn import (
    build_irn_topology,
    load_irn_graph,
    snap_points_to_irn_csv,
)
from connected_river.validation.match_decisions import (
    classify_candidate_matches,
    select_candidates_by_scientific_gate,
)


@dataclass(frozen=True)
class NetworkIntegrationSettings:
    """Inputs and scientific choices for one network-integration run."""

    biology_site_year_path: str | Path
    edm_asset_year_path: str | Path
    water_quality_raw_dir: str | Path
    irn_raw_csv: str | Path
    output_dir: str | Path
    years: list[int] | tuple[int, ...]
    edm_snap_path: str | Path | None = None
    discharge_csv: str | Path | None = None
    wfd_status_path: str | Path | None = None
    wfd_pressure_path: str | Path | None = None
    max_snap_distance_m: float = 1000
    max_euclidean_distance_m: float = 15000
    max_network_distance_m: float = 10000
    tributary_penalty_points: float = 0.0
    tributary_flow_proxy_exponent: float = 0.72
    major_tributary_flow_fraction: float = 0.25
    tributary_transport_decay_scale_m: float | None = None
    side_tributary_upstream_cutoff_m: float = 10_000.0
    candidate_direction: str = "either_direction"
    candidate_support_window_months: int = 6
    min_wq_sampling_dates: int = 3
    min_wq_target_determinands: int = 3
    max_acceptable_snap_distance_m: float = 250.0
    allow_proxy_minor_tributaries: bool = False
    allow_downstream_wq: bool = True
    target_determinand_complete_from_year: int | None = None
    enforce_target_determinand_completeness: bool = True


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


# Candidate building only needs summaries, but the chosen station's individual
# samples are kept for the final time-window calculation.
def _retain_selected_station_observations(
    observations_path: str | Path, selected: pd.DataFrame,
) -> tuple[int, int]:
    """Compact the survey-window table to stations selected anywhere in the run."""
    observations_path = Path(observations_path)
    station_ids = selected[["wq_site_id"]].dropna().drop_duplicates().copy()
    station_ids["wq_site_id"] = station_ids["wq_site_id"].astype("string")
    if station_ids.empty:
        return 0, 0
    temporary = observations_path.with_name(
        f".{observations_path.stem}.selected.part{observations_path.suffix}"
    )
    connection = duckdb.connect()
    try:
        connection.register("selected_station_ids", station_ids)
        before = int(connection.sql(
            f"SELECT count(*) AS n FROM read_parquet('{_sql_path(observations_path)}')"
        ).fetchone()[0])
        connection.execute(f"""
            COPY (
                SELECT observations.*
                FROM read_parquet('{_sql_path(observations_path)}') AS observations
                SEMI JOIN selected_station_ids
                  ON observations.wq_site_id = selected_station_ids.wq_site_id
            ) TO '{_sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        after = int(connection.sql(
            f"SELECT count(*) AS n FROM read_parquet('{_sql_path(temporary)}')"
        ).fetchone()[0])
        temporary.replace(observations_path)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)
    return before, after


def _snap_summary(table: pd.DataFrame, expected: int) -> dict[str, object]:
    distances = pd.to_numeric(table.get("snap_distance_m"), errors="coerce")
    snapped = int(distances.notna().sum())
    return {
        "expected_points": int(expected), "snapped_points": snapped,
        "snap_rate": snapped / expected if expected else None,
        "median_snap_distance_m": float(distances.median()) if snapped else None,
        "p95_snap_distance_m": float(distances.quantile(0.95)) if snapped else None,
    }


# This function joins the main network steps in the same order used in the
# notebook: snap sites, build candidates, rank them, then attach route context.
def build_network_integrated_evidence(
    settings: NetworkIntegrationSettings | None = None,
    **legacy_settings: Any,
) -> dict[str, Path]:
    """Match sites in both river directions and attach routed EDM information."""
    if settings is None:
        settings = NetworkIntegrationSettings(**legacy_settings)
    elif legacy_settings:
        raise TypeError("Pass a settings object or keyword settings, not both")

    biology_site_year_path = settings.biology_site_year_path
    edm_asset_year_path = settings.edm_asset_year_path
    water_quality_raw_dir = settings.water_quality_raw_dir
    irn_raw_csv = settings.irn_raw_csv
    output_dir = settings.output_dir
    years = settings.years
    edm_snap_path = settings.edm_snap_path
    discharge_csv = settings.discharge_csv
    wfd_status_path = settings.wfd_status_path
    wfd_pressure_path = settings.wfd_pressure_path
    max_snap_distance_m = settings.max_snap_distance_m
    max_euclidean_distance_m = settings.max_euclidean_distance_m
    max_network_distance_m = settings.max_network_distance_m
    tributary_penalty_points = settings.tributary_penalty_points
    tributary_flow_proxy_exponent = settings.tributary_flow_proxy_exponent
    major_tributary_flow_fraction = settings.major_tributary_flow_fraction
    tributary_transport_decay_scale_m = settings.tributary_transport_decay_scale_m
    side_tributary_upstream_cutoff_m = settings.side_tributary_upstream_cutoff_m
    candidate_direction = settings.candidate_direction
    candidate_support_window_months = settings.candidate_support_window_months
    min_wq_sampling_dates = settings.min_wq_sampling_dates
    min_wq_target_determinands = settings.min_wq_target_determinands
    max_acceptable_snap_distance_m = settings.max_acceptable_snap_distance_m
    allow_proxy_minor_tributaries = settings.allow_proxy_minor_tributaries
    allow_downstream_wq = settings.allow_downstream_wq
    target_determinand_complete_from_year = settings.target_determinand_complete_from_year
    enforce_target_determinand_completeness = settings.enforce_target_determinand_completeness

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The index retains the preceding year in its observation table while
    # limiting candidate support to the requested integration years.
    wq_paths = build_water_quality_index(
        water_quality_raw_dir,
        output_dir / "water_quality",
        years,
        require_target_determinand_completeness=enforce_target_determinand_completeness,
        target_determinand_completeness_years=range(
            int(target_determinand_complete_from_year)
            if target_determinand_complete_from_year is not None else min(years) - 1,
            max(years) + 1,
        ),
    )

    topology_dir = output_dir / "network"
    topology_paths = {
        "edges": topology_dir / "irn_edges.parquet", "nodes": topology_dir / "irn_nodes.parquet",
        "graph": topology_dir / "irn_graph.pkl",
    }
    if not all(path.exists() for path in topology_paths.values()):
        topology_paths = build_irn_topology(irn_raw_csv, topology_dir)
    graph = load_irn_graph(topology_paths["graph"])
    if not graph_has_tributary_attributes(graph):
        topology_paths = build_irn_topology(irn_raw_csv, topology_dir)
        graph = load_irn_graph(topology_paths["graph"])

    biology = pd.read_parquet(biology_site_year_path)
    biology = biology[biology["sample_year"].isin(years)].copy()
    biology["bio_site_id"] = biology["assemblage"].astype(str) + ":" + biology["site_id"].astype(str)
    biology["year"] = pd.to_numeric(biology["sample_year"], errors="coerce").astype("Int64")
    bio_points = biology.groupby("bio_site_id", as_index=False).agg(easting=("easting", "median"), northing=("northing", "median"))

    wq = pd.read_parquet(wq_paths["site_year"])
    wq_points = wq.groupby("wq_site_id", as_index=False).agg(easting=("easting", "median"), northing=("northing", "median"))
    bio_snap_path = output_dir / "biology_site_irn_snaps.parquet"
    wq_snap_path = output_dir / "water_quality_site_irn_snaps.parquet"
    bio_snaps = snap_points_to_irn_csv(
        bio_points,
        irn_raw_csv,
        point_id_col="bio_site_id",
        max_distance_m=max_snap_distance_m,
        graph=graph,
    )
    wq_snaps = snap_points_to_irn_csv(
        wq_points,
        irn_raw_csv,
        point_id_col="wq_site_id",
        max_distance_m=max_snap_distance_m,
        graph=graph,
    )
    bio_snaps.to_parquet(bio_snap_path, index=False)
    wq_snaps.to_parquet(wq_snap_path, index=False)

    ranked = build_directed_candidates(
        biology[[
            "bio_site_id", "year", "easting", "northing", "first_sample_date_in_year",
        ]], bio_snaps, wq, wq_snaps, graph,
        max_euclidean_m=max_euclidean_distance_m,
        max_network_m=max_network_distance_m,
        water_quality_observations=wq_paths["observations"],
        tributary_penalty_points=tributary_penalty_points,
        tributary_flow_proxy_exponent=tributary_flow_proxy_exponent,
        major_tributary_flow_fraction=major_tributary_flow_fraction,
        tributary_transport_decay_scale_m=tributary_transport_decay_scale_m,
        transport_plausibility_decay_scale_m=(
            float(tributary_transport_decay_scale_m)
            if tributary_transport_decay_scale_m is not None
            else 10000.0
        ),
        candidate_direction=candidate_direction,
        candidate_support_window_months=candidate_support_window_months,
    )
    if ranked.empty:
        raise RuntimeError("No directed biology-chemistry candidates were produced")
    edm = pd.read_parquet(edm_asset_year_path)
    if edm_snap_path is not None and Path(edm_snap_path).exists():
        edm_snaps = pd.read_parquet(edm_snap_path)
    else:
        edm_points = edm[["asset_id", "outlet_ngr"]].drop_duplicates("asset_id").copy()
        coordinates = edm_points["outlet_ngr"].map(ngr_to_easting_northing)
        edm_points["easting"] = coordinates.map(lambda value: value[0] if value else pd.NA)
        edm_points["northing"] = coordinates.map(lambda value: value[1] if value else pd.NA)
        edm_snaps = snap_points_to_irn_csv(
            edm_points, irn_raw_csv, point_id_col="asset_id", max_distance_m=max_snap_distance_m,
            graph=graph,
        )
        edm_snap_path = output_dir / "edm_asset_irn_snaps.parquet"
        edm_snaps.to_parquet(edm_snap_path, index=False)
    candidate_edm = attribute_edm_to_selected_paths(
        ranked,
        edm,
        edm_snaps,
        graph,
        side_tributary_upstream_cutoff_m=side_tributary_upstream_cutoff_m,
    )
    ranked = ranked.merge(
        candidate_edm,
        on=["bio_site_id", "year", "wq_site_id"],
        how="left",
        validate="one_to_one",
    )

    candidate_discharge = pd.DataFrame()
    if discharge_csv is not None:
        candidate_discharge_source_path = output_dir / "sewage_discharge_context.parquet"
        build_discharge_context(discharge_csv, candidate_discharge_source_path)
        discharges = pd.read_parquet(candidate_discharge_source_path)
        discharge_snaps = snap_points_to_irn_csv(
            discharges[["discharge_id", "easting", "northing"]],
            irn_raw_csv,
            point_id_col="discharge_id",
            max_distance_m=max_snap_distance_m,
            graph=graph,
        )
        discharge_snaps.to_parquet(
            output_dir / "sewage_discharge_irn_snaps.parquet", index=False
        )
        candidate_discharge = attribute_static_discharges_to_selected_paths(
            ranked,
            discharges,
            discharge_snaps,
            graph,
            side_tributary_upstream_cutoff_m=side_tributary_upstream_cutoff_m,
        )
        ranked = ranked.merge(
            candidate_discharge,
            on=["bio_site_id", "year", "wq_site_id"],
            how="left",
            validate="one_to_one",
        )

    ranked = classify_candidate_matches(
        ranked,
        min_sampling_dates=min_wq_sampling_dates,
        min_target_determinands=min_wq_target_determinands,
        max_snap_distance_m=max_acceptable_snap_distance_m,
        allow_proxy_minor_tributaries=allow_proxy_minor_tributaries,
        allow_downstream_wq=allow_downstream_wq,
    )
    sensitivity = scientific_sensitivity_summary(ranked)
    selected = select_candidates_by_scientific_gate(ranked)

    ranked_path = output_dir / "ranked_biology_chemistry_matches.parquet"
    selected_path = output_dir / "selected_biology_chemistry_matches.parquet"
    sensitivity_path = output_dir / "match_distance_sensitivity.parquet"
    ranked.to_parquet(ranked_path, index=False)
    selected.to_parquet(selected_path, index=False)
    sensitivity.to_parquet(sensitivity_path, index=False)

    selected_keys = selected[["bio_site_id", "year", "wq_site_id"]]
    path_edm = candidate_edm.merge(
        selected_keys,
        on=["bio_site_id", "year", "wq_site_id"],
        how="inner",
        validate="one_to_one",
    )
    path_edm_path = output_dir / "selected_match_route_edm.parquet"
    path_edm.to_parquet(path_edm_path, index=False)

    observation_rows_before, observation_rows_after = _retain_selected_station_observations(
        wq_paths["observations"], selected,
    )

    features = pd.read_parquet(wq_paths["features"])
    integrated = biology.merge(selected, on=["bio_site_id", "year"], how="left", validate="one_to_one")
    integrated = integrated.merge(features, on=["wq_site_id", "year"], how="left", validate="many_to_one")
    integrated["chemistry_temporal_method"] = (
        "six_months_through_first_site_year_survey"
        if candidate_support_window_months == 6
        else f"{candidate_support_window_months}_calendar_months_through_first_site_year_survey"
    )
    integrated["chemistry_window_months"] = int(candidate_support_window_months)
    integrated["chemistry_below_loq_treatment"] = "reported_limit_value"
    for optional_path in (wfd_status_path, wfd_pressure_path):
        if optional_path is None or not Path(optional_path).exists():
            continue
        context = pd.read_parquet(optional_path).rename(columns={"Water Body ID": "wfd_waterbody_id"})
        integrated = integrated.merge(context, on="wfd_waterbody_id", how="left", validate="many_to_one")
    integrated_path = output_dir / "river_connected_site_year_evidence.parquet"
    integrated.to_parquet(integrated_path, index=False)

    selected_confidence = selected["selection_confidence"].value_counts(dropna=False).to_dict()
    selected_penalties = pd.to_numeric(
        selected.get("score_tributary_penalty", pd.Series(index=selected.index, dtype=float)),
        errors="coerce",
    )
    selected_represented_flow = pd.to_numeric(
        selected.get(
            "estimated_flow_represented_fraction",
            pd.Series(index=selected.index, dtype=float),
        ),
        errors="coerce",
    )
    selected_largest_side_flow = pd.to_numeric(
        selected.get(
            "largest_side_tributary_estimated_flow_fraction_at_biology",
            pd.Series(index=selected.index, dtype=float),
        ),
        errors="coerce",
    )
    selected_major = selected.get(
        "major_tributary_between_sites", pd.Series(False, index=selected.index)
    ).fillna(False).astype(bool)
    selected_intervening = pd.to_numeric(
        selected.get("intervening_tributary_count", pd.Series(0, index=selected.index)),
        errors="coerce",
    ).fillna(0).gt(0)
    selected_changed = selected.get(
        "winner_changed_by_tributary", pd.Series(False, index=selected.index)
    ).fillna(False).astype(bool)
    selected_influence_classes = selected.get(
        "tributary_influence_class", pd.Series("unavailable", index=selected.index)
    ).fillna("unavailable").astype(str).value_counts(dropna=False)
    selected_wq_link_resolution = selected.get(
        "tributary_wq_link_resolution",
        pd.Series("unavailable", index=selected.index),
    ).fillna("unavailable").astype(str).value_counts(dropna=False)
    selected_bio_link_resolution = selected.get(
        "tributary_bio_link_resolution",
        pd.Series("unavailable", index=selected.index),
    ).fillna("unavailable").astype(str).value_counts(dropna=False)
    selected_parallel_alias = selected.get(
        "tributary_parallel_alias_resolved", pd.Series(False, index=selected.index)
    ).fillna(False).astype(bool)
    qa = {
        "biology_snapping": _snap_summary(bio_snaps, len(bio_points)),
        "water_quality_snapping": _snap_summary(wq_snaps, len(wq_points)),
        "edm_snapping": _snap_summary(edm_snaps, edm["asset_id"].nunique()),
        "candidate_rows": len(ranked), "selected_rows": len(selected),
        "biology_site_year_rows": len(biology),
        "water_quality_observation_rows_before_selected_station_filter": observation_rows_before,
        "water_quality_observation_rows_for_selected_stations": observation_rows_after,
        "irn_topology": {
            "graph_schema_version": graph.graph.get("irn_graph_schema_version"),
            "raw_link_count": graph.graph.get("raw_link_count"),
            "collapsed_edge_count": graph.graph.get("collapsed_edge_count"),
            "parallel_edge_group_count": graph.graph.get("parallel_edge_group_count"),
            "parallel_link_alias_count": graph.graph.get("parallel_link_alias_count"),
            "parallel_hydrology_differs_from_primary_count": graph.graph.get(
                "parallel_hydrology_differs_from_primary_count"
            ),
            "node_continuity_tolerance_m": graph.graph.get(
                "node_continuity_tolerance_m"
            ),
            "spatially_split_raw_node_count": graph.graph.get(
                "spatially_split_raw_node_count"
            ),
            "spatially_split_endpoint_count": graph.graph.get(
                "spatially_split_endpoint_count"
            ),
            "spatial_surrogate_node_count": graph.graph.get(
                "spatial_surrogate_node_count"
            ),
            "node_identity_policy": (
                "raw node IDs are retained when all represented line endpoints are "
                "within 25 m; geographically separate reuse of one raw ID is split "
                "into separate graph nodes to prevent zero-distance jumps"
            ),
            "parallel_edge_policy": (
                "shortest raw link supplies routing length; every raw link remains an "
                "recorded alias; the highest positive Shreve alias supplies internal "
                "hydrology without summing parallel channels"
            ),
        },
        "matched_site_year_rate": float(len(selected) / len(biology)) if len(biology) else None,
        "route_modifiers_screened_before_candidate_selection": bool(
            discharge_csv is not None
        ),
        "selection_confidence": {str(key): int(value) for key, value in selected_confidence.items()},
        "network_distance_median": float(pd.to_numeric(selected["network_distance_m"], errors="coerce").median()),
        "network_distance_p95": float(pd.to_numeric(selected["network_distance_m"], errors="coerce").quantile(0.95)),
        "selected_tributary_influence": {
            "score_tributary_penalty_median": float(selected_penalties.median()),
            "score_tributary_penalty_p95": float(selected_penalties.quantile(0.95)),
            "estimated_flow_represented_fraction_median": float(
                selected_represented_flow.median()
            ),
            "largest_side_tributary_estimated_flow_fraction_median": float(
                selected_largest_side_flow.median()
            ),
            "largest_side_tributary_estimated_flow_fraction_p95": float(
                selected_largest_side_flow.quantile(0.95)
            ),
            "routes_with_intervening_tributaries": int(selected_intervening.sum()),
            "routes_with_major_intervening_tributary": int(selected_major.sum()),
            "tributary_influence_class": {
                str(key): int(value) for key, value in selected_influence_classes.items()
            },
            "wq_link_resolution": {
                str(key): int(value) for key, value in selected_wq_link_resolution.items()
            },
            "biology_link_resolution": {
                str(key): int(value) for key, value in selected_bio_link_resolution.items()
            },
            "routes_resolved_through_parallel_link_alias": int(
                selected_parallel_alias.sum()
            ),
            "winners_changed_by_tributary_penalty": int(selected_changed.sum()),
            "winners_changed_by_tributary_penalty_percent": (
                float(100.0 * selected_changed.mean()) if len(selected_changed) else None
            ),
        },
        "matching_parameters": {
            "max_snap_distance_m": float(max_snap_distance_m),
            "max_euclidean_prefilter_m": float(max_euclidean_distance_m),
            "max_network_distance_m": float(max_network_distance_m),
            "direction_rule": candidate_direction,
            "candidate_support_window_months": int(candidate_support_window_months),
            "scientific_candidate_selection": (
                "attach available route evidence to every candidate; choose the best accepted "
                "candidate; keep review or rejected fallbacks only for checking"
            ),
            "score_role": (
                "ordinal ranking inside the scientific decision pool; explicit evidence gates, "
                "not score class, determine eligibility"
            ),
            "allow_proxy_minor_tributaries_for_analysis": bool(
                allow_proxy_minor_tributaries
            ),
            "allow_downstream_water_quality_for_analysis": bool(allow_downstream_wq),
            "tributary_penalty_points": float(tributary_penalty_points),
            "tributary_penalty_method": (
                "configured points multiplied by unrepresented calibrated-flow proxy; "
                "raw Shreve fraction is the transparent fallback"
            ),
            "tributary_flow_proxy_exponent": float(tributary_flow_proxy_exponent),
            "tributary_flow_proxy_method": (
                "compositional confluence branch weights from Shreve raised to the configured "
                "exponent; calibrated proxy, not observed tributary flow"
            ),
            "tributary_transport_decay_scale_m": (
                None if tributary_transport_decay_scale_m is None
                else float(tributary_transport_decay_scale_m)
            ),
            "major_tributary_flow_fraction": float(major_tributary_flow_fraction),
            "side_tributary_upstream_cutoff_m": float(side_tributary_upstream_cutoff_m),
            "side_tributary_attribution_method": (
                "cached bounded reverse-upstream traversal from intervening tributary mouths"
            ),
        },
    }
    qa_path = output_dir / "network_integration_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {
        **wq_paths, **topology_paths, "biology_snaps": bio_snap_path, "wq_snaps": wq_snap_path,
        "edm_snaps": Path(edm_snap_path),
        "ranked_matches": ranked_path, "selected_matches": selected_path, "sensitivity": sensitivity_path,
        "path_edm": path_edm_path, "integrated": integrated_path, "qa": qa_path,
    }
