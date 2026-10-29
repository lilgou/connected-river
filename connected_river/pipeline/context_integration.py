"""Integrate NRFA hydrology and static wastewater context into a release."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from connected_river.download.nrfa import (
    combine_nrfa_flow_partitions,
    download_nrfa_flows,
)
from connected_river.features.flow_features import build_antecedent_flow_features
from connected_river.features.model_table import build_analysis_table
from connected_river.ingest.discharge_context import build_discharge_context
from connected_river.matching.flow_matching import match_nrfa_gauges
from connected_river.matching.irn_matching import (
    attribute_static_discharges_to_selected_paths,
)
from connected_river.matching.year_aware_flow import (
    select_site_year_nrfa_gauges,
    standardise_daily_flows,
)
from connected_river.spatial.irn import load_irn_graph, snap_points_to_irn_csv
from connected_river.validation.match_decisions import classify_selected_matches
from connected_river.validation.tributary_flow_proxy import (
    validate_tributary_flow_proxy,
)


# Add supporting evidence to the matched survey rows without changing which
# chemistry station was selected. Matching and interpretation stay separate.
def build_context_integrated_analysis(
    *, integrated_path: str | Path, selected_match_path: str | Path, sensitivity_path: str | Path,
    biology_snap_path: str | Path, graph_path: str | Path, irn_raw_csv: str | Path,
    nrfa_raw_dir: str | Path, discharge_csv: str | Path, output_dir: str | Path,
    years: list[int] | tuple[int, ...] | None = None, refresh_flow: bool = False,
    min_wq_sampling_dates: int = 3, min_wq_target_determinands: int = 3,
    max_acceptable_snap_distance_m: float = 250,
    tributary_flow_proxy_exponent: float = 0.72,
    side_tributary_upstream_cutoff_m: float = 10_000.0,
    allow_proxy_minor_tributaries: bool = False,
    allow_downstream_wq: bool = True,
) -> dict[str, Path]:
    """Add quality-controlled NRFA flow and route context to the joined table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = load_irn_graph(graph_path)
    integrated = pd.read_parquet(integrated_path)
    bio_points = integrated.groupby("bio_site_id", as_index=False).agg(
        easting=("easting", "median"), northing=("northing", "median"),
    )
    bio_snaps = pd.read_parquet(biology_snap_path)

    nrfa_raw_dir = Path(nrfa_raw_dir)
    cached_station_ids = {
        int(re.search(r"(\d+)", path.stem).group(1))
        for path in (nrfa_raw_dir / "flows").glob("nrfa_gdf_*.parquet")
    }
    if not cached_station_ids and not refresh_flow:
        raise FileNotFoundError("No cached NRFA gauged daily-flow partitions are available")
    catalogue = pd.read_parquet(nrfa_raw_dir / "nrfa_station_catalogue.parquet")
    candidate_catalogue = catalogue if refresh_flow else catalogue[catalogue["id"].isin(cached_station_ids)]
    gauges = candidate_catalogue.rename(columns={"id": "nrfa_station_id"}).copy()
    catalogue_gauges = catalogue.rename(columns={"id": "nrfa_station_id"}).copy()
    gauge_snaps = snap_points_to_irn_csv(
        catalogue_gauges[["nrfa_station_id", "easting", "northing"]].dropna(), irn_raw_csv,
        point_id_col="nrfa_station_id", max_distance_m=1000, graph=graph,
    )
    gauge_snaps_path = output_dir / "nrfa_gauge_irn_snaps.parquet"
    gauge_snaps.to_parquet(gauge_snaps_path, index=False)
    flow_proxy_summary, flow_proxy_audit = validate_tributary_flow_proxy(
        gauge_snaps,
        catalogue,
        graph,
        flow_proxy_exponent=tributary_flow_proxy_exponent,
    )
    flow_proxy_summary_path = output_dir / "tributary_flow_proxy_validation.json"
    flow_proxy_summary_path.write_text(
        json.dumps(flow_proxy_summary, indent=2) + "\n", encoding="utf-8",
    )
    flow_proxy_audit_path = output_dir / "tributary_flow_proxy_nrfa_gauges.parquet"
    flow_proxy_audit.to_parquet(flow_proxy_audit_path, index=False)
    ranked_gauges = match_nrfa_gauges(
        bio_points, bio_snaps, gauges, gauge_snaps, graph,
        max_euclidean_m=30000, max_network_m=25000,
    )
    ranked_gauges_path = output_dir / "ranked_biology_nrfa_gauges.parquet"
    ranked_gauges.to_parquet(ranked_gauges_path, index=False)

    if refresh_flow:
        relevant_station_ids = sorted(
            pd.to_numeric(
                ranked_gauges.loc[ranked_gauges["gauge_rank"].le(3), "nrfa_station_id"], errors="coerce",
            ).dropna().astype(int).unique()
        )
        if not relevant_station_ids:
            raise RuntimeError("No connected NRFA candidate gauges were found for the current biological sites")
        download_nrfa_flows(relevant_station_ids, nrfa_raw_dir / "flows", overwrite=True, workers=10)
    combine_nrfa_flow_partitions(
        nrfa_raw_dir / "flows", nrfa_raw_dir / "nrfa_gdf_matched_stations.parquet",
    )

    flow_filters = None
    if years:
        flow_filters = [
            ("date", ">=", pd.Timestamp(f"{min(int(year) for year in years) - 1}-01-01")),
            ("date", "<=", pd.Timestamp(f"{max(int(year) for year in years)}-12-31")),
        ]
    daily_flows = standardise_daily_flows(pd.read_parquet(
        nrfa_raw_dir / "nrfa_gdf_matched_stations.parquet",
        filters=flow_filters,
    ))
    biology_flow_spine = integrated[["bio_site_id", "year", "first_sample_date_in_year"]]
    selected_gauges = select_site_year_nrfa_gauges(
        biology_flow_spine, ranked_gauges, daily_flows,
    )
    selected_gauges_path = output_dir / "selected_biology_nrfa_gauges_by_year.parquet"
    selected_gauges.to_parquet(selected_gauges_path, index=False)
    flow_features = build_antecedent_flow_features(biology_flow_spine, selected_gauges, daily_flows)
    flow_features_path = output_dir / "biology_site_year_nrfa_flow_features.parquet"
    flow_features.to_parquet(flow_features_path, index=False)

    discharges_path = output_dir / "sewage_discharge_context.parquet"
    build_discharge_context(discharge_csv, discharges_path)
    discharges = pd.read_parquet(discharges_path)
    discharge_snaps = snap_points_to_irn_csv(
        discharges[["discharge_id", "easting", "northing"]], irn_raw_csv,
        point_id_col="discharge_id", max_distance_m=1000, graph=graph,
    )
    discharge_snaps_path = output_dir / "sewage_discharge_irn_snaps.parquet"
    discharge_snaps.to_parquet(discharge_snaps_path, index=False)
    selected = pd.read_parquet(selected_match_path)
    discharge_context = attribute_static_discharges_to_selected_paths(
        selected,
        discharges,
        discharge_snaps,
        graph,
        side_tributary_upstream_cutoff_m=side_tributary_upstream_cutoff_m,
    )
    discharge_context_path = output_dir / "selected_match_discharge_context.parquet"
    discharge_context.to_parquet(discharge_context_path, index=False)

    route_context = discharge_context.copy()
    edm_context_columns = [
        column
        for column in (
            "bio_site_id", "year", "wq_site_id", "path_edm_assets",
            "path_counted_spills", "path_spill_duration_hours",
        )
        if column in integrated
    ]
    if {"bio_site_id", "year", "wq_site_id"} <= set(edm_context_columns):
        route_context = route_context.merge(
            integrated[edm_context_columns],
            on=["bio_site_id", "year", "wq_site_id"],
            how="left",
            validate="one_to_one",
        )
    decisions = classify_selected_matches(
        selected,
        pd.read_parquet(sensitivity_path),
        route_context=route_context,
        min_sampling_dates=min_wq_sampling_dates,
        min_target_determinands=min_wq_target_determinands,
        max_snap_distance_m=max_acceptable_snap_distance_m,
        allow_proxy_minor_tributaries=allow_proxy_minor_tributaries,
        allow_downstream_wq=allow_downstream_wq,
    )
    decisions_path = output_dir / "biology_chemistry_match_decisions.parquet"
    decisions.to_parquet(decisions_path, index=False)
    analysis = build_analysis_table(integrated, decisions, flow_features, discharge_context)
    analysis_path = output_dir / "river_health_analysis_site_year.parquet"
    analysis.to_parquet(analysis_path, index=False)

    qa = {
        "flow_source": "NRFA gauged daily flow",
        "nrfa_candidate_gauges": len(gauges),
        "nrfa_catalogue_gauges_snapped": int(gauge_snaps["snap_distance_m"].notna().sum()),
        "nrfa_candidate_gauges_snapped": int(
            gauge_snaps.loc[gauge_snaps["nrfa_station_id"].isin(gauges["nrfa_station_id"]), "snap_distance_m"]
            .notna()
            .sum()
        ),
        "biology_site_years_with_selected_nrfa_gauge": int(selected_gauges["flow_station_id"].notna().sum()),
        "site_years_with_30d_nrfa_flow": int(flow_features["flow_mean_30d_m3s"].notna().sum()),
        "nrfa_selection_reasons": {
            str(key): int(value) for key, value in selected_gauges["flow_selection_reason"].value_counts().items()
        },
        "river_sewage_outlets": len(discharges),
        "sewage_outlets_snapped": int(discharge_snaps["snap_distance_m"].notna().sum()),
        "matched_paths_with_treatment_works": int(discharge_context["path_treatment_works"].gt(0).sum()),
        "match_decisions": {str(key): int(value) for key, value in decisions["match_decision"].value_counts().items()},
        "match_acceptance_parameters": {
            "score_role": "ordinal ranking within the scientific decision pool; not an eligibility gate",
            "minimum_sampling_dates_per_site_year": int(min_wq_sampling_dates),
            "minimum_target_determinands_per_site_year": int(min_wq_target_determinands),
            "maximum_biology_and_chemistry_snap_distance_m": float(max_acceptable_snap_distance_m),
            "side_tributary_upstream_cutoff_m": float(side_tributary_upstream_cutoff_m),
            "allow_proxy_minor_tributaries_for_analysis": bool(
                allow_proxy_minor_tributaries
            ),
            "allow_downstream_water_quality_for_analysis": bool(allow_downstream_wq),
            "treatment_works_between_sites_policy": "reject",
            "other_permitted_or_spilling_sewage_outlet_policy": "review",
        },
        "tributary_flow_proxy_validation": flow_proxy_summary,
    }
    qa_path = output_dir / "context_integration_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {
        "gauge_snaps": gauge_snaps_path,
        "flow_proxy_validation": flow_proxy_summary_path,
        "flow_proxy_gauges": flow_proxy_audit_path,
        "ranked_gauges": ranked_gauges_path,
        "selected_gauges": selected_gauges_path,
        "flow_features": flow_features_path,
        "daily_flows": nrfa_raw_dir / "nrfa_gdf_matched_stations.parquet",
        "discharges": discharges_path,
        "discharge_snaps": discharge_snaps_path,
        "discharge_context": discharge_context_path,
        "match_decisions": decisions_path,
        "analysis": analysis_path,
        "qa": qa_path,
    }
