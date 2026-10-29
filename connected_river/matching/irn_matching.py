"""Directed IRN candidate construction and storm-overflow path attribution."""

from __future__ import annotations

import ast
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from connected_river.matching.annual_matching import rank_annual_matches
from connected_river.matching.tributary_influence import candidate_tributary_metrics
from connected_river.spatial.irn import (
    edge_attributes_for_link,
    edge_link_ids,
    normalise_irn_link_id,
    same_irn_link_id,
)

CORE_CHEMISTRY = {
    "ammoniacal_n_mg_l", "nitrate_n_mg_l", "orthophosphate_p_mg_l",
    "dissolved_oxygen_mg_l", "dissolved_oxygen_percent", "ph",
}
METALS = {"zinc", "copper", "cadmium", "lead", "nickel", "chromium", "mercury", "arsenic"}
SIDE_TRIBUTARY_UPSTREAM_CUTOFF_M = 10_000.0
CANDIDATE_DIRECTIONS = {
    "upstream_only",
    "either_direction",
}


def _actual_site_endpoint_metrics(metrics: dict[str, object]) -> dict[str, object]:
    """Restore WQ/biology endpoint labels after checking a biology-to-WQ route.

    ``candidate_tributary_metrics`` follows physical flow from an upstream
    endpoint to a downstream endpoint.  For a downstream WQ candidate those
    physical roles are the reverse of the site types, so endpoint-specific
    attributes must be swapped back before they are written to the candidate
    table.  The neutral represented/unrepresented fractions deliberately keep
    their physical upstream-to-downstream meaning.
    """
    out = dict(metrics)
    for suffix in ("strahler", "shreve"):
        wq_key = f"wq_{suffix}"
        bio_key = f"bio_{suffix}"
        out[wq_key], out[bio_key] = out.get(bio_key), out.get(wq_key)
    wq_resolution = out.get("tributary_wq_link_resolution")
    out["tributary_wq_link_resolution"] = out.get("tributary_bio_link_resolution")
    out["tributary_bio_link_resolution"] = wq_resolution
    return out


def _antecedent_candidate_support(
    candidates: pd.DataFrame,
    observations: pd.DataFrame | str | Path,
    *,
    window_months: int = 6,
) -> pd.DataFrame:
    """Replace calendar-year support with survey-antecedent support.

    Ranking must not benefit from samples collected after the biological
    survey. The default site-year match is therefore evaluated using the
    configured calendar months ending at that site's first survey of the year.
    """
    if isinstance(window_months, bool) or int(window_months) != window_months:
        raise ValueError("window_months must be a whole number")
    window_months = int(window_months)
    if not 1 <= window_months <= 12:
        raise ValueError(
            "window_months must be between 1 and 12; longer windows require an "
            "required source-download settings"
        )
    out = candidates.copy()
    keys = ["bio_site_id", "year", "wq_site_id"]
    windows = out[keys + ["survey_date"]].drop_duplicates().copy()
    windows["survey_date"] = pd.to_datetime(windows["survey_date"], errors="coerce")
    connection = duckdb.connect()
    try:
        connection.register("candidate_windows", windows)
        if isinstance(observations, pd.DataFrame):
            connection.register("wq_observations", observations)
            source = "wq_observations"
            observation_columns = set(observations.columns)
        else:
            observation_path = Path(observations)
            source_path = str(observation_path.resolve()).replace("'", "''")
            source = f"read_parquet('{source_path}')"
            observation_columns = set(pq.read_schema(observation_path).names)
        matching_join = (
            "AND observations.is_matching_determinand"
            if "is_matching_determinand" in observation_columns else ""
        )
        core_values = ", ".join(f"'{value}'" for value in sorted(CORE_CHEMISTRY))
        metal_values = ", ".join(f"'{value}'" for value in sorted(METALS))
        support = connection.sql(f"""
            SELECT
                windows.bio_site_id,
                windows.year,
                windows.wq_site_id,
                count(DISTINCT CAST(observations.sample_datetime AS DATE)) AS dates,
                count(DISTINCT date_trunc('month', observations.sample_datetime)) AS months,
                count(DISTINCT observations.determinand_key) AS determinands,
                coalesce(bool_or(observations.determinand_key IN ({core_values})), false) AS core,
                coalesce(bool_or(observations.determinand_key IN ({metal_values})), false) AS metals
            FROM candidate_windows AS windows
            LEFT JOIN {source} AS observations
             ON observations.wq_site_id = windows.wq_site_id
             AND CAST(observations.sample_datetime AS DATE)
                 BETWEEN CAST(windows.survey_date AS DATE) - INTERVAL '{window_months} months'
                     AND CAST(windows.survey_date AS DATE)
             {matching_join}
            GROUP BY windows.bio_site_id, windows.year, windows.wq_site_id
        """).df()
    finally:
        connection.close()
    support = support.rename(columns={
        "dates": "_window_dates",
        "months": "_window_months",
        "determinands": "_window_determinands",
        "core": "_window_core",
        "metals": "_window_metals",
    })
    out = out.merge(support, on=keys, how="left", validate="many_to_one")
    replacements = {
        "wq_n_sampling_dates": "_window_dates",
        "wq_n_months_sampled": "_window_months",
        "wq_n_target_determinands": "_window_determinands",
        "wq_has_core_chemistry": "_window_core",
        "wq_has_metals": "_window_metals",
    }
    for target, source in replacements.items():
        out[target] = out[source].fillna(0 if target.startswith("wq_n_") else False)
    out["candidate_support_window_start"] = (
        pd.to_datetime(out["survey_date"], errors="coerce")
        - pd.DateOffset(months=window_months)
    )
    out["candidate_support_window_end"] = pd.to_datetime(out["survey_date"], errors="coerce")
    out["candidate_support_window_months"] = window_months
    out["candidate_support_temporal_method"] = (
        "six_months_through_first_survey"
        if window_months == 6
        else f"{window_months}_calendar_months_through_first_survey"
    )
    return out.drop(columns=list(replacements.values()))


# Start with a quick straight-line search, then use the directed river graph to
# keep only sites that are genuinely connected along the network.
def build_directed_candidates(
    biology: pd.DataFrame, biology_snaps: pd.DataFrame, water_quality: pd.DataFrame,
    wq_snaps: pd.DataFrame, graph: nx.DiGraph, *, max_euclidean_m: float = 15000,
    max_network_m: float = 10000,
    water_quality_observations: pd.DataFrame | str | Path | None = None,
    tributary_penalty_points: float = 0.0,
    tributary_flow_proxy_exponent: float = 0.72,
    major_tributary_flow_fraction: float = 0.25,
    tributary_transport_decay_scale_m: float | None = None,
    transport_plausibility_decay_scale_m: float = 10000.0,
    tributary_observed_flow_by_link_id: dict[object, object] | None = None,
    candidate_direction: str = "either_direction",
    candidate_support_window_months: int = 6,
) -> pd.DataFrame:
    if candidate_direction not in CANDIDATE_DIRECTIONS:
        allowed = ", ".join(sorted(CANDIDATE_DIRECTIONS))
        raise ValueError(f"candidate_direction must be one of: {allowed}")
    bio = biology.merge(biology_snaps, on="bio_site_id", how="inner")
    wq = water_quality.merge(wq_snaps, on="wq_site_id", how="inner")
    bio = bio[bio["snapped_link_id"].notna()].copy()
    wq = wq[wq["snapped_link_id"].notna()].copy()
    rows = []
    reverse_graph = graph.reverse(copy=False)
    for year in sorted(set(bio["year"].dropna()) & set(wq["year"].dropna())):
        b = bio[bio["year"].eq(year)].reset_index(drop=True)
        q = wq[wq["year"].eq(year)].reset_index(drop=True)
        if b.empty or q.empty:
            continue
        tree = cKDTree(q[["easting", "northing"]].astype(float).to_numpy())
        nearby = tree.query_ball_point(b[["easting", "northing"]].astype(float).to_numpy(), max_euclidean_m)
        pairs_by_wq: dict[int, list[int]] = {}
        for bio_index, candidates in enumerate(nearby):
            for wq_index in candidates:
                pairs_by_wq.setdefault(int(wq_index), []).append(int(bio_index))
        for wq_index, bio_indices in pairs_by_wq.items():
            qrow = q.iloc[wq_index]
            source = int(qrow["snapped_end_node"])
            distances, paths = (
                nx.single_source_dijkstra(
                    graph, source, cutoff=max_network_m, weight="length_m"
                )
                if source in graph
                else ({}, {})
            )
            # A reverse search is only needed when downstream chemistry sites
            # are allowed as local river context.
            reverse_distances, reverse_paths = (
                nx.single_source_dijkstra(
                    reverse_graph,
                    int(qrow["snapped_start_node"]),
                    cutoff=max_network_m,
                    weight="length_m",
                )
                if (
                    candidate_direction == "either_direction"
                    and int(qrow["snapped_start_node"]) in reverse_graph
                )
                else ({}, {})
            )
            for bio_index in bio_indices:
                brow = b.iloc[bio_index]
                same_link = same_irn_link_id(
                    qrow["snapped_link_id"], brow["snapped_link_id"],
                )
                if same_link and float(qrow["position_on_link_m"]) <= float(brow["position_on_link_m"]):
                    network_distance = float(brow["position_on_link_m"]) - float(qrow["position_on_link_m"])
                    relationship = "same_link"
                    network_direction = "wq_upstream_of_bio"
                    metric_relationship = "same_link"
                    route_start_node = int(qrow["snapped_end_node"])
                    route_end_node = int(brow["snapped_start_node"])
                    route_origin_site_type = "water_quality"
                    route_destination_site_type = "biology"
                elif same_link and candidate_direction == "either_direction":
                    network_distance = float(qrow["position_on_link_m"]) - float(
                        brow["position_on_link_m"]
                    )
                    relationship = "same_link"
                    network_direction = "wq_downstream_of_bio"
                    metric_relationship = "same_link"
                    route_start_node = int(brow["snapped_end_node"])
                    route_end_node = int(qrow["snapped_start_node"])
                    route_origin_site_type = "biology"
                    route_destination_site_type = "water_quality"
                elif int(brow["snapped_start_node"]) in distances:
                    network_distance = (
                        float(qrow["snapped_link_length_m"]) - float(qrow["position_on_link_m"])
                        + float(distances[int(brow["snapped_start_node"])])
                        + float(brow["position_on_link_m"])
                    )
                    relationship = "wq_upstream_of_bio"
                    network_direction = relationship
                    metric_relationship = relationship
                    route_start_node = int(qrow["snapped_end_node"])
                    route_end_node = int(brow["snapped_start_node"])
                    route_origin_site_type = "water_quality"
                    route_destination_site_type = "biology"
                elif (
                    candidate_direction == "either_direction"
                    and int(brow["snapped_end_node"]) in reverse_distances
                ):
                    network_distance = (
                        float(brow["snapped_link_length_m"])
                        - float(brow["position_on_link_m"])
                        + float(reverse_distances[int(brow["snapped_end_node"])])
                        + float(qrow["position_on_link_m"])
                    )
                    relationship = "wq_downstream_of_bio"
                    network_direction = relationship
                    metric_relationship = "wq_upstream_of_bio"
                    route_start_node = int(brow["snapped_end_node"])
                    route_end_node = int(qrow["snapped_start_node"])
                    route_origin_site_type = "biology"
                    route_destination_site_type = "water_quality"
                else:
                    continue
                if network_distance > max_network_m:
                    continue
                if relationship == "wq_upstream_of_bio":
                    path_nodes = tuple(paths[int(brow["snapped_start_node"])])
                elif relationship == "wq_downstream_of_bio":
                    path_nodes = tuple(
                        reversed(reverse_paths[int(brow["snapped_end_node"])])
                    )
                else:
                    path_nodes = (int(qrow["snapped_end_node"]),)
                downstream_wq = network_direction == "wq_downstream_of_bio"
                tributary_metrics = candidate_tributary_metrics(
                    graph,
                    relationship=metric_relationship,
                    path_nodes=path_nodes,
                    wq_link_id=(
                        brow["snapped_link_id"] if downstream_wq else qrow["snapped_link_id"]
                    ),
                    bio_link_id=(
                        qrow["snapped_link_id"] if downstream_wq else brow["snapped_link_id"]
                    ),
                    bio_link_position_m=float(
                        qrow["position_on_link_m"] if downstream_wq else brow["position_on_link_m"]
                    ),
                    flow_proxy_exponent=tributary_flow_proxy_exponent,
                    major_tributary_flow_fraction=major_tributary_flow_fraction,
                    transport_decay_scale_m=tributary_transport_decay_scale_m,
                    tributary_observed_flow_by_link_id=tributary_observed_flow_by_link_id,
                )
                if downstream_wq:
                    tributary_metrics = _actual_site_endpoint_metrics(tributary_metrics)
                row = {
                    "bio_site_id": brow["bio_site_id"], "year": year, "wq_site_id": qrow["wq_site_id"],
                    "survey_date": brow.get("first_sample_date_in_year", pd.NaT),
                    "network_distance_m": network_distance,
                    "network_relationship": relationship,
                    "network_direction": network_direction,
                    "candidate_direction_rule": candidate_direction,
                    "scientific_evidence_role": (
                        "downstream_river_chemistry_context"
                        if downstream_wq
                        else "upstream_river_chemistry_context"
                    ),
                    "route_start_node": route_start_node,
                    "route_end_node": route_end_node,
                    "route_origin_site_type": route_origin_site_type,
                    "route_destination_site_type": route_destination_site_type,
                    "tributary_influence_reference_site_type": route_destination_site_type,
                    "bio_snap_distance_m": brow["snap_distance_m"], "wq_snap_distance_m": qrow["snap_distance_m"],
                    "wq_n_sampling_dates": qrow.get("n_sampling_dates", 0), "wq_n_months_sampled": qrow.get("n_months_sampled", 0),
                    "wq_n_target_determinands": qrow.get("n_target_determinands", 0),
                    "wq_has_core_chemistry": qrow.get("has_core_chemistry", False), "wq_has_metals": qrow.get("has_metals", False),
                    "prefilter_euclidean_distance_m": float(np.hypot(float(brow["easting"]) - float(qrow["easting"]), float(brow["northing"]) - float(qrow["northing"]))),
                    "wq_start_node": qrow["snapped_end_node"], "bio_end_node": brow["snapped_start_node"],
                    "wq_link_id": qrow["snapped_link_id"], "bio_link_id": brow["snapped_link_id"],
                }
                row.update(tributary_metrics)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    candidates = pd.DataFrame(rows)
    if water_quality_observations is not None:
        candidates = _antecedent_candidate_support(
            candidates,
            water_quality_observations,
            window_months=candidate_support_window_months,
        )
    return rank_annual_matches(
        candidates,
        tributary_penalty_points=tributary_penalty_points,
        transport_plausibility_decay_scale_m=transport_plausibility_decay_scale_m,
    )


def _coerce_link_ids(value: object) -> tuple[int | str, ...]:
    """Normalise in-memory and Parquet list representations of tributary links."""
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in stripped.split(";") if part.strip()]
        value = parsed
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    normalised = (normalise_irn_link_id(item) for item in value)
    return tuple(dict.fromkeys(item for item in normalised if item is not None))


def _graph_link_index(graph: nx.DiGraph) -> dict[int | str, tuple[object, object, dict]]:
    """Index every raw link, including aliases retained on collapsed edges."""
    index: dict[int | str, tuple[object, object, dict]] = {}
    for upstream, downstream, data in graph.edges(data=True):
        for raw_link_id in edge_link_ids(data):
            link_id = normalise_irn_link_id(raw_link_id)
            resolved = edge_attributes_for_link(data, raw_link_id)
            if link_id is not None and resolved is not None:
                index.setdefault(link_id, (upstream, downstream, resolved))
    return index


def _bounded_side_tributary_links(
    graph: nx.DiGraph,
    tributary_link_ids: object,
    *,
    exact_path_links: set[int | str],
    link_index: dict[int | str, tuple[object, object, dict]],
    cache: dict[int | str, frozenset[int | str]],
    cutoff_m: float = SIDE_TRIBUTARY_UPSTREAM_CUTOFF_M,
) -> set[int | str]:
    """Return links intersecting the bounded upstream reach of tributary mouths.

    Each intervening tributary link fixes the branch to enter at the confluence,
    avoiding reverse traversal up the matched main-stem route. Results are cached
    by mouth link because many selected site-years reuse the same confluences.
    """
    side_links: set[int | str] = set()
    reverse = graph.reverse(copy=False)
    for tributary_link_id in _coerce_link_ids(tributary_link_ids):
        if tributary_link_id not in cache:
            edge = link_index.get(tributary_link_id)
            if edge is None:
                cache[tributary_link_id] = frozenset()
                continue
            upstream, _mouth, data = edge
            direct_length = pd.to_numeric(
                pd.Series([data.get("length_m")]), errors="coerce"
            ).iloc[0]
            direct_length = float(direct_length) if pd.notna(direct_length) else 0.0
            links = {
                normalised
                for alias_id in edge_link_ids(data)
                if (normalised := normalise_irn_link_id(alias_id)) is not None
            }
            remaining = max(0.0, float(cutoff_m) - max(0.0, direct_length))
            if upstream in reverse and remaining > 0:
                distances = nx.single_source_dijkstra_path_length(
                    reverse, upstream, cutoff=remaining, weight="length_m"
                )
                # An asset is snapped only to a link, not to a position on that
                # link. Include links that intersect the 10 km reach by entering
                # any node whose distance from the branch seed is within bounds.
                for node in distances:
                    for _edge_upstream, _node, edge_data in graph.in_edges(node, data=True):
                        links.update(
                            normalised
                            for alias_id in edge_link_ids(edge_data)
                            if (normalised := normalise_irn_link_id(alias_id)) is not None
                        )
            cache[tributary_link_id] = frozenset(links)
        side_links.update(cache[tributary_link_id])
    return side_links - exact_path_links


def _selected_path_links(graph: nx.DiGraph, match: object) -> set[int | str]:
    route_start = getattr(match, "route_start_node", match.wq_start_node)
    route_end = getattr(match, "route_end_node", match.bio_end_node)
    try:
        nodes = nx.shortest_path(
            graph, int(route_start), int(route_end), weight="length_m"
        )
        route_links = {
            normalise_irn_link_id(graph[u][v].get("link_id"))
            for u, v in zip(nodes[:-1], nodes[1:])
        }
        links = {link_id for link_id in route_links if link_id is not None}
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        links = set()
    links.update(
        link_id
        for link_id in (
            normalise_irn_link_id(match.wq_link_id),
            normalise_irn_link_id(match.bio_link_id),
        )
        if link_id is not None
    )
    return links


def _matched_path_cache_key(match: object) -> tuple[object, ...]:
    """Return the year-independent route identity used by context caches."""
    return (
        getattr(match, "route_start_node", getattr(match, "wq_start_node", None)),
        getattr(match, "route_end_node", getattr(match, "bio_end_node", None)),
        getattr(match, "wq_link_id", None),
        getattr(match, "bio_link_id", None),
    )


def _matched_context_cache_key(
    match: object, *, include_year: bool,
) -> tuple[object, ...]:
    """Return a hashable identity for repeated route-context calculations."""
    key = _matched_path_cache_key(match) + (
        repr(getattr(match, "intervening_tributary_link_ids", ())),
    )
    return (getattr(match, "year", None), *key) if include_year else key


# Once a match has been chosen, attach overflows on the actual route and on
# nearby tributaries separately. They answer different environmental questions.
def attribute_edm_to_selected_paths(
    selected: pd.DataFrame, edm_asset_year: pd.DataFrame, edm_snaps: pd.DataFrame, graph: nx.DiGraph,
    *, side_tributary_upstream_cutoff_m: float = SIDE_TRIBUTARY_UPSTREAM_CUTOFF_M,
) -> pd.DataFrame:
    edm = edm_asset_year.merge(edm_snaps, on="asset_id", how="inner")
    crosswalked = edm.get(
        "location_match_method", pd.Series("", index=edm.index)
    ).astype("string").str.startswith("exact_")
    edm["crosswalked_asset_id"] = edm["asset_id"].where(crosswalked)
    edm["reporting_year"] = pd.to_numeric(edm["reporting_year"], errors="coerce").astype("Int64")
    edm["snapped_link_id"] = pd.to_numeric(edm["snapped_link_id"], errors="coerce").astype("Int64")
    by_link = edm.groupby(["reporting_year", "snapped_link_id"], dropna=False).agg(
        asset_ids=("asset_id", lambda values: frozenset(values.dropna().astype(str))),
        crosswalked_asset_ids=(
            "crosswalked_asset_id",
            lambda values: frozenset(values.dropna().astype(str)),
        ),
        counted_spills=("counted_spills", lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)),
        spill_duration_hours=("spill_duration_hours", lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)),
    ).to_dict("index")
    available_years = set(
        edm.loc[edm["asset_id"].notna()]
        .groupby("reporting_year")["asset_id"]
        .nunique()
        .loc[lambda values: values.ge(100)]
        .index.astype(int)
    )
    link_index = _graph_link_index(graph)
    side_link_cache: dict[int | str, frozenset[int | str]] = {}
    path_link_cache: dict[tuple[object, ...], frozenset[int | str]] = {}
    evidence_cache: dict[tuple[object, ...], dict[str, object]] = {}
    rows = []
    for match in selected.itertuples(index=False):
        if int(match.year) not in available_years:
            rows.append({
                "bio_site_id": match.bio_site_id,
                "year": match.year,
                "wq_site_id": match.wq_site_id,
                "path_edm_assets": pd.NA,
                "path_edm_crosswalked_assets": pd.NA,
                "path_edm_location_provenance": "unavailable_no_national_return",
                "path_counted_spills": pd.NA,
                "path_spill_duration_hours": pd.NA,
                "path_attribution_method": "unavailable_no_national_return",
                "side_tributary_edm_assets": pd.NA,
                "side_tributary_edm_crosswalked_assets": pd.NA,
                "side_tributary_edm_location_provenance": (
                    "unavailable_no_national_return"
                ),
                "side_tributary_counted_spills": pd.NA,
                "side_tributary_spill_duration_hours": pd.NA,
                "side_tributary_attribution_method": "unavailable_no_national_return",
                "side_tributary_upstream_cutoff_m": side_tributary_upstream_cutoff_m,
            })
            continue
        path_key = _matched_path_cache_key(match)
        context_key = _matched_context_cache_key(match, include_year=True)
        if context_key in evidence_cache:
            rows.append({
                "bio_site_id": match.bio_site_id,
                "year": match.year,
                "wq_site_id": match.wq_site_id,
                **evidence_cache[context_key],
            })
            continue
        if path_key not in path_link_cache:
            path_link_cache[path_key] = frozenset(_selected_path_links(graph, match))
        links = set(path_link_cache[path_key])
        side_links = _bounded_side_tributary_links(
            graph,
            getattr(match, "intervening_tributary_link_ids", ()),
            exact_path_links=links,
            link_index=link_index,
            cache=side_link_cache,
            cutoff_m=side_tributary_upstream_cutoff_m,
        )
        evidence = [
            by_link[(int(match.year), int(link))]
            for link in links
            if (int(match.year), int(link)) in by_link
        ]
        side_evidence = [
            by_link[(int(match.year), int(link))]
            for link in side_links
            if (int(match.year), int(link)) in by_link
        ]
        asset_ids = set().union(*(item["asset_ids"] for item in evidence)) if evidence else set()
        crosswalked_asset_ids = (
            set().union(*(item["crosswalked_asset_ids"] for item in evidence))
            if evidence else set()
        )
        spill_counts = pd.Series([item["counted_spills"] for item in evidence], dtype="float64")
        durations = pd.Series([item["spill_duration_hours"] for item in evidence], dtype="float64")
        side_asset_ids = (
            set().union(*(item["asset_ids"] for item in side_evidence))
            if side_evidence
            else set()
        )
        side_crosswalked_asset_ids = (
            set().union(
                *(item["crosswalked_asset_ids"] for item in side_evidence)
            )
            if side_evidence else set()
        )
        side_spill_counts = pd.Series(
            [item["counted_spills"] for item in side_evidence], dtype="float64"
        )
        side_durations = pd.Series(
            [item["spill_duration_hours"] for item in side_evidence], dtype="float64"
        )
        route_context = {
            "path_edm_assets": len(asset_ids), "path_counted_spills": spill_counts.sum(min_count=1),
            "path_edm_crosswalked_assets": len(crosswalked_asset_ids),
            "path_edm_location_provenance": (
                "includes_exact_cross_year_location_crosswalk"
                if crosswalked_asset_ids else "publisher_reported_locations_only"
            ),
            "path_spill_duration_hours": durations.sum(min_count=1),
            "path_attribution_method": "directed_irn_selected_path",
            "side_tributary_edm_assets": len(side_asset_ids),
            "side_tributary_edm_crosswalked_assets": len(
                side_crosswalked_asset_ids
            ),
            "side_tributary_edm_location_provenance": (
                "includes_exact_cross_year_location_crosswalk"
                if side_crosswalked_asset_ids
                else "publisher_reported_locations_only"
            ),
            "side_tributary_counted_spills": side_spill_counts.sum(min_count=1),
            "side_tributary_spill_duration_hours": side_durations.sum(min_count=1),
            "side_tributary_attribution_method": (
                "bounded_reverse_upstream_from_intervening_tributary_mouths"
            ),
            "side_tributary_upstream_cutoff_m": side_tributary_upstream_cutoff_m,
        }
        evidence_cache[context_key] = route_context
        rows.append({
            "bio_site_id": match.bio_site_id,
            "year": match.year,
            "wq_site_id": match.wq_site_id,
            **route_context,
        })
    return pd.DataFrame(rows)


def attribute_static_discharges_to_selected_paths(
    selected: pd.DataFrame, discharges: pd.DataFrame, discharge_snaps: pd.DataFrame, graph: nx.DiGraph,
    *, side_tributary_upstream_cutoff_m: float = SIDE_TRIBUTARY_UPSTREAM_CUTOFF_M,
) -> pd.DataFrame:
    """Count treatment and permitted sewage outlets on each selected matching path."""
    points = discharges.merge(discharge_snaps, on="discharge_id", how="inner")
    points["snapped_link_id"] = pd.to_numeric(points["snapped_link_id"], errors="coerce").astype("Int64")
    by_link = points.groupby("snapped_link_id", dropna=False).agg(
        discharge_ids=("discharge_id", lambda values: frozenset(values.dropna().astype(str))),
        treatment_works=("is_treatment_works", "sum"), water_company_outlets=("is_water_company", "sum"),
    ).to_dict("index")
    link_index = _graph_link_index(graph)
    side_link_cache: dict[int | str, frozenset[int | str]] = {}
    path_link_cache: dict[tuple[object, ...], frozenset[int | str]] = {}
    evidence_cache: dict[tuple[object, ...], dict[str, object]] = {}
    rows = []
    for match in selected.itertuples(index=False):
        path_key = _matched_path_cache_key(match)
        context_key = _matched_context_cache_key(match, include_year=False)
        if context_key in evidence_cache:
            rows.append({
                "bio_site_id": match.bio_site_id,
                "year": match.year,
                "wq_site_id": match.wq_site_id,
                **evidence_cache[context_key],
            })
            continue
        if path_key not in path_link_cache:
            path_link_cache[path_key] = frozenset(_selected_path_links(graph, match))
        links = set(path_link_cache[path_key])
        side_links = _bounded_side_tributary_links(
            graph,
            getattr(match, "intervening_tributary_link_ids", ()),
            exact_path_links=links,
            link_index=link_index,
            cache=side_link_cache,
            cutoff_m=side_tributary_upstream_cutoff_m,
        )
        evidence = [by_link[int(link)] for link in links if int(link) in by_link]
        side_evidence = [by_link[int(link)] for link in side_links if int(link) in by_link]
        ids = set().union(*(item["discharge_ids"] for item in evidence)) if evidence else set()
        side_ids = (
            set().union(*(item["discharge_ids"] for item in side_evidence))
            if side_evidence
            else set()
        )
        route_context = {
            "path_permitted_sewage_outlets": len(ids),
            "path_treatment_works": int(sum(item["treatment_works"] for item in evidence)),
            "path_water_company_outlets": int(sum(item["water_company_outlets"] for item in evidence)),
            "discharge_context_method": "static_open_extract_on_directed_selected_path",
            "side_tributary_permitted_sewage_outlets": len(side_ids),
            "side_tributary_treatment_works": int(
                sum(item["treatment_works"] for item in side_evidence)
            ),
            "side_tributary_water_company_outlets": int(
                sum(item["water_company_outlets"] for item in side_evidence)
            ),
            "side_tributary_context_method": (
                "static_open_extract_bounded_reverse_upstream_from_tributary_mouths"
            ),
            "side_tributary_upstream_cutoff_m": side_tributary_upstream_cutoff_m,
        }
        evidence_cache[context_key] = route_context
        rows.append({
            "bio_site_id": match.bio_site_id,
            "year": match.year,
            "wq_site_id": match.wq_site_id,
            **route_context,
        })
    return pd.DataFrame(rows)
