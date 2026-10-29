"""Connect NRFA gauges to biological sites using the river network."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# Search both directions along the river network. The relationship is retained
# because an upstream and downstream gauge do not describe the site equally.
def _match_flow_gauges(
    biology_points: pd.DataFrame, biology_snaps: pd.DataFrame,
    gauges: pd.DataFrame, gauge_snaps: pd.DataFrame, graph: nx.DiGraph,
    *, gauge_id_col: str, source: str, station_name_col: str, river_col: str,
    max_euclidean_m: float = 30000, max_network_m: float = 50000,
    allow_indirect: bool = False,
) -> pd.DataFrame:
    """Rank one provider's independently located gauges on the river network."""
    bio = biology_points.merge(biology_snaps, on="bio_site_id", how="inner")
    gauge = gauges.merge(gauge_snaps, on=gauge_id_col, how="inner")
    bio = bio[bio["snapped_link_id"].notna()].reset_index(drop=True)
    gauge = gauge[gauge["snapped_link_id"].notna()].reset_index(drop=True)
    tree = cKDTree(bio[["easting", "northing"]].astype(float).to_numpy())
    nearby = tree.query_ball_point(gauge[["easting", "northing"]].astype(float).to_numpy(), max_euclidean_m)
    undirected = graph.to_undirected(as_view=True)
    reverse = graph.reverse(copy=False)
    rows = []
    for gauge_index, bio_indices in enumerate(nearby):
        grow = gauge.iloc[gauge_index]
        g_start, g_end = int(grow["snapped_start_node"]), int(grow["snapped_end_node"])
        from_start = nx.single_source_dijkstra_path_length(undirected, g_start, cutoff=max_network_m, weight="length_m")
        from_end = nx.single_source_dijkstra_path_length(undirected, g_end, cutoff=max_network_m, weight="length_m")
        downstream = nx.single_source_dijkstra_path_length(graph, g_end, cutoff=max_network_m, weight="length_m")
        upstream = nx.single_source_dijkstra_path_length(reverse, g_start, cutoff=max_network_m, weight="length_m")
        g_pos = float(grow["position_on_link_m"])
        g_remaining = float(grow["snapped_link_length_m"]) - g_pos
        for bio_index in bio_indices:
            brow = bio.iloc[int(bio_index)]
            b_start, b_end = int(brow["snapped_start_node"]), int(brow["snapped_end_node"])
            b_pos = float(brow["position_on_link_m"])
            b_remaining = float(brow["snapped_link_length_m"]) - b_pos
            if str(grow["snapped_link_id"]) == str(brow["snapped_link_id"]):
                distance = abs(g_pos - b_pos)
                relationship = "gauge_upstream" if g_pos <= b_pos else "gauge_downstream"
            else:
                options = [
                    g_pos + from_start.get(b_start, np.inf) + b_pos,
                    g_pos + from_start.get(b_end, np.inf) + b_remaining,
                    g_remaining + from_end.get(b_start, np.inf) + b_pos,
                    g_remaining + from_end.get(b_end, np.inf) + b_remaining,
                ]
                distance = float(min(options))
                if b_start in downstream:
                    relationship = "gauge_upstream"
                elif b_end in upstream:
                    relationship = "gauge_downstream"
                else:
                    relationship = "connected_indirectly"
            if relationship == "connected_indirectly" and not allow_indirect:
                continue
            if not np.isfinite(distance) or distance > max_network_m:
                continue
            result = {
                "bio_site_id": brow["bio_site_id"], "flow_source": source,
                "flow_station_id": str(grow[gauge_id_col]),
                "flow_station_name": grow.get(station_name_col), "flow_river": grow.get(river_col),
                "gauge_network_distance_m": distance, "gauge_relationship": relationship,
                "gauge_snap_distance_m": grow["snap_distance_m"], "bio_snap_distance_m": brow["snap_distance_m"],
            }
            result.update({
                "nrfa_station_id": int(grow[gauge_id_col]),
                "nrfa_station_name": grow.get(station_name_col), "nrfa_river": grow.get(river_col),
                "gdf_percent_complete": grow.get("gdf-percent-complete"),
                "gdf_q95_flow": grow.get("gdf-q95-flow"), "gdf_q10_flow": grow.get("gdf-q10-flow"),
            })
            rows.append(result)
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked["gauge_rank"] = ranked.groupby("bio_site_id")["gauge_network_distance_m"].rank(method="first")
    ranked["gauge_candidate_count"] = ranked.groupby("bio_site_id")["flow_station_id"].transform("size")
    return ranked.sort_values(["bio_site_id", "gauge_rank"]).reset_index(drop=True)


# Snap the two site sets to the same network before looking for route distance;
# a close gauge across a watershed should never count as a match.
def match_nrfa_gauges(
    biology_points: pd.DataFrame, biology_snaps: pd.DataFrame,
    gauges: pd.DataFrame, gauge_snaps: pd.DataFrame, graph: nx.DiGraph,
    *, max_euclidean_m: float = 30000, max_network_m: float = 50000,
    allow_indirect: bool = False,
) -> pd.DataFrame:
    """Rank quality-controlled NRFA gauges on the same river network as biology sites."""
    return _match_flow_gauges(
        biology_points, biology_snaps, gauges, gauge_snaps, graph,
        gauge_id_col="nrfa_station_id", source="nrfa", station_name_col="name", river_col="river",
        max_euclidean_m=max_euclidean_m, max_network_m=max_network_m, allow_indirect=allow_indirect,
    )
