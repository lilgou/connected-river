"""Compact river geometry saved with each set of matched routes.

The raw Integrated River Network (IRN) line file is hundreds of megabytes and
is scanned once at release build time, retaining only the exact routes and site
reaches needed by the release candidate decisions.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import from_wkt, reverse, to_wkb
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring, transform

from connected_river.spatial.irn import load_irn_graph
from connected_river.utils.manifest import path_sha256

ROUTE_ASSET_NAME = "river_connection_routes.parquet"
ROUTE_ASSET_SCHEMA_VERSION = "4.0"
DIRECTED_ROUTE_LENGTH_ABSOLUTE_TOLERANCE_M = 50.0
DIRECTED_ROUTE_LENGTH_RELATIVE_TOLERANCE = 0.01
ROUTE_COMPONENT_CONTINUITY_TOLERANCE_M = 25.0
SAME_LINK_ROUTE_LENGTH_TOLERANCE_M = 0.05
ROUTE_ASSET_COLUMNS = {
    "bio_site_id",
    "year",
    "match_rank",
    "wq_site_id",
    "network_relationship",
    "network_distance_m",
    "biology_longitude",
    "biology_latitude",
    "wq_longitude",
    "wq_latitude",
    "route_wkb",
    "biology_link_wkb",
    "wq_link_wkb",
}


def _neighbouring_link_ids(
    graph: nx.Graph,
    anchors: Iterable[int],
    *,
    max_hops: int = 1,
    max_links: int = 80,
) -> set[int]:
    """Collect a bounded local river neighbourhood around route and site nodes."""
    frontier = {int(node) for node in anchors}
    visited = set(frontier)
    link_ids: set[int] = set()
    for _ in range(max_hops + 1):
        next_frontier: set[int] = set()
        for node in sorted(frontier):
            edge_pairs = (
                [(neighbour, node) for neighbour in graph.predecessors(node)]
                + [(node, neighbour) for neighbour in graph.successors(node)]
                if graph.is_directed()
                else [(node, neighbour) for neighbour in graph.neighbors(node)]
            )
            for start, end in edge_pairs:
                neighbour = start if end == node else end
                attributes = _edge_attributes(graph, start, end)
                link_id = attributes.get("link_id")
                if link_id is not None:
                    link_ids.add(int(link_id))
                if neighbour not in visited:
                    next_frontier.add(int(neighbour))
                if len(link_ids) >= max_links:
                    return set(sorted(link_ids)[:max_links])
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return link_ids


def _edge_attributes(graph: nx.Graph, start: int, end: int) -> dict[str, Any]:
    attributes = graph[start][end]
    if graph.is_multigraph():
        return min(
            attributes.values(),
            key=lambda item: float(item.get("length_m", float("inf"))),
        )
    return attributes


def _route_key(row: pd.Series) -> tuple[object, ...]:
    route_start = row.get("route_start_node", row["wq_start_node"])
    route_end = row.get("route_end_node", row["bio_end_node"])
    return (
        str(row["bio_site_id"]),
        str(row["wq_site_id"]),
        str(row["network_relationship"]),
        int(route_start),
        int(route_end),
        int(row["wq_link_id"]),
        int(row["bio_link_id"]),
    )


def _path_links(graph: nx.Graph, row: pd.Series) -> list[int]:
    """Return the directed interior IRN links for a candidate route.

    Same-link candidates are handled explicitly because routing between their
    terminal nodes can otherwise draw an incorrect loop.
    """
    relationship = str(row["network_relationship"])
    if relationship == "same_link":
        if int(row["wq_link_id"]) != int(row["bio_link_id"]):
            raise ValueError("A same-link match names two different IRN links")
        return []
    nodes = nx.shortest_path(
        graph,
        int(row.get("route_start_node", row["wq_start_node"])),
        int(row.get("route_end_node", row["bio_end_node"])),
        weight="length_m",
    )
    return [
        int(_edge_attributes(graph, start, end)["link_id"])
        for start, end in zip(nodes, nodes[1:])
    ]


# Read only links used by selected routes or their immediate neighbours. Loading
# the full national line file would make a portable route layer unnecessarily large.
def _load_required_lines(
    raw_csv: str | Path,
    link_ids: set[int],
    *,
    chunksize: int = 100_000,
) -> dict[int, LineString]:
    """Scan the raw IRN once and parse geometry only for required links."""
    geometries: dict[int, LineString] = {}
    columns = ["RIVERS_ID", "OBJECTID", "geometry"]
    for chunk in pd.read_csv(
        raw_csv,
        usecols=columns,
        chunksize=chunksize,
        low_memory=False,
    ):
        identifiers = pd.to_numeric(chunk["RIVERS_ID"], errors="coerce").fillna(
            pd.to_numeric(chunk["OBJECTID"], errors="coerce")
        )
        selected = chunk.loc[identifiers.isin(link_ids), ["geometry"]].copy()
        if selected.empty:
            continue
        selected["link_id"] = identifiers.loc[selected.index].astype("int64")
        parsed = from_wkt(selected["geometry"].to_numpy(), on_invalid="ignore")
        for link_id, geometry in zip(selected["link_id"], parsed):
            if geometry is not None and not geometry.is_empty:
                geometries.setdefault(int(link_id), geometry)
    missing = sorted(link_ids.difference(geometries))
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise ValueError(
            f"The raw IRN is missing {len(missing):,} required links ({preview})"
        )
    return geometries


def _network_point(snap: pd.Series, geometries: dict[int, LineString]) -> Point:
    line = geometries[int(snap["snapped_link_id"])]
    distance = min(max(float(snap["position_on_link_m"]), 0.0), line.length)
    return line.interpolate(distance)


def _point_to_endpoint(
    snap: pd.Series,
    endpoint: int,
    geometries: dict[int, LineString],
) -> LineString:
    """Cut the snapped link from the monitoring point towards an endpoint."""
    line = geometries[int(snap["snapped_link_id"])]
    distance = min(max(float(snap["position_on_link_m"]), 0.0), line.length)
    if endpoint == int(snap["snapped_start_node"]):
        return reverse(substring(line, 0, distance))
    if endpoint == int(snap["snapped_end_node"]):
        return substring(line, distance, line.length)
    raise ValueError("Route endpoint is not on the snapped water-quality link")


def _endpoint_to_point(
    snap: pd.Series,
    endpoint: int,
    geometries: dict[int, LineString],
) -> LineString:
    """Cut the snapped link from an endpoint towards the monitoring point."""
    line = geometries[int(snap["snapped_link_id"])]
    distance = min(max(float(snap["position_on_link_m"]), 0.0), line.length)
    if endpoint == int(snap["snapped_start_node"]):
        return substring(line, 0, distance)
    if endpoint == int(snap["snapped_end_node"]):
        return reverse(substring(line, distance, line.length))
    raise ValueError("Route endpoint is not on the snapped biology link")


# Sites on the same published link still need the correct section of line and
# direction, rather than the geometry of the whole link.
def _same_link_route(
    wq_snap: pd.Series,
    bio_snap: pd.Series,
    geometries: dict[int, LineString],
) -> LineString:
    link_id = int(wq_snap["snapped_link_id"])
    if link_id != int(bio_snap["snapped_link_id"]):
        raise ValueError("A same-link route has mismatched snap links")
    line = geometries[link_id]
    wq_position = min(max(float(wq_snap["position_on_link_m"]), 0.0), line.length)
    bio_position = min(max(float(bio_snap["position_on_link_m"]), 0.0), line.length)
    if wq_position <= bio_position:
        route = substring(line, wq_position, bio_position)
    else:
        # Preserve a consistent chemistry-to-biology direction while allowing
        # downstream chemistry on the same link.
        route = reverse(substring(line, bio_position, wq_position))
    if isinstance(route, Point):
        # Zero-distance matches still need a serialisable line.  The map shows
        # the co-located markers even though this line has no visible length.
        coordinate = route.coords[0]
        return LineString([coordinate, coordinate])
    return route


# Assemble the start fragment, complete middle links and end fragment into one
# continuous line that follows the selected network path.
def _route_geometry(
    row: pd.Series,
    *,
    path_links: list[int],
    bio_snap: pd.Series,
    wq_snap: pd.Series,
    geometries: dict[int, LineString],
    component_gaps_m: list[float] | None = None,
) -> LineString | MultiLineString:
    """Assemble route parts, rejecting endpoint gaps above the declared 25 m tolerance."""
    if str(row["network_relationship"]) == "same_link":
        return _same_link_route(wq_snap, bio_snap, geometries)

    terminal_ids = {
        int(wq_snap["snapped_link_id"]),
        int(bio_snap["snapped_link_id"]),
    }
    downstream_wq = str(row.get("route_origin_site_type", "water_quality")) == "biology"
    if downstream_wq:
        first_snap, last_snap = bio_snap, wq_snap
    else:
        first_snap, last_snap = wq_snap, bio_snap
    route_start = int(row.get("route_start_node", row["wq_start_node"]))
    route_end = int(row.get("route_end_node", row["bio_end_node"]))
    parts = [
        _point_to_endpoint(first_snap, route_start, geometries),
        *[
            geometries[link_id]
            for link_id in path_links
            if link_id not in terminal_ids
        ],
        _endpoint_to_point(last_snap, route_end, geometries),
    ]
    usable = [part for part in parts if not part.is_empty and part.length > 0]
    gaps = [
        Point(left.coords[-1]).distance(Point(right.coords[0]))
        for left, right in zip(usable, usable[1:])
    ]
    if component_gaps_m is not None:
        component_gaps_m.extend(float(gap) for gap in gaps)
    if gaps and max(gaps) > ROUTE_COMPONENT_CONTINUITY_TOLERANCE_M:
        raise ValueError(
            "Route geometry components are discontinuous: maximum endpoint gap "
            f"{max(gaps):.2f} m exceeds the "
            f"{ROUTE_COMPONENT_CONTINUITY_TOLERANCE_M:.0f} m tolerance"
        )
    if not usable:
        wq_point = _network_point(wq_snap, geometries)
        bio_point = _network_point(bio_snap, geometries)
        return LineString([wq_point.coords[0], bio_point.coords[0]])
    if len(usable) == 1:
        return usable[0]
    return MultiLineString(usable)


# Check the saved routes against the candidate table before relying on them for
# maps. IDs, distances and direction should all describe the same match.
def route_asset_matches_candidates(
    candidates: pd.DataFrame,
    routes: pd.DataFrame,
    *,
    top_n: int = 5,
) -> bool:
    """Return whether a filtered asset exactly matches the visible candidates."""
    if not ROUTE_ASSET_COLUMNS.issubset(routes.columns):
        return False
    required = {
        "bio_site_id", "year", "match_rank", "wq_site_id",
        "network_relationship", "network_distance_m",
    }
    if not required.issubset(candidates.columns):
        return False
    key_columns = ["bio_site_id", "year"]
    comparison_columns = [
        *key_columns, "match_rank", "wq_site_id",
        "network_relationship", "network_distance_m",
    ]
    expected = (
        candidates.assign(
            match_rank=pd.to_numeric(candidates["match_rank"], errors="coerce")
        )
        .dropna(subset=comparison_columns)
        .loc[lambda frame: frame["match_rank"].between(1, top_n), comparison_columns]
        .sort_values([*key_columns, "match_rank"])
        .assign(
            match_rank=lambda frame: frame["match_rank"].astype(int),
            wq_site_id=lambda frame: frame["wq_site_id"].astype(str),
            network_relationship=lambda frame: frame["network_relationship"].astype(str),
            network_distance_m=lambda frame: pd.to_numeric(
                frame["network_distance_m"], errors="coerce",
            ).astype(float),
        )
        .reset_index(drop=True)
    )
    observed = (
        routes[comparison_columns]
        .dropna()
        .sort_values([*key_columns, "match_rank"])
        .assign(
            match_rank=lambda frame: frame["match_rank"].astype(int),
            wq_site_id=lambda frame: frame["wq_site_id"].astype(str),
            network_relationship=lambda frame: frame["network_relationship"].astype(str),
            network_distance_m=lambda frame: pd.to_numeric(
                frame["network_distance_m"], errors="coerce",
            ).astype(float),
        )
        .reset_index(drop=True)
    )
    for frame in (expected, observed):
        if "bio_site_id" in frame:
            frame["bio_site_id"] = frame["bio_site_id"].astype(str)
        if "year" in frame:
            frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype(int)
    identity_columns = [
        column for column in comparison_columns if column != "network_distance_m"
    ]
    if not expected[identity_columns].equals(observed[identity_columns]):
        return False
    if not np.allclose(
        expected["network_distance_m"].to_numpy(dtype=float),
        observed["network_distance_m"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-6,
    ):
        return False
    top_two = routes[pd.to_numeric(routes["match_rank"], errors="coerce").le(2)]
    return bool(
        routes[[
            "biology_longitude", "biology_latitude",
            "wq_longitude", "wq_latitude",
        ]].notna().all().all()
        and top_two["route_wkb"].notna().all()
    )


# Save just enough river geometry to explain the selected matches later. This
# avoids requiring every notebook reader to download the full national network.
def build_river_connection_route_asset(
    *,
    ranked_path: str | Path,
    biology_snap_path: str | Path,
    water_quality_snap_path: str | Path,
    graph_path: str | Path,
    irn_raw_csv: str | Path,
    output_path: str | Path,
    years: Iterable[int],
    top_n: int = 5,
    route_ranks: tuple[int, ...] = (1, 2),
) -> Path:
    """Build the compact exact-geography asset used by River Connections."""
    ranked_path = Path(ranked_path)
    biology_snap_path = Path(biology_snap_path)
    water_quality_snap_path = Path(water_quality_snap_path)
    graph_path = Path(graph_path)
    irn_raw_csv = Path(irn_raw_csv)
    output_path = Path(output_path)
    included_years = sorted({int(year) for year in years})
    if not included_years:
        raise ValueError("At least one public year is required for the route asset")
    columns = [
        "bio_site_id", "year", "wq_site_id", "network_distance_m",
        "network_relationship", "wq_start_node", "bio_end_node",
        "wq_link_id", "bio_link_id", "match_rank",
    ]
    schema_columns = set(pq.read_schema(ranked_path).names)
    columns.extend(
        column
        for column in (
            "route_start_node", "route_end_node", "route_origin_site_type",
        )
        if column in schema_columns
    )
    ranked = pd.read_parquet(ranked_path, columns=columns)
    if "route_start_node" not in ranked:
        ranked["route_start_node"] = ranked["wq_start_node"]
    if "route_end_node" not in ranked:
        ranked["route_end_node"] = ranked["bio_end_node"]
    if "route_origin_site_type" not in ranked:
        ranked["route_origin_site_type"] = "water_quality"
    ranked["year"] = pd.to_numeric(ranked["year"], errors="coerce").astype("Int64")
    ranked["match_rank"] = pd.to_numeric(
        ranked["match_rank"], errors="coerce"
    ).astype("Int64")
    public = (
        ranked[
            ranked["year"].isin(included_years)
            & ranked["match_rank"].between(1, top_n)
        ]
        .dropna(subset=[
            "bio_site_id", "year", "wq_site_id", "match_rank",
            "network_relationship", "network_distance_m", "route_start_node",
            "route_end_node", "wq_link_id", "bio_link_id",
        ])
        .sort_values(["bio_site_id", "year", "match_rank"])
        .drop_duplicates(["bio_site_id", "year", "match_rank"])
        .reset_index(drop=True)
    )
    if public.empty:
        raise ValueError("No public ranked candidates were available for the route asset")

    bio_snaps = pd.read_parquet(biology_snap_path).drop_duplicates("bio_site_id")
    wq_snaps = pd.read_parquet(water_quality_snap_path).drop_duplicates("wq_site_id")
    bio_snaps["bio_site_id"] = bio_snaps["bio_site_id"].astype(str)
    wq_snaps["wq_site_id"] = wq_snaps["wq_site_id"].astype(str)
    bio_snaps = bio_snaps.set_index("bio_site_id")
    wq_snaps = wq_snaps.set_index("wq_site_id")
    missing_bio = sorted(set(public["bio_site_id"].astype(str)) - set(bio_snaps.index))
    missing_wq = sorted(set(public["wq_site_id"].astype(str)) - set(wq_snaps.index))
    if missing_bio or missing_wq:
        raise ValueError(
            "Route asset snaps do not cover the ranked candidates "
            f"({len(missing_bio)} biology, {len(missing_wq)} chemistry missing)"
        )

    graph = load_irn_graph(graph_path)
    path_by_key: dict[tuple[object, ...], list[int]] = {}
    route_rows = public[public["match_rank"].isin(route_ranks)]
    for _, row in route_rows.drop_duplicates([
        "bio_site_id", "wq_site_id", "network_relationship",
        "route_start_node", "route_end_node", "wq_link_id", "bio_link_id",
    ]).iterrows():
        path_by_key[_route_key(row)] = _path_links(graph, row)

    neighbouring_links_by_site_year: dict[tuple[str, int], set[int]] = {}
    for (bio_site_id, year), group in public.groupby(["bio_site_id", "year"]):
        bio_snap = bio_snaps.loc[str(bio_site_id)]
        anchors = {
            int(bio_snap["snapped_start_node"]),
            int(bio_snap["snapped_end_node"]),
        }
        for wq_site_id in group["wq_site_id"].astype(str):
            wq_snap = wq_snaps.loc[wq_site_id]
            anchors.update({
                int(wq_snap["snapped_start_node"]),
                int(wq_snap["snapped_end_node"]),
            })
        neighbouring_links_by_site_year[(str(bio_site_id), int(year))] = (
            _neighbouring_link_ids(graph, anchors)
        )

    required_links = set(
        pd.concat([public["wq_link_id"], public["bio_link_id"]])
        .astype(int)
        .tolist()
    )
    for link_ids in path_by_key.values():
        required_links.update(link_ids)
    for link_ids in neighbouring_links_by_site_year.values():
        required_links.update(link_ids)
    geometries = _load_required_lines(irn_raw_csv, required_links)
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    transformed_lines = {
        link_id: transform(transformer.transform, geometry)
        for link_id, geometry in geometries.items()
    }
    link_wkb = {
        link_id: to_wkb(geometry, output_dimension=2)
        for link_id, geometry in transformed_lines.items()
    }
    neighbouring_wkb = {
        key: to_wkb(
            MultiLineString([
                transformed_lines[link_id]
                for link_id in sorted(link_ids)
                if link_id in transformed_lines
            ]),
            output_dimension=2,
        )
        for key, link_ids in neighbouring_links_by_site_year.items()
        if link_ids
    }
    bio_points = {
        site_id: transform(
            transformer.transform,
            _network_point(bio_snaps.loc[site_id], geometries),
        )
        for site_id in public["bio_site_id"].astype(str).unique()
    }
    wq_points = {
        site_id: transform(
            transformer.transform,
            _network_point(wq_snaps.loc[site_id], geometries),
        )
        for site_id in public["wq_site_id"].astype(str).unique()
    }

    route_wkb: dict[tuple[object, ...], bytes] = {}
    route_length_m: dict[tuple[object, ...], float] = {}
    route_component_gaps_m: list[float] = []
    for _, row in route_rows.drop_duplicates([
        "bio_site_id", "wq_site_id", "network_relationship",
        "wq_start_node", "bio_end_node", "wq_link_id", "bio_link_id",
    ]).iterrows():
        key = _route_key(row)
        route = _route_geometry(
            row,
            path_links=path_by_key[key],
            bio_snap=bio_snaps.loc[str(row["bio_site_id"])],
            wq_snap=wq_snaps.loc[str(row["wq_site_id"])],
            geometries=geometries,
            component_gaps_m=route_component_gaps_m,
        )
        route_length_m[key] = float(route.length)
        route_wkb[key] = to_wkb(
            transform(transformer.transform, route), output_dimension=2
        )

    records = []
    route_length_errors: list[float] = []
    same_link_length_errors: list[float] = []
    directed_length_errors: list[float] = []
    excessive_directed_errors: list[tuple[float, float]] = []
    for _, row in public.iterrows():
        bio_site_id = str(row["bio_site_id"])
        wq_site_id = str(row["wq_site_id"])
        rank = int(row["match_rank"])
        bio_point = bio_points[bio_site_id]
        wq_point = wq_points[wq_site_id]
        route_key = _route_key(row)
        if rank in route_ranks:
            length_error = abs(
                route_length_m[route_key] - float(row["network_distance_m"])
            )
            route_length_errors.append(length_error)
            if str(row["network_relationship"]) == "same_link":
                same_link_length_errors.append(length_error)
            elif str(row["network_relationship"]) in {
                "wq_upstream_of_bio", "wq_downstream_of_bio",
            }:
                ranked_distance = abs(float(row["network_distance_m"]))
                tolerance = max(
                    DIRECTED_ROUTE_LENGTH_ABSOLUTE_TOLERANCE_M,
                    DIRECTED_ROUTE_LENGTH_RELATIVE_TOLERANCE * ranked_distance,
                )
                directed_length_errors.append(length_error)
                if length_error > tolerance:
                    excessive_directed_errors.append((length_error, tolerance))
        records.append({
            "bio_site_id": bio_site_id,
            "year": int(row["year"]),
            "match_rank": rank,
            "wq_site_id": wq_site_id,
            "network_relationship": str(row["network_relationship"]),
            "network_distance_m": float(row["network_distance_m"]),
            "biology_longitude": float(bio_point.x),
            "biology_latitude": float(bio_point.y),
            "wq_longitude": float(wq_point.x),
            "wq_latitude": float(wq_point.y),
            "route_wkb": route_wkb.get(route_key) if rank in route_ranks else None,
            "biology_link_wkb": link_wkb[int(row["bio_link_id"])],
            "wq_link_wkb": link_wkb[int(row["wq_link_id"])],
            "neighbouring_links_wkb": neighbouring_wkb.get((bio_site_id, int(row["year"]))),
        })
    asset = pd.DataFrame.from_records(records)
    asset["year"] = asset["year"].astype("int16")
    asset["match_rank"] = asset["match_rank"].astype("int8")
    if not route_asset_matches_candidates(public, asset, top_n=top_n):
        raise ValueError("The generated route asset does not align with its ranked candidates")
    if same_link_length_errors and max(
        same_link_length_errors
    ) > SAME_LINK_ROUTE_LENGTH_TOLERANCE_M:
        raise ValueError("Same-link route geometry does not reproduce the ranked distance")
    if excessive_directed_errors:
        largest_error, tolerance = max(excessive_directed_errors)
        raise ValueError(
            "Directed route geometry does not reproduce the ranked distance: "
            f"{largest_error:.2f} m error exceeds the {tolerance:.2f} m tolerance"
        )

    table = pa.Table.from_pandas(asset, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({
        b"route_asset_schema_version": ROUTE_ASSET_SCHEMA_VERSION.encode(),
        b"crs": b"EPSG:4326",
        b"source_ranked_sha256": path_sha256(ranked_path).encode(),
        b"source_irn_graph_sha256": path_sha256(graph_path).encode(),
        b"source_raw_irn_sha256": path_sha256(irn_raw_csv).encode(),
        b"source_biology_snap_sha256": path_sha256(biology_snap_path).encode(),
        b"source_chemistry_snap_sha256": path_sha256(
            water_quality_snap_path
        ).encode(),
        b"public_years": ",".join(map(str, included_years)).encode(),
        b"candidate_rows": str(len(asset)).encode(),
        b"site_years": str(
            asset[["bio_site_id", "year"]].drop_duplicates().shape[0]
        ).encode(),
        b"route_rows": str(asset["route_wkb"].notna().sum()).encode(),
        b"route_length_error_median": str(
            float(pd.Series(route_length_errors).median())
        ).encode(),
        b"route_length_error_p95": str(
            float(pd.Series(route_length_errors).quantile(0.95))
        ).encode(),
        b"route_length_error_max": str(max(route_length_errors)).encode(),
        b"same_link_length_error_max": str(
            max(same_link_length_errors, default=0.0)
        ).encode(),
        b"directed_route_length_error_max": str(
            max(directed_length_errors, default=0.0)
        ).encode(),
        b"directed_route_length_tolerance": (
            b"max(50 metres, 1 percent of ranked network_distance_m)"
        ),
        b"irn_spatially_split_raw_node_count": str(
            int(graph.graph.get("spatially_split_raw_node_count", 0))
        ).encode(),
        b"route_component_gap_max_m": str(
            max(route_component_gaps_m, default=0.0)
        ).encode(),
        b"route_component_continuity_tolerance_m": str(
            ROUTE_COMPONENT_CONTINUITY_TOLERANCE_M
        ).encode(),
        b"geometry_method": (
            b"exact_irn_substrings_and_directed_shortest_paths;"
            b"same_link_routes_cut_directly_between_snapped_positions;"
            b"one_hop_local_river_neighbourhood_capped_at_80_links"
        ),
    })
    table = table.replace_schema_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.part")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            row_group_size=2048,
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def route_asset_metadata(path: str | Path) -> dict[str, str]:
    """Read the source details stored with a route file."""
    metadata = pq.read_metadata(path).metadata or {}
    return {
        key.decode(): value.decode()
        for key, value in metadata.items()
        if key != b"ARROW:schema"
    }
