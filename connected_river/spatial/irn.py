"""UKCEH Integrated River Network topology and exact point snapping.

The source is a large CSV with WKT linework. Processing is chunked so releases
do not require a second, hand-generated copy of the network.
"""

from __future__ import annotations

import math
import pickle
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from shapely import from_wkt, get_point, get_x, get_y
from shapely import points as make_points
from shapely.strtree import STRtree

EDGE_COLUMNS = [
    "OBJECTID", "RIVERS_ID", "FNODE_FULL", "TNODE_FULL", "FLOWDIR",
    "TRACEABLE", "ENABLED", "STRAHLER", "SHREVE", "OS_NAME", "HA",
    "LENGTH", "SHAPE_LEN", "geometry",
]

IRN_GRAPH_SCHEMA_VERSION = 3
IRN_NODE_CONTINUITY_TOLERANCE_M = 25.0
_LINK_ATTRIBUTE_NAMES = (
    "link_id",
    "length_m",
    "strahler",
    "shreve",
    "river_name",
    "hydrological_area",
)


def _plain_scalar(value: object) -> object | None:
    """Return a pickle- and JSON-friendly scalar, preserving source values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalise_irn_link_id(value: object) -> int | str | None:
    """Return a stable raw IRN link ID without lossy numeric coercion.

    Integral numeric representations such as ``65``, ``65.0`` and ``"65.0"``
    resolve to the same integer. Missing values, infinities and non-integral
    numeric values are invalid rather than being silently truncated. Clearly
    nonnumeric strings are retained for compatibility with external networks.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.casefold() in {
            "<na>", "na", "n/a", "nan", "none", "null",
        }:
            return None
        try:
            number = Decimal(stripped)
        except InvalidOperation:
            return stripped
        if not number.is_finite() or number != number.to_integral_value():
            return None
        return int(number)
    return None


def same_irn_link_id(left: object, right: object) -> bool:
    """Return whether two valid values identify the same raw IRN link."""
    left_value = normalise_irn_link_id(left)
    right_value = normalise_irn_link_id(right)
    if left_value is None or right_value is None:
        return False
    return left_value == right_value


def edge_link_ids(edge_attributes: dict[str, object]) -> tuple[object, ...]:
    """Return every raw IRN link represented by one collapsed graph edge."""
    values: list[object] = []
    primary = normalise_irn_link_id(edge_attributes.get("link_id"))
    if primary is not None:
        values.append(primary)
    aliases = edge_attributes.get("link_aliases")
    if isinstance(aliases, dict):
        for value in aliases:
            normalised = normalise_irn_link_id(value)
            if normalised is not None and not any(
                same_irn_link_id(normalised, existing) for existing in values
            ):
                values.append(normalised)
    return tuple(values)


def edge_attributes_for_link(
    edge_attributes: dict[str, object], link_id: object,
) -> dict[str, object] | None:
    """Resolve attributes for a primary or collapsed parallel raw link ID."""
    aliases = edge_attributes.get("link_aliases")
    if isinstance(aliases, dict):
        for alias_id, alias_attributes in aliases.items():
            if not same_irn_link_id(alias_id, link_id):
                continue
            resolved = dict(edge_attributes)
            if isinstance(alias_attributes, dict):
                resolved.update(alias_attributes)
            resolved["link_id"] = normalise_irn_link_id(alias_id)
            return resolved
    if same_irn_link_id(edge_attributes.get("link_id"), link_id):
        return dict(edge_attributes)
    return None


def _link_attributes(attributes: dict[str, object]) -> dict[str, object]:
    """Copy the raw-link-specific subset, excluding nested aliases."""
    return {
        name: attributes.get(name)
        for name in _LINK_ATTRIBUTE_NAMES
    }


# A pair of network nodes can have more than one published link between them.
# Keep every link ID, but choose one consistently for routing and hydrology.
def _set_parallel_hydrology_representative(
    edge_attributes: dict[str, object],
) -> None:
    """Use one real alias, never a sum, for collapsed-edge order metadata."""
    aliases = edge_attributes.get("link_aliases")
    if not isinstance(aliases, dict) or not aliases:
        edge_attributes["hydrology_link_id"] = edge_attributes.get("link_id")
        edge_attributes["hydrology_attribute_method"] = "single_raw_link"
        return

    candidates: list[tuple[float, float, str, object, dict[str, object]]] = []
    for alias_id, alias_attributes in aliases.items():
        if not isinstance(alias_attributes, dict):
            continue
        try:
            shreve = float(alias_attributes.get("shreve"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(shreve) or shreve <= 0:
            continue
        try:
            length = float(alias_attributes.get("length_m"))
        except (TypeError, ValueError):
            length = math.inf
        candidates.append((
            -shreve,
            length if math.isfinite(length) else math.inf,
            str(alias_id),
            alias_id,
            alias_attributes,
        ))

    if candidates:
        _, _, _, selected_id, selected = min(candidates)
        method = "parallel_alias_max_positive_shreve"
    else:
        selected_id = edge_attributes.get("link_id")
        selected = next(
            (
                attributes
                for alias_id, attributes in aliases.items()
                if same_irn_link_id(alias_id, selected_id)
                and isinstance(attributes, dict)
            ),
            _link_attributes(edge_attributes),
        )
        method = "parallel_alias_no_positive_shreve_primary"

    edge_attributes["strahler"] = selected.get("strahler")
    edge_attributes["shreve"] = selected.get("shreve")
    edge_attributes["hydrology_link_id"] = normalise_irn_link_id(selected_id)
    edge_attributes["hydrology_attribute_method"] = method


# Some raw node IDs are reused at places that are not physically close. Split
# those cases before graph building so they cannot create zero-distance jumps.
def _resolve_spatially_ambiguous_nodes(
    edges: pd.DataFrame,
    *,
    tolerance_m: float = IRN_NODE_CONTINUITY_TOLERANCE_M,
) -> tuple[pd.Series, pd.Series, dict[tuple[object, str], int], dict[str, int]]:
    """Split reused node IDs whose line endpoints are not spatially continuous.

    A small number of raw IRN node identifiers are reused at geographically
    separate endpoints. Treating them as one graph node creates zero-length
    teleports between rivers. Endpoints are greedily clustered so every pair
    in a cluster is within the same declared geometry-continuity tolerance.
    """
    endpoints = pd.concat(
        [
            pd.DataFrame({
                "edge_index": edges.index,
                "raw_node": edges["from_node"],
                "link_id": edges["link_id"],
                "endpoint_role": "from",
                "x": edges["from_x"],
                "y": edges["from_y"],
            }),
            pd.DataFrame({
                "edge_index": edges.index,
                "raw_node": edges["to_node"],
                "link_id": edges["link_id"],
                "endpoint_role": "to",
                "x": edges["to_x"],
                "y": edges["to_y"],
            }),
        ],
        ignore_index=True,
    ).dropna(subset=["raw_node", "link_id", "x", "y"])
    graph_from = edges["from_node"].astype("Int64").copy()
    graph_to = edges["to_node"].astype("Int64").copy()
    resolution: dict[tuple[object, str], int] = {}
    split_nodes = 0
    split_endpoints = 0
    next_surrogate = -1

    for raw_node, group in endpoints.groupby("raw_node", sort=True):
        if len(group) < 2:
            continue
        x_span = float(group["x"].max() - group["x"].min())
        y_span = float(group["y"].max() - group["y"].min())
        if math.hypot(x_span, y_span) <= tolerance_m:
            continue
        ordered = group.sort_values(
            ["x", "y", "link_id", "endpoint_role", "edge_index"]
        )
        clusters: list[list[object]] = []
        for record in ordered.itertuples(index=False):
            target: list[object] | None = None
            for cluster in clusters:
                if all(
                    math.hypot(float(record.x) - float(item.x), float(record.y) - float(item.y))
                    <= tolerance_m
                    for item in cluster
                ):
                    target = cluster
                    break
            if target is None:
                target = []
                clusters.append(target)
            target.append(record)
        if len(clusters) < 2:
            continue

        split_nodes += 1
        split_endpoints += len(group)
        for cluster in clusters:
            surrogate = next_surrogate
            next_surrogate -= 1
            for record in cluster:
                edge_index = int(record.edge_index)
                role = str(record.endpoint_role)
                if role == "from":
                    graph_from.at[edge_index] = surrogate
                else:
                    graph_to.at[edge_index] = surrogate
                link_id = normalise_irn_link_id(record.link_id)
                if link_id is not None:
                    resolution[(link_id, role)] = surrogate

    return graph_from, graph_to, resolution, {
        "spatially_split_raw_node_count": split_nodes,
        "spatially_split_endpoint_count": split_endpoints,
        "spatial_surrogate_node_count": -next_surrogate - 1,
    }


def _resolved_endpoint_node(
    graph: nx.Graph | None,
    *,
    link_id: object,
    endpoint_role: str,
    raw_node: object,
) -> object:
    """Resolve a raw endpoint to the graph's spatially disambiguated node."""
    normalised_link = normalise_irn_link_id(link_id)
    if graph is not None and normalised_link is not None:
        resolution = graph.graph.get("split_node_resolution", {})
        if isinstance(resolution, dict):
            resolved = resolution.get((normalised_link, endpoint_role))
            if resolved is not None:
                return resolved
    raw = normalise_irn_link_id(raw_node)
    return raw if raw is not None else raw_node


# Prepare the large raw network once. The smaller edge and node tables are much
# quicker to load during repeated matching experiments.
def build_irn_topology(raw_csv: str | Path, output_dir: str | Path) -> dict[str, Path]:
    raw_csv, output_dir = Path(raw_csv), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        raw_csv,
        usecols=lambda column: column in EDGE_COLUMNS,
        chunksize=100_000,
        low_memory=False,
    ):
        edge = chunk.rename(columns={
            "RIVERS_ID": "link_id", "FNODE_FULL": "from_node", "TNODE_FULL": "to_node",
            "OS_NAME": "river_name", "HA": "hydrological_area",
        }).copy()
        edge["link_id"] = pd.to_numeric(edge["link_id"], errors="coerce").fillna(pd.to_numeric(edge["OBJECTID"], errors="coerce")).astype("Int64")
        edge["from_node"] = pd.to_numeric(edge["from_node"], errors="coerce").astype("Int64")
        edge["to_node"] = pd.to_numeric(edge["to_node"], errors="coerce").astype("Int64")
        edge["length_m"] = pd.to_numeric(edge.get("LENGTH"), errors="coerce").fillna(pd.to_numeric(edge.get("SHAPE_LEN"), errors="coerce"))
        edge = edge.dropna(subset=["link_id", "from_node", "to_node", "length_m"])
        geometry = from_wkt(edge["geometry"].to_numpy(), on_invalid="ignore")
        starts = get_point(geometry, 0)
        ends = get_point(geometry, -1)
        edge["from_x"] = get_x(starts)
        edge["from_y"] = get_y(starts)
        edge["to_x"] = get_x(ends)
        edge["to_y"] = get_y(ends)
        edge = edge.drop(columns="geometry")
        frames.append(edge)
    edges = pd.concat(frames, ignore_index=True)
    graph_from, graph_to, resolution, split_summary = _resolve_spatially_ambiguous_nodes(
        edges,
    )
    edges["graph_from_node"] = graph_from
    edges["graph_to_node"] = graph_to

    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        u = int(row.graph_from_node)
        v = int(row.graph_to_node)
        length = float(row.length_m)
        attributes = {
            "length_m": length,
            "link_id": int(row.link_id),
            "strahler": _plain_scalar(getattr(row, "STRAHLER", None)),
            "shreve": _plain_scalar(getattr(row, "SHREVE", None)),
            "river_name": _plain_scalar(getattr(row, "river_name", None)),
            "hydrological_area": _plain_scalar(
                getattr(row, "hydrological_area", None)
            ),
        }
        previous = graph.get_edge_data(u, v)
        if previous is None:
            _set_parallel_hydrology_representative(attributes)
            graph.add_edge(u, v, **attributes)
            continue

        aliases = {
            alias_id: dict(alias_attributes)
            for alias_id, alias_attributes in (
                previous.get("link_aliases", {}).items()
                if isinstance(previous.get("link_aliases"), dict)
                else []
            )
        }
        previous_id = normalise_irn_link_id(previous.get("link_id"))
        if previous_id is not None:
            aliases.setdefault(previous_id, _link_attributes(previous))
        current_id = normalise_irn_link_id(attributes["link_id"])
        if current_id is not None:
            aliases[current_id] = _link_attributes(attributes)

        if length < previous["length_m"]:
            # ``DiGraph.add_edge`` updates rather than replaces an existing
            # attribute dictionary. Clear it so the new shortest primary
            # cannot retain stale scalar metadata from the old primary.
            previous.clear()
            previous.update(attributes)
        previous["link_aliases"] = aliases
        _set_parallel_hydrology_representative(previous)

    parallel_edges = [
        data
        for _, _, data in graph.edges(data=True)
        if isinstance(data.get("link_aliases"), dict)
    ]
    graph.graph.update({
        "irn_graph_schema_version": IRN_GRAPH_SCHEMA_VERSION,
        "supports_link_aliases": True,
        "raw_link_count": len(edges),
        "collapsed_edge_count": int(graph.number_of_edges()),
        "parallel_edge_group_count": len(parallel_edges),
        "parallel_link_alias_count": int(
            sum(max(0, len(data["link_aliases"]) - 1) for data in parallel_edges)
        ),
        "parallel_hydrology_differs_from_primary_count": int(sum(
            not same_irn_link_id(data.get("hydrology_link_id"), data.get("link_id"))
            for data in parallel_edges
        )),
        "node_continuity_tolerance_m": IRN_NODE_CONTINUITY_TOLERANCE_M,
        "split_node_resolution": resolution,
        **split_summary,
    })
    nodes = pd.DataFrame({
        "node_id": pd.concat(
            [edges["graph_from_node"], edges["graph_to_node"]],
            ignore_index=True,
        ).drop_duplicates().astype("Int64")
    })
    paths = {"edges": output_dir / "irn_edges.parquet", "nodes": output_dir / "irn_nodes.parquet", "graph": output_dir / "irn_graph.pkl"}
    edges.to_parquet(paths["edges"], index=False)
    nodes.to_parquet(paths["nodes"], index=False)
    with paths["graph"].open("wb") as stream:
        pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return paths


# Turn the published river lines into a directed graph once, then reuse that
# graph for all of the site-to-site route searches.
def load_irn_graph(path: str | Path) -> nx.DiGraph:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


# Find the nearest plausible river link for each monitoring location and keep
# the snap distance so users can decide what level of uncertainty is acceptable.
def snap_points_to_irn_csv(
    points: pd.DataFrame, raw_csv: str | Path, *, point_id_col: str,
    max_distance_m: float = 500, chunksize: int = 50_000,
    graph: nx.Graph | None = None,
) -> pd.DataFrame:
    """Find the exact nearest IRN polyline for every valid OSGB point."""
    work = points.copy().reset_index(drop=True)
    work["easting"] = pd.to_numeric(work["easting"], errors="coerce")
    work["northing"] = pd.to_numeric(work["northing"], errors="coerce")
    valid = work["easting"].between(0, 700000) & work["northing"].between(0, 1300000)
    work = work[valid & work[point_id_col].notna()].copy().reset_index(drop=True)
    point_geometry = make_points(work["easting"].to_numpy(), work["northing"].to_numpy())
    best: dict[int, dict[str, object]] = {}
    usecols = ["RIVERS_ID", "OBJECTID", "FNODE_FULL", "TNODE_FULL", "LENGTH", "SHAPE_LEN", "OS_NAME", "geometry"]
    for chunk in pd.read_csv(raw_csv, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
        geometry = from_wkt(chunk["geometry"].to_numpy(), on_invalid="ignore")
        good = ~pd.isna(geometry)
        if not np.any(good):
            continue
        lines = geometry[good]
        source_rows = chunk.loc[good].reset_index(drop=True)
        result, distances = STRtree(lines).query_nearest(point_geometry, max_distance=max_distance_m, return_distance=True)
        if result.size == 0:
            continue
        for point_index, line_index, distance in zip(result[0], result[1], distances):
            old = best.get(int(point_index))
            if old is not None and float(old["snap_distance_m"]) <= float(distance):
                continue
            row = source_rows.iloc[int(line_index)]
            line = lines[int(line_index)]
            position = float(line.project(point_geometry[int(point_index)]))
            length = float(line.length)
            link_id = normalise_irn_link_id(row.get("RIVERS_ID"))
            if link_id is None:
                link_id = normalise_irn_link_id(row.get("OBJECTID"))
            source_length = pd.to_numeric(
                pd.Series([row.get("LENGTH")]), errors="coerce",
            ).iloc[0]
            if pd.isna(source_length):
                source_length = pd.to_numeric(
                    pd.Series([row.get("SHAPE_LEN")]), errors="coerce",
                ).iloc[0]
            raw_start_node = row.get("FNODE_FULL")
            raw_end_node = row.get("TNODE_FULL")
            best[int(point_index)] = {
                "snapped_link_id": link_id,
                "snapped_raw_start_node": raw_start_node,
                "snapped_raw_end_node": raw_end_node,
                "snapped_start_node": _resolved_endpoint_node(
                    graph,
                    link_id=link_id,
                    endpoint_role="from",
                    raw_node=raw_start_node,
                ),
                "snapped_end_node": _resolved_endpoint_node(
                    graph,
                    link_id=link_id,
                    endpoint_role="to",
                    raw_node=raw_end_node,
                ),
                "snapped_link_length_m": source_length,
                "position_on_link_m": position, "fraction_along_link": position / length if length else np.nan,
                "snap_distance_m": float(distance), "snapped_river_name": row.get("OS_NAME"),
            }
    rows = []
    for index, point in work.iterrows():
        row = {point_id_col: point[point_id_col]}
        row.update(best.get(index, {}))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["snap_graph_schema_version"] = (
        graph.graph.get("irn_graph_schema_version") if graph is not None else pd.NA
    )
    result["snap_endpoint_node_method"] = (
        "geometry_disambiguated_irn_endpoint"
        if graph is not None
        else "raw_irn_endpoint"
    )
    return result
