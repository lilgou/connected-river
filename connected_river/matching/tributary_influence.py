"""Estimate tributary influence between matched river sites.

Shreve magnitude is an additive catchment-size proxy, not observed discharge.
For the optional flow proxy, incoming branch weights are ``Shreve ** 0.72``
and are normalised at each confluence.  The exponent is the rounded log-log
slope from project validation against positive National River Flow Archive
mean-flow records (Spearman correlation about 0.83). Component shares are
propagated through later confluences so the final represented and
unrepresented shares remain compositional and sum to one.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from connected_river.spatial.irn import (
    IRN_GRAPH_SCHEMA_VERSION,
    edge_attributes_for_link,
)

FLOW_PROXY_EXPONENT = 0.72
MAJOR_TRIBUTARY_FLOW_FRACTION = 0.25
FLOW_PROXY_METHOD = (
    "compositional_incoming_branch_weights_shreve_pow_exponent_"
    "nrfa_mean_flow_calibrated_proxy_not_observed_flow"
)
DISTANCE_DECAY_FLOW_METHOD = (
    "distance_decayed_compositional_incoming_branch_weights_using_confluence_to_biology_distance"
)


def graph_has_tributary_attributes(graph: nx.DiGraph) -> bool:
    """Whether every graph edge has the required order attributes.

    The keys, rather than non-null values, are checked so genuine source-data
    gaps remain distinct from an incompatible cached graph.
    """
    if (
        graph.graph.get("irn_graph_schema_version") != IRN_GRAPH_SCHEMA_VERSION
        or graph.graph.get("supports_link_aliases") is not True
    ):
        return False
    found_edge = False
    for _, _, data in graph.edges(data=True):
        found_edge = True
        if any(
            name not in data
            for name in (
                "strahler",
                "shreve",
                "hydrology_link_id",
                "hydrology_attribute_method",
            )
        ):
            return False
    return found_edge


def _number(value: object) -> int | float | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    if not isinstance(value, Number):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _edge_number(data: dict[str, Any] | None, name: str) -> int | float | None:
    if not data:
        return None
    value = _number(data.get(name, data.get(name.upper())))
    if name.lower() in {"shreve", "strahler"} and (
        value is None or float(value) <= 0
    ):
        return None
    return value


def _serialisable_link_id(value: object) -> int | float | str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _coerce_flow_lookup(mapping: object) -> dict[int | float | str, float]:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise TypeError("tributary_observed_flow_by_link_id must be a mapping")
    lookup: dict[int | float | str, float] = {}
    for key, raw_value in mapping.items():
        normalised_key = _serialisable_link_id(key)
        if normalised_key is None:
            continue
        value = _number(raw_value)
        if value is None or value <= 0:
            continue
        lookup[normalised_key] = float(value)
    return lookup


def _lookup_flow_by_link(
    observed_flow_by_link: dict[int | float | str, float],
    primary_link_id: object,
    hydrology_link_id: object,
) -> float | None:
    for key in (
        _serialisable_link_id(hydrology_link_id),
        _serialisable_link_id(primary_link_id),
    ):
        if key is None:
            continue
        candidate = observed_flow_by_link.get(key)
        if candidate is not None and math.isfinite(candidate):
            return float(candidate)
    return None


def _edge_weight(
    shreve: int | float | None,
    observed_flow: float | None,
    flow_proxy_exponent: float,
) -> float | None:
    if observed_flow is not None and observed_flow > 0:
        return float(observed_flow)
    if shreve is None:
        return None
    if shreve <= 0:
        return None
    return float(shreve) ** flow_proxy_exponent


def _distance_to_biology(
    nodes: list[Hashable],
    graph: nx.DiGraph,
    bio_link_position_m: float | None,
) -> list[float] | None:
    if not nodes:
        return None
    if bio_link_position_m is None or not math.isfinite(float(bio_link_position_m)):
        return None
    if len(nodes) == 1:
        distance = float(bio_link_position_m)
        return [distance] if distance >= 0 else None
    edge_lengths: list[float] = []
    for left, right in zip(nodes, nodes[1:]):
        if not graph.has_edge(left, right):
            return None
        length = _edge_number(graph[left][right], "length_m")
        if length is None or length < 0:
            return None
        edge_lengths.append(float(length))
    if len(edge_lengths) != len(nodes) - 1:
        return None
    distances = [0.0] * len(nodes)
    running = float(bio_link_position_m)
    if running < 0:
        return None
    distances[-1] = running
    for index in range(len(nodes) - 2, -1, -1):
        running += edge_lengths[index]
        distances[index] = running
    return distances


def _same_link_id(left: object, right: object) -> bool:
    left_value = _serialisable_link_id(left)
    right_value = _serialisable_link_id(right)
    if left_value is None or right_value is None:
        return False
    if isinstance(left_value, Number) and isinstance(right_value, Number):
        return float(left_value) == float(right_value)
    return str(left_value) == str(right_value)


def _find_incoming_link(
    graph: nx.DiGraph, node: Hashable, link_id: object
) -> tuple[Hashable, Hashable, dict[str, Any], str] | None:
    if node not in graph:
        return None
    incident = list(graph.in_edges(node, data=True))
    for source, target, data in incident:
        resolved = edge_attributes_for_link(data, link_id)
        if resolved is not None:
            method = (
                "primary_link_id"
                if _same_link_id(data.get("link_id"), link_id)
                else "parallel_link_alias"
            )
            return source, target, resolved, method
    if len(incident) == 1:
        source, target, data = incident[0]
        return source, target, dict(data), "unique_incident_edge_fallback"
    return None


def _find_outgoing_link(
    graph: nx.DiGraph, node: Hashable, link_id: object
) -> tuple[Hashable, Hashable, dict[str, Any], str] | None:
    if node not in graph:
        return None
    incident = list(graph.out_edges(node, data=True))
    for source, target, data in incident:
        resolved = edge_attributes_for_link(data, link_id)
        if resolved is not None:
            method = (
                "primary_link_id"
                if _same_link_id(data.get("link_id"), link_id)
                else "parallel_link_alias"
            )
            return source, target, resolved, method
    if len(incident) == 1:
        source, target, data = incident[0]
        return source, target, dict(data), "unique_incident_edge_fallback"
    return None


def _monotonic(values: list[int | float | None]) -> bool | None:
    if len(values) < 2 or any(value is None for value in values):
        return None
    return all(float(left) <= float(right) for left, right in zip(values, values[1:]))


def _base_metrics(
    flow_proxy_exponent: float, major_tributary_flow_fraction: float
    ) -> dict[str, object]:
    return {
        "intervening_confluence_count": 0,
        "intervening_tributary_count": 0,
        "intervening_tributary_link_ids": [],
        "intervening_tributary_hydrology_link_ids": [],
        "intervening_tributary_missing_link_id_count": 0,
        "wq_strahler": None,
        "wq_shreve": None,
        "bio_strahler": None,
        "bio_shreve": None,
        "represented_shreve_ratio_raw": None,
        "represented_shreve_fraction": None,
        "unrepresented_shreve_fraction": None,
        "estimated_flow_represented_fraction": None,
        "estimated_flow_unrepresented_fraction": None,
        "estimated_transport_represented_fraction": None,
        "estimated_transport_unrepresented_fraction": None,
        "largest_tributary_shreve_fraction": None,
        "largest_tributary_local_shreve_fraction": None,
        "largest_side_tributary_estimated_flow_fraction_at_biology": None,
        "largest_side_tributary_local_flow_fraction": None,
        "largest_side_tributary_transport_fraction_at_biology": None,
        "largest_tributary_link_id": None,
        "largest_tributary_hydrology_link_id": None,
        "largest_side_tributary_estimated_flow_link_id": None,
        "largest_side_tributary_estimated_flow_hydrology_link_id": None,
        "transport_decay_scale_m": None,
        "structural_major_tributary_count": 0,
        "major_influence_tributary_count": 0,
        "major_tributary_count": 0,
        "major_tributary_link_ids": [],
        "major_tributary_hydrology_link_ids": [],
        "major_structural_tributary_between_sites": False,
        "major_influence_tributary_between_sites": False,
        "combined_tributary_influence_major": False,
        "major_tributary_between_sites": False,
        "tributary_shreve_monotonic": None,
        "shreve_monotonic_downstream": None,
        "strahler_monotonic_downstream": None,
        "network_order_monotonic": None,
        "tributary_topology_complete": False,
        "tributary_attribute_coverage_fraction": None,
        "tributary_wq_link_resolution": "unavailable",
        "tributary_bio_link_resolution": "unavailable",
        "tributary_parallel_alias_resolved": False,
        "flow_proxy_exponent": float(flow_proxy_exponent),
        "major_tributary_flow_fraction_threshold": float(
            major_tributary_flow_fraction
        ),
        "estimated_flow_method": FLOW_PROXY_METHOD,
        "tributary_influence_class": "unavailable",
        "tributary_influence_method": "unavailable",
    }


@dataclass(frozen=True)
# Keep the route description together while the tributary calculations are
# being built. This is easier to follow than passing the same pieces separately.
class _RouteTopology:
    nodes: list[Hashable]
    wq_edge: tuple[Hashable, Hashable, dict[str, Any], str]
    bio_edge: tuple[Hashable, Hashable, dict[str, Any], str] | None
    wq_data: dict[str, Any]
    bio_data: dict[str, Any] | None
    node_to_biology_distance_m: list[float] | None


def _apply_same_link_topology(
    metrics: dict[str, object],
    graph: nx.DiGraph,
    nodes: list[Hashable],
    wq_link_id: object,
    bio_link_id: object,
) -> None:
    """Record the no-tributary result for two sites on the same link."""
    metrics.update({
        "represented_shreve_ratio_raw": 1.0,
        "represented_shreve_fraction": 1.0,
        "unrepresented_shreve_fraction": 0.0,
        "estimated_flow_represented_fraction": 1.0,
        "estimated_flow_unrepresented_fraction": 0.0,
        "estimated_transport_represented_fraction": 1.0,
        "estimated_transport_unrepresented_fraction": 0.0,
        "largest_tributary_shreve_fraction": 0.0,
        "largest_tributary_local_shreve_fraction": 0.0,
        "largest_side_tributary_estimated_flow_fraction_at_biology": 0.0,
        "largest_side_tributary_local_flow_fraction": 0.0,
        "largest_side_tributary_transport_fraction_at_biology": 0.0,
        "tributary_shreve_monotonic": True,
        "shreve_monotonic_downstream": True,
        "strahler_monotonic_downstream": True,
        "network_order_monotonic": True,
        "tributary_topology_complete": True,
        "tributary_attribute_coverage_fraction": 1.0,
        "tributary_influence_class": "none",
        "tributary_influence_method": "same_link_no_intervening_tributary",
    })
    if not nodes:
        return

    wq_edge = _find_incoming_link(graph, nodes[0], wq_link_id)
    bio_edge = _find_incoming_link(graph, nodes[0], bio_link_id)
    if wq_edge is None:
        wq_edge = _find_outgoing_link(graph, nodes[-1], wq_link_id)
    if bio_edge is None:
        bio_edge = _find_outgoing_link(graph, nodes[-1], bio_link_id)
    if wq_edge is not None:
        metrics["wq_strahler"] = _edge_number(wq_edge[2], "strahler")
        metrics["wq_shreve"] = _edge_number(wq_edge[2], "shreve")
        metrics["tributary_wq_link_resolution"] = wq_edge[3]
    if bio_edge is not None:
        metrics["bio_strahler"] = _edge_number(bio_edge[2], "strahler")
        metrics["bio_shreve"] = _edge_number(bio_edge[2], "shreve")
        metrics["tributary_bio_link_resolution"] = bio_edge[3]
    metrics["tributary_parallel_alias_resolved"] = any(
        edge is not None and edge[3] == "parallel_link_alias"
        for edge in (wq_edge, bio_edge)
    )


def _resolve_route_topology(
    metrics: dict[str, object],
    graph: nx.DiGraph,
    nodes: list[Hashable],
    *,
    relationship: str,
    wq_link_id: object,
    bio_link_id: object,
    bio_link_position_m: float | None,
) -> _RouteTopology | None:
    """Validate a directed path and attach its endpoint/order evidence."""
    if relationship != "wq_upstream_of_bio" or not nodes:
        metrics["tributary_influence_method"] = "not_applicable_or_path_unavailable"
        return None
    if any(node not in graph for node in nodes):
        metrics["tributary_influence_method"] = "path_node_missing_from_irn_graph"
        return None
    if any(not graph.has_edge(left, right) for left, right in zip(nodes, nodes[1:])):
        metrics["tributary_influence_method"] = "invalid_directed_irn_path"
        return None

    wq_edge = _find_incoming_link(graph, nodes[0], wq_link_id)
    bio_edge = _find_outgoing_link(graph, nodes[-1], bio_link_id)
    if wq_edge is None:
        metrics["tributary_influence_method"] = "wq_link_missing_at_path_start"
        return None

    wq_data = wq_edge[2]
    bio_data = bio_edge[2] if bio_edge is not None else None
    metrics.update({
        "wq_strahler": _edge_number(wq_data, "strahler"),
        "wq_shreve": _edge_number(wq_data, "shreve"),
        "bio_strahler": _edge_number(bio_data, "strahler"),
        "bio_shreve": _edge_number(bio_data, "shreve"),
        "tributary_topology_complete": bio_edge is not None,
        "tributary_wq_link_resolution": wq_edge[3],
        "tributary_bio_link_resolution": (
            bio_edge[3] if bio_edge is not None else "unavailable"
        ),
        "tributary_parallel_alias_resolved": (
            wq_edge[3] == "parallel_link_alias"
            or (bio_edge is not None and bio_edge[3] == "parallel_link_alias")
        ),
    })

    route_data = [wq_data]
    route_data.extend(graph[left][right] for left, right in zip(nodes, nodes[1:]))
    if bio_data is not None:
        route_data.append(bio_data)
    shreve_values = [_edge_number(data, "shreve") for data in route_data]
    strahler_values = [_edge_number(data, "strahler") for data in route_data]
    metrics["shreve_monotonic_downstream"] = _monotonic(shreve_values)
    metrics["tributary_shreve_monotonic"] = metrics[
        "shreve_monotonic_downstream"
    ]
    metrics["strahler_monotonic_downstream"] = _monotonic(strahler_values)
    if (
        metrics["shreve_monotonic_downstream"] is not None
        and metrics["strahler_monotonic_downstream"] is not None
    ):
        metrics["network_order_monotonic"] = bool(
            metrics["shreve_monotonic_downstream"]
            and metrics["strahler_monotonic_downstream"]
        )

    wq_shreve = metrics["wq_shreve"]
    bio_shreve = metrics["bio_shreve"]
    if wq_shreve is not None and bio_shreve is not None and float(bio_shreve) > 0:
        raw_ratio = float(wq_shreve) / float(bio_shreve)
        represented = min(1.0, max(0.0, raw_ratio))
        metrics.update({
            "represented_shreve_ratio_raw": raw_ratio,
            "represented_shreve_fraction": represented,
            "unrepresented_shreve_fraction": 1.0 - represented,
        })

    return _RouteTopology(
        nodes=nodes,
        wq_edge=wq_edge,
        bio_edge=bio_edge,
        wq_data=wq_data,
        bio_data=bio_data,
        node_to_biology_distance_m=_distance_to_biology(
            nodes,
            graph,
            _number(bio_link_position_m),
        ),
    )


def _collect_route_tributaries(
    metrics: dict[str, object],
    graph: nx.DiGraph,
    topology: _RouteTopology,
    observed_flow_lookup: dict[int | float | str, float],
    flow_proxy_exponent: float,
) -> tuple[list[dict[str, object]], dict[int, list[int]]]:
    """Enumerate tributary mouths and record their local topology evidence."""
    tributaries: list[dict[str, object]] = []
    confluence_side_indices: dict[int, list[int]] = {}
    attribute_values = 0
    attribute_slots = 0
    bio_shreve = metrics["bio_shreve"]

    for index, node in enumerate(topology.nodes):
        if index == 0:
            main_source = topology.wq_edge[0]
            main_data = topology.wq_data
        else:
            main_source = topology.nodes[index - 1]
            main_data = graph[main_source][node]
        side_edges = [
            (source, data)
            for source, _, data in graph.in_edges(node, data=True)
            if source != main_source
        ]
        if side_edges:
            metrics["intervening_confluence_count"] += 1

        main_strahler = _edge_number(main_data, "strahler")
        main_shreve = _edge_number(main_data, "shreve")
        main_flow_weight = _edge_weight(
            main_shreve,
            _lookup_flow_by_link(
                observed_flow_lookup,
                main_data.get("link_id"),
                main_data.get("hydrology_link_id"),
            ),
            flow_proxy_exponent,
        )
        incoming_shreves = [main_shreve]
        incoming_shreves.extend(
            _edge_number(data, "shreve") for _, data in side_edges
        )
        local_denominator = (
            sum(float(value) for value in incoming_shreves)
            if all(
                value is not None and float(value) >= 0
                for value in incoming_shreves
            )
            else None
        )

        side_item_indices: list[int] = []
        side_flow_weights = [
            _edge_weight(
                _edge_number(data, "shreve"),
                _lookup_flow_by_link(
                    observed_flow_lookup,
                    data.get("link_id"),
                    data.get("hydrology_link_id"),
                ),
                flow_proxy_exponent,
            )
            for _, data in side_edges
        ]
        all_side_denominator = None
        if main_flow_weight is not None and all(
            weight is not None and float(weight) >= 0
            for weight in side_flow_weights
        ):
            all_side_denominator = float(main_flow_weight) + sum(
                float(weight) for weight in side_flow_weights
            )

        for side_index, (_, data) in enumerate(side_edges):
            side_flow_weight = side_flow_weights[side_index]
            side_strahler = _edge_number(data, "strahler")
            side_shreve = _edge_number(data, "shreve")
            attribute_slots += 2
            attribute_values += int(side_strahler is not None) + int(
                side_shreve is not None
            )
            record = {
                "link_id": _serialisable_link_id(data.get("link_id")),
                "hydrology_link_id": _serialisable_link_id(
                    data.get("hydrology_link_id", data.get("link_id"))
                ),
                "strahler": side_strahler,
                "shreve": side_shreve,
                "raw_at_biology": (
                    float(side_shreve) / float(bio_shreve)
                    if side_shreve is not None
                    and bio_shreve is not None
                    and float(bio_shreve) > 0
                    else None
                ),
                "local_shreve": (
                    float(side_shreve) / local_denominator
                    if side_shreve is not None and local_denominator
                    else None
                ),
                "local_flow": None,
                "structural_major": (
                    side_strahler is not None
                    and main_strahler is not None
                    and float(side_strahler) >= float(main_strahler)
                ),
                "final_flow": None,
                "distance_to_biology_m": (
                    topology.node_to_biology_distance_m[index]
                    if topology.node_to_biology_distance_m is not None
                    and index < len(topology.node_to_biology_distance_m)
                    else None
                ),
            }
            side_item_indices.append(len(tributaries))
            tributaries.append(record)
            if side_flow_weight is not None and all_side_denominator is not None:
                record["local_flow"] = (
                    float(side_flow_weight) / all_side_denominator
                )

        if side_item_indices:
            confluence_side_indices[index] = side_item_indices

    metrics["intervening_tributary_count"] = len(tributaries)
    metrics["intervening_tributary_link_ids"] = [
        item["link_id"] for item in tributaries if item["link_id"] is not None
    ]
    metrics["intervening_tributary_hydrology_link_ids"] = [
        item["hydrology_link_id"]
        for item in tributaries
        if item["hydrology_link_id"] is not None
    ]
    metrics["intervening_tributary_missing_link_id_count"] = sum(
        item["link_id"] is None for item in tributaries
    )
    metrics["tributary_attribute_coverage_fraction"] = (
        attribute_values / attribute_slots if attribute_slots else 1.0
    )
    return tributaries, confluence_side_indices


# Most tributaries are not gauged. The network-size proxy gives a consistent
# indication of relative influence, but is kept separate from observed flow.
def _apply_flow_proxy(
    metrics: dict[str, object],
    graph: nx.DiGraph,
    topology: _RouteTopology,
    tributaries: list[dict[str, object]],
    confluence_side_indices: dict[int, list[int]],
    observed_flow_lookup: dict[int | float | str, float],
    flow_proxy_exponent: float,
    transport_decay_scale_m: float | None,
) -> bool:
    """Compose branch weights downstream and optionally apply distance decay."""
    flow_composition_available = bool(tributaries)
    downstream_main_fraction = 1.0
    for index in reversed(range(len(topology.nodes))):
        if index == 0:
            main_source = topology.wq_edge[0]
            main_data = topology.wq_data
        else:
            main_source = topology.nodes[index - 1]
            main_data = graph[main_source][topology.nodes[index]]
        side_indices = confluence_side_indices.get(index)
        if not side_indices:
            continue
        side_edges = [
            (source, data)
            for source, _, data in graph.in_edges(
                topology.nodes[index], data=True
            )
            if source != main_source
        ]
        if len(side_edges) != len(side_indices):
            flow_composition_available = False
            break

        main_flow_weight = _edge_weight(
            _edge_number(main_data, "shreve"),
            _lookup_flow_by_link(
                observed_flow_lookup,
                main_data.get("link_id"),
                main_data.get("hydrology_link_id"),
            ),
            flow_proxy_exponent,
        )
        side_weights: list[float] = []
        side_records: list[tuple[dict[str, object], float]] = []
        for side_index, (_, data) in zip(side_indices, side_edges):
            side_flow_weight = _edge_weight(
                _edge_number(data, "shreve"),
                _lookup_flow_by_link(
                    observed_flow_lookup,
                    data.get("link_id"),
                    data.get("hydrology_link_id"),
                ),
                flow_proxy_exponent,
            )
            if side_flow_weight is None:
                flow_composition_available = False
                break
            side_weights.append(float(side_flow_weight))
            side_records.append((tributaries[side_index], float(side_flow_weight)))

        if not flow_composition_available or main_flow_weight is None:
            flow_composition_available = False
            break
        total_weight = float(main_flow_weight) + sum(side_weights)
        if not math.isfinite(total_weight) or total_weight <= 0:
            flow_composition_available = False
            break

        main_share = float(main_flow_weight) / total_weight
        for side_record, side_weight in side_records:
            side_record["final_flow"] = (
                side_weight / total_weight * downstream_main_fraction
            )
        downstream_main_fraction *= main_share

    if not tributaries:
        downstream_main_fraction = 1.0
        flow_composition_available = True
    if flow_composition_available:
        metrics["estimated_flow_represented_fraction"] = downstream_main_fraction
        metrics["estimated_flow_unrepresented_fraction"] = max(
            0.0, 1.0 - downstream_main_fraction
        )
    else:
        metrics["estimated_flow_represented_fraction"] = None
        metrics["estimated_flow_unrepresented_fraction"] = None

    transport_decay_available = (
        flow_composition_available
        and transport_decay_scale_m is not None
        and transport_decay_scale_m > 0
        and all(
            item.get("distance_to_biology_m") is not None
            and math.isfinite(float(item["distance_to_biology_m"]))
            for item in tributaries
        )
    )
    if transport_decay_available:
        transport_unrepresented = 0.0
        for item in tributaries:
            distance = float(item["distance_to_biology_m"])
            decay = math.exp(-distance / float(transport_decay_scale_m))
            final_flow = item.get("final_flow")
            if final_flow is None:
                transport_decay_available = False
                break
            transport_unrepresented += float(final_flow) * decay
            item["transport_fraction_at_biology"] = float(final_flow) * decay
        if transport_decay_available:
            metrics["estimated_transport_unrepresented_fraction"] = min(
                1.0, max(0.0, transport_unrepresented)
            )
            metrics["estimated_transport_represented_fraction"] = (
                1.0 - metrics["estimated_transport_unrepresented_fraction"]
            )
    return flow_composition_available


def _summarise_tributary_decision(
    metrics: dict[str, object],
    tributaries: list[dict[str, object]],
    *,
    major_tributary_flow_fraction: float,
    flow_composition_available: bool,
) -> None:
    """Summarise magnitudes and assign the final tributary evidence class."""
    raw_fractions = [
        float(item["raw_at_biology"])
        for item in tributaries
        if item["raw_at_biology"] is not None
    ]
    local_shreve_fractions = [
        float(item["local_shreve"])
        for item in tributaries
        if item["local_shreve"] is not None
    ]
    local_flow_fractions = [
        float(item["local_flow"])
        for item in tributaries
        if item["local_flow"] is not None
    ]
    final_flow_fractions = [
        float(item["final_flow"])
        for item in tributaries
        if item["final_flow"] is not None
    ]
    if raw_fractions:
        metrics["largest_tributary_shreve_fraction"] = max(raw_fractions)
    elif not tributaries:
        metrics["largest_tributary_shreve_fraction"] = 0.0
    if local_shreve_fractions:
        metrics["largest_tributary_local_shreve_fraction"] = max(
            local_shreve_fractions
        )
    elif not tributaries:
        metrics["largest_tributary_local_shreve_fraction"] = 0.0
    if local_flow_fractions:
        metrics["largest_side_tributary_local_flow_fraction"] = max(
            local_flow_fractions
        )
    elif not tributaries:
        metrics["largest_side_tributary_local_flow_fraction"] = 0.0

    transport_fractions = [
        float(item["transport_fraction_at_biology"])
        for item in tributaries
        if item.get("transport_fraction_at_biology") is not None
    ]
    if transport_fractions:
        metrics["largest_side_tributary_transport_fraction_at_biology"] = max(
            transport_fractions
        )
    elif not tributaries:
        metrics["largest_side_tributary_transport_fraction_at_biology"] = 0.0
    if (
        metrics["estimated_transport_represented_fraction"] is None
        and metrics["estimated_transport_unrepresented_fraction"] is None
        and not transport_fractions
        and flow_composition_available
        and not tributaries
    ):
        metrics["estimated_transport_represented_fraction"] = 1.0
        metrics["estimated_transport_unrepresented_fraction"] = 0.0
    if final_flow_fractions:
        metrics["largest_side_tributary_estimated_flow_fraction_at_biology"] = max(
            final_flow_fractions
        )
    elif not tributaries:
        metrics["largest_side_tributary_estimated_flow_fraction_at_biology"] = 0.0

    structural_major = [item for item in tributaries if item["structural_major"]]
    influence_major = [
        item
        for item in tributaries
        if item["final_flow"] is not None
        and float(item["final_flow"]) >= major_tributary_flow_fraction
    ]
    major_ids: list[object] = []
    major_hydrology_ids: list[object] = []
    for item in structural_major + influence_major:
        if item["link_id"] is not None and item["link_id"] not in major_ids:
            major_ids.append(item["link_id"])
        if (
            item["hydrology_link_id"] is not None
            and item["hydrology_link_id"] not in major_hydrology_ids
        ):
            major_hydrology_ids.append(item["hydrology_link_id"])
    metrics.update({
        "structural_major_tributary_count": len(structural_major),
        "major_influence_tributary_count": len(influence_major),
        "major_tributary_count": len(
            {id(item) for item in structural_major + influence_major}
        ),
        "major_tributary_link_ids": major_ids,
        "major_tributary_hydrology_link_ids": major_hydrology_ids,
        "major_structural_tributary_between_sites": bool(structural_major),
        "combined_tributary_influence_major": (
            metrics["estimated_flow_unrepresented_fraction"] is not None
            and float(metrics["estimated_flow_unrepresented_fraction"])
            >= major_tributary_flow_fraction
        ),
    })
    metrics["major_influence_tributary_between_sites"] = bool(
        influence_major or metrics["combined_tributary_influence_major"]
    )
    metrics["major_tributary_between_sites"] = bool(
        metrics["major_structural_tributary_between_sites"]
        or metrics["major_influence_tributary_between_sites"]
    )

    if tributaries:
        candidates_with_raw = [
            item for item in tributaries if item["raw_at_biology"] is not None
        ]
        largest = (
            max(candidates_with_raw, key=lambda item: item["raw_at_biology"])
            if candidates_with_raw
            else tributaries[0]
        )
        metrics["largest_tributary_link_id"] = largest["link_id"]
        metrics["largest_tributary_hydrology_link_id"] = largest[
            "hydrology_link_id"
        ]

        candidates_with_estimated_flow = [
            item for item in tributaries if item["final_flow"] is not None
        ]
        if candidates_with_estimated_flow:
            largest_flow = max(
                candidates_with_estimated_flow,
                key=lambda item: item["final_flow"],
            )
            metrics["largest_side_tributary_estimated_flow_link_id"] = (
                largest_flow["link_id"]
            )
            metrics[
                "largest_side_tributary_estimated_flow_hydrology_link_id"
            ] = largest_flow["hydrology_link_id"]

    if not metrics["tributary_topology_complete"]:
        metrics["tributary_influence_class"] = "unknown"
        metrics["tributary_influence_method"] = (
            "directed_irn_topology_missing_bio_link"
        )
    elif not tributaries:
        metrics["tributary_influence_class"] = "none"
        metrics["tributary_influence_method"] = (
            "directed_irn_no_intervening_tributary"
        )
    elif metrics["major_tributary_between_sites"]:
        metrics["tributary_influence_class"] = "major"
        metrics["tributary_influence_method"] = (
            "directed_irn_confluences_with_compositional_nrfa_shreve_"
            "flow_proxy_not_observed_flow"
        )
    elif flow_composition_available:
        metrics["tributary_influence_class"] = "minor"
        metrics["tributary_influence_method"] = (
            "directed_irn_confluences_with_compositional_nrfa_shreve_"
            "flow_proxy_not_observed_flow"
        )
    else:
        metrics["tributary_influence_class"] = "unknown"
        metrics["tributary_influence_method"] = (
            "directed_irn_confluences_topology_or_order_attributes_only"
        )


# Pull the topology, flow proxy and final decision fields together for one
# candidate route. Each part is kept as its own helper above for readability.
def candidate_tributary_metrics(
    graph: nx.DiGraph,
    *,
    relationship: object,
    path_nodes: Sequence[Hashable] | None,
    wq_link_id: object,
    bio_link_id: object,
    bio_link_position_m: float | None = None,
    flow_proxy_exponent: float = FLOW_PROXY_EXPONENT,
    major_tributary_flow_fraction: float = MAJOR_TRIBUTARY_FLOW_FRACTION,
    transport_decay_scale_m: float | None = None,
    tributary_observed_flow_by_link_id: dict[object, object] | None = None,
) -> dict[str, object]:
    """Measure tributaries entering between a chemistry and biology site.

    ``path_nodes`` runs downstream from the end node of the WQ link to the
    start node of the biology link.  Every other incoming edge at those nodes
    is treated as a tributary mouth.  Results are plain Python scalars/lists so
    they can be written directly to tabular release outputs.
    """
    if not math.isfinite(flow_proxy_exponent) or flow_proxy_exponent <= 0:
        raise ValueError("flow_proxy_exponent must be finite and greater than zero")
    if (
        transport_decay_scale_m is not None
        and (not math.isfinite(float(transport_decay_scale_m)) or float(transport_decay_scale_m) <= 0)
    ):
        raise ValueError("tributary_transport_decay_scale_m must be finite and greater than zero if provided")
    if not 0 <= major_tributary_flow_fraction <= 1:
        raise ValueError("major_tributary_flow_fraction must be between zero and one")

    metrics = _base_metrics(flow_proxy_exponent, major_tributary_flow_fraction)
    observed_flow_lookup = _coerce_flow_lookup(tributary_observed_flow_by_link_id)
    relation = str(relationship) if relationship is not None else ""
    nodes = [] if path_nodes is None else list(path_nodes)
    if transport_decay_scale_m is not None:
        metrics["transport_decay_scale_m"] = float(transport_decay_scale_m)

    if relation == "same_link":
        _apply_same_link_topology(
            metrics,
            graph,
            nodes,
            wq_link_id,
            bio_link_id,
        )
        return metrics

    topology = _resolve_route_topology(
        metrics,
        graph,
        nodes,
        relationship=relation,
        wq_link_id=wq_link_id,
        bio_link_id=bio_link_id,
        bio_link_position_m=bio_link_position_m,
    )
    if topology is None:
        return metrics

    tributaries, confluence_side_indices = _collect_route_tributaries(
        metrics,
        graph,
        topology,
        observed_flow_lookup,
        flow_proxy_exponent,
    )

    flow_composition_available = _apply_flow_proxy(
        metrics,
        graph,
        topology,
        tributaries,
        confluence_side_indices,
        observed_flow_lookup,
        flow_proxy_exponent,
        transport_decay_scale_m,
    )

    _summarise_tributary_decision(
        metrics,
        tributaries,
        major_tributary_flow_fraction=major_tributary_flow_fraction,
        flow_composition_available=flow_composition_available,
    )
    return metrics


compute_tributary_influence = candidate_tributary_metrics
