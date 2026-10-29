"""Validate the river-network flow proxy against observed gauged flow.

The Integrated River Network's Shreve magnitude is an additive topology
measure, not an observed discharge.  This module keeps that distinction
explicit: it compares link magnitude with National River Flow Archive gauge
metadata and records the empirical exponent used to turn branch magnitudes
into relative, normalized flow-influence weights at a confluence.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from connected_river.spatial.irn import (
    edge_attributes_for_link,
    edge_link_ids,
    normalise_irn_link_id,
)

DEFAULT_FLOW_PROXY_EXPONENT = 0.72


def _edge_attributes_by_link(
    graph: nx.DiGraph,
) -> dict[int | str, dict[str, object]]:
    """Index every raw source link, including collapsed parallel aliases."""
    indexed: dict[int | str, dict[str, object]] = {}
    for _start, _end, attributes in graph.edges(data=True):
        primary_id = normalise_irn_link_id(attributes.get("link_id"))
        for raw_link_id in edge_link_ids(attributes):
            link_id = normalise_irn_link_id(raw_link_id)
            if link_id is None:
                continue
            resolved = edge_attributes_for_link(attributes, raw_link_id)
            if resolved is None:
                continue
            resolved["irn_link_resolution"] = (
                "primary_link_id"
                if primary_id == link_id
                else "parallel_link_alias"
            )
            indexed[link_id] = resolved
    return indexed


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    # Shreve magnitude, catchment area and mean discharge are meaningful for
    # this log-scale validation only when strictly positive. IRN zero is an
    # unavailable/special-link value, not a zero-flow observation.
    valid = left.notna() & right.notna() & left.gt(0) & right.gt(0)
    if int(valid.sum()) < 3:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return value if math.isfinite(value) else None


# Compare the network-size proxy with observed gauge relationships where both
# are available. This checks usefulness without pretending the proxy is flow.
def validate_tributary_flow_proxy(
    gauge_snaps: pd.DataFrame,
    station_catalogue: pd.DataFrame,
    graph: nx.DiGraph,
    *,
    flow_proxy_exponent: float = DEFAULT_FLOW_PROXY_EXPONENT,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare snapped-link Shreve magnitude with observed gauge metadata.

    Returns a machine-readable summary and one audit row per successfully
    joined gauge.  The fitted relationship is descriptive validation only;
    the configured exponent remains fixed so annual updates do not silently
    rewrite earlier match scores.
    """
    if flow_proxy_exponent <= 0:
        raise ValueError("flow_proxy_exponent must be positive")

    catalogue = station_catalogue.copy()
    station_id = "nrfa_station_id" if "nrfa_station_id" in catalogue else "id"
    if station_id not in catalogue:
        raise ValueError("station catalogue needs an id or nrfa_station_id column")
    if "nrfa_station_id" not in gauge_snaps:
        raise ValueError("gauge snaps need an nrfa_station_id column")

    # ``nrfa-mean-flow`` is a catalogue availability flag, not a discharge.
    mean_flow_column = "gdf-mean-flow" if "gdf-mean-flow" in catalogue else None
    catchment_column = next(
        (name for name in ("catchment-area", "ihdtm-catchment-area") if name in catalogue),
        None,
    )
    keep = [station_id]
    keep.extend(name for name in (mean_flow_column, catchment_column) if name is not None)
    metadata = catalogue[keep].rename(columns={station_id: "nrfa_station_id"}).copy()
    metadata["nrfa_station_id"] = pd.to_numeric(
        metadata["nrfa_station_id"], errors="coerce",
    ).astype("Int64")

    evidence = gauge_snaps.copy()
    evidence["nrfa_station_id"] = pd.to_numeric(
        evidence["nrfa_station_id"], errors="coerce",
    ).astype("Int64")
    evidence = evidence.merge(metadata, on="nrfa_station_id", how="left", validate="many_to_one")

    edge_lookup = _edge_attributes_by_link(graph)
    link_ids = evidence.get("snapped_link_id", pd.Series(index=evidence.index, dtype="object"))
    edge_attributes = []
    for value in link_ids:
        link_id = normalise_irn_link_id(value)
        edge_attributes.append(edge_lookup.get(link_id, {}) if link_id is not None else {})
    evidence["irn_shreve_magnitude"] = pd.to_numeric(
        pd.Series([attributes.get("shreve") for attributes in edge_attributes], index=evidence.index),
        errors="coerce",
    )
    evidence["irn_strahler_order"] = pd.to_numeric(
        pd.Series([attributes.get("strahler") for attributes in edge_attributes], index=evidence.index),
        errors="coerce",
    )
    evidence["irn_link_resolution"] = pd.Series(
        [attributes.get("irn_link_resolution", "unavailable") for attributes in edge_attributes],
        index=evidence.index,
        dtype="string",
    )
    evidence["observed_mean_flow_m3s"] = (
        pd.to_numeric(evidence[mean_flow_column], errors="coerce")
        if mean_flow_column is not None
        else np.nan
    )
    evidence["catalogue_catchment_area_km2"] = (
        pd.to_numeric(evidence[catchment_column], errors="coerce")
        if catchment_column is not None
        else np.nan
    )

    positive = (
        evidence["irn_shreve_magnitude"].gt(0)
        & evidence["observed_mean_flow_m3s"].gt(0)
    )
    fitted = evidence.loc[positive].copy()
    fitted_exponent: float | None = None
    fitted_intercept: float | None = None
    median_factor_error: float | None = None
    p90_factor_error: float | None = None
    if len(fitted) >= 3 and fitted["irn_shreve_magnitude"].nunique() >= 2:
        log_shreve = np.log(fitted["irn_shreve_magnitude"].to_numpy(dtype=float))
        log_flow = np.log(fitted["observed_mean_flow_m3s"].to_numpy(dtype=float))
        fitted_exponent, fitted_intercept = [float(value) for value in np.polyfit(log_shreve, log_flow, 1)]
        predicted_log_flow = fitted_intercept + fitted_exponent * log_shreve
        factor_error = np.exp(np.abs(log_flow - predicted_log_flow))
        median_factor_error = float(np.median(factor_error))
        p90_factor_error = float(np.quantile(factor_error, 0.9))

    snapped = evidence.get("snap_distance_m", pd.Series(index=evidence.index, dtype=float)).notna()
    positive_shreve = evidence["irn_shreve_magnitude"].gt(0)
    summary: dict[str, object] = {
        "method": "NRFA gauge validation of additive IRN Shreve magnitude",
        "interpretation": (
            "Calibrated exponent for normalized network-size weights at confluences; "
            "not observed flow at an ungauged tributary and not a substitute for local hydrology."
        ),
        "fixed_flow_proxy_exponent": float(flow_proxy_exponent),
        "station_catalogue_rows": len(catalogue),
        "gauge_snap_rows": len(evidence),
        "gauges_with_valid_snap": int(snapped.sum()),
        "gauges_with_positive_shreve": int((snapped & positive_shreve).sum()),
        "gauges_resolved_from_parallel_alias": int(
            evidence["irn_link_resolution"].eq("parallel_link_alias").sum()
        ),
        "positive_shreve_coverage_of_snapped": (
            float((snapped & positive_shreve).sum() / snapped.sum()) if snapped.any() else None
        ),
        "gauges_in_mean_flow_fit": len(fitted),
        "correlation_filter": (
            "strictly positive Shreve/catchment-area and observed mean-flow values"
        ),
        "spearman_shreve_vs_observed_mean_flow": _rank_correlation(
            evidence["irn_shreve_magnitude"], evidence["observed_mean_flow_m3s"],
        ),
        "spearman_shreve_vs_catalogue_catchment_area": _rank_correlation(
            evidence["irn_shreve_magnitude"], evidence["catalogue_catchment_area_km2"],
        ),
        "spearman_catchment_area_vs_observed_mean_flow": _rank_correlation(
            evidence["catalogue_catchment_area_km2"], evidence["observed_mean_flow_m3s"],
        ),
        "fitted_log_flow_on_log_shreve_exponent": fitted_exponent,
        "fitted_log_flow_on_log_shreve_intercept": fitted_intercept,
        "median_prediction_factor_error": median_factor_error,
        "p90_prediction_factor_error": p90_factor_error,
        "calibration_policy": (
            "The fixed exponent is release configuration. Observed annual validation is "
            "reported but does not automatically change historical scores."
        ),
    }
    audit_columns = [
        "nrfa_station_id", "snapped_link_id", "snap_distance_m", "irn_link_resolution",
        "irn_shreve_magnitude", "irn_strahler_order", "observed_mean_flow_m3s",
        "catalogue_catchment_area_km2",
    ]
    return summary, evidence.reindex(columns=audit_columns)
