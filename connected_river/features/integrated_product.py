"""Build the survey-level connected-river product."""

from __future__ import annotations

import numpy as np
import pandas as pd

from connected_river.chemistry_registry import (
    TARGET_DETERMINANDS,
    additional_target_determinands,
    exact_feature_columns,
)
from connected_river.features.flow_features import build_survey_antecedent_nrfa_features
from connected_river.features.model_table import build_pressure_indicators


def _normalise_object_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Restore logical booleans and use nullable strings for remaining text."""
    out = table.copy()
    for column in out.select_dtypes(include=["object", "string"]):
        values = out[column].astype("string")
        normalised = values.str.strip().str.casefold()
        observed = set(normalised.dropna().unique())
        if observed and observed <= {"true", "false"}:
            out[column] = normalised.map({"true": True, "false": False}).astype(
                "boolean"
            )
        else:
            out[column] = values
    return out


def _numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(np.nan, index=table.index, dtype="float64")
    return pd.to_numeric(table[column], errors="coerce")


# Tidy the shared fields after all sources have been joined. Source-specific
# detail stays in the full table rather than being silently discarded.
def harmonise_connected_river_product(table: pd.DataFrame) -> pd.DataFrame:
    """Add compact interpretation fields to the current-source product."""
    out = table.copy()

    analysis_aliases: dict[str, pd.Series] = {}
    for item in TARGET_DETERMINANDS:
        target = item.key
        # Compact analysis aliases are always six-month medians and always use
        # the unit encoded by the target name (metals: mg/L).
        if f"{target}_median_6mo" in out:
            analysis_aliases[f"{target}_median"] = _numeric(out, f"{target}_median_6mo")

    if analysis_aliases:
        out = pd.concat([out, pd.DataFrame(analysis_aliases, index=out.index)], axis=1)
    missing_exact = [column for column in exact_feature_columns() if column not in out]
    if missing_exact:
        out = pd.concat([
            out,
            pd.DataFrame(float("nan"), index=out.index, columns=missing_exact),
        ], axis=1)
    # Consolidate the very wide table before adding the small set of release
    # interpretation fields below. This avoids pandas fragmentation warnings
    # during annual builds without changing values or column order semantics.
    out = out.copy()

    additional_count_columns = {
        f"{item.key}_observation_count_6mo"
        for item in additional_target_determinands()
    }
    automated_count_columns = [
        column for column in out
        if column.endswith("_observation_count_6mo")
        and not column.startswith("target_exact_")
        and column not in additional_count_columns
    ]
    exact_count_columns = [
        column for column in out
        if column.startswith("target_exact_")
        and column.endswith("_observation_count_6mo")
    ]
    automated_counts = out[automated_count_columns].apply(pd.to_numeric, errors="coerce")
    observed_breadth = automated_counts.gt(0).sum(axis=1)
    existing_breadth = _numeric(out, "chemistry_determinand_count_6mo")
    out["chemistry_determinand_count_6mo"] = existing_breadth.fillna(
        observed_breadth
    ).astype("Int64")
    exact_breadth = out[exact_count_columns].apply(pd.to_numeric, errors="coerce").gt(0).sum(axis=1)
    existing_exact_breadth = _numeric(out, "target_exact_determinand_count_6mo")
    out["target_exact_determinand_count_6mo"] = existing_exact_breadth.fillna(
        exact_breadth
    ).astype("Int64")
    automated_frequency = automated_counts.max(axis=1, skipna=True)
    out["wq_observation_count_max_6mo"] = automated_frequency
    automated_flow = _numeric(out, "nrfa_flow_n_6mo").gt(0)
    out["flow_evidence_available"] = automated_flow

    if "integration_unit" not in out:
        out["integration_unit"] = "biological survey"
    else:
        out["integration_unit"] = out["integration_unit"].fillna("biological survey")
    if "match_decision" not in out:
        out["match_decision"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if "match_use_recommended" not in out:
        out["match_use_recommended"] = pd.Series(pd.NA, index=out.index, dtype="boolean")

    breadth = _numeric(out, "chemistry_determinand_count_6mo")
    recommended = out["match_use_recommended"].astype("boolean").fillna(False)
    excluded = ~recommended
    pending_review = out.get(
        "scientific_decision", pd.Series(pd.NA, index=out.index, dtype="string")
    ).astype("string").eq("review")
    no_candidate = out["match_decision"].astype("string").eq("no_candidate_match").fillna(False)
    out["chemistry_included_in_release"] = breadth.gt(0) & ~excluded
    out["chemistry_release_status"] = "no_observations_in_six_month_window"
    out.loc[out["chemistry_included_in_release"], "chemistry_release_status"] = "included"
    out.loc[excluded, "chemistry_release_status"] = "excluded_no_defensible_match"
    out.loc[pending_review, "chemistry_release_status"] = (
        "excluded_pending_scientific_review"
    )
    out.loc[no_candidate, "chemistry_release_status"] = "no_candidate_match"
    return out


# Keep review and no-match rows in the full product, but blank chemistry values
# that should not be used as accepted analytical evidence.
def exclude_no_defensible_chemistry(table: pd.DataFrame) -> pd.DataFrame:
    """Retain survey rows and match diagnostics, but suppress rejected evidence."""
    out = table.copy()
    if "match_decision" not in out:
        return out
    rejected = out["match_decision"].astype("string").eq("no_defensible_match")
    if "match_use_recommended" in out:
        rejected |= ~out["match_use_recommended"].astype("boolean").fillna(False)
    chemistry_fields = [
        column for column in out
        if column.endswith((
            "_observation_count_6mo", "_minimum_6mo", "_maximum_6mo",
            "_median_6mo", "_mean_6mo", "_std_6mo",
            "_below_loq_count_6mo", "_below_loq_fraction_6mo",
            "_above_range_count_6mo", "_above_range_fraction_6mo",
        ))
    ]
    out.loc[rejected, chemistry_fields] = np.nan
    if "chemistry_determinand_count_6mo" in out:
        out.loc[rejected, "chemistry_determinand_count_6mo"] = 0
    if "target_exact_determinand_count_6mo" in out:
        out.loc[rejected, "target_exact_determinand_count_6mo"] = 0
    return out


# This is where the separate evidence tables finally become one survey-level
# dataset. Joins are checked as they happen to avoid quietly multiplying rows.
def build_connected_river_product(
    chemistry_survey: pd.DataFrame,
    selected_gauges: pd.DataFrame,
    daily_flows: pd.DataFrame,
    site_year_analysis: pd.DataFrame,
    automated_start_year: int | None = None,
    flow_reference_start: str | pd.Timestamp = "1974-05-20",
    flow_reference_end: str | pd.Timestamp = "2019-03-29",
    minimum_flow_reference_coverage: float = 0.5,
) -> pd.DataFrame:
    """Combine biological surveys with antecedent chemistry and NRFA context.

    The default six-calendar-month summary was inspired by the published study,
    while all rows and observations are acquired by this reusable pipeline.
    Site matching always uses the configured river-network method.
    """
    product = chemistry_survey.copy()
    if automated_start_year is not None:
        product = product[
            pd.to_numeric(product["year"], errors="coerce").ge(int(automated_start_year))
        ].copy()
    record_key = "integration_record_id"
    if record_key not in product:
        raise ValueError("Connected-river survey table requires integration_record_id")
    flow = build_survey_antecedent_nrfa_features(
        product[[record_key, "bio_site_id", "year", "sample_date"]],
        selected_gauges,
        daily_flows,
        record_id_column=record_key,
        reference_start=flow_reference_start,
        reference_end=flow_reference_end,
        minimum_reference_coverage=minimum_flow_reference_coverage,
    )
    product = product.merge(flow, on=record_key, how="left", validate="one_to_one")

    context_names = {
        "bio_site_id", "year", "path_edm_assets", "path_edm_crosswalked_assets",
        "path_edm_location_provenance", "path_counted_spills",
        "path_spill_duration_hours", "path_attribution_method",
        "path_permitted_sewage_outlets", "path_treatment_works",
        "path_water_company_outlets", "chemical_status", "ecological_status",
        "overall_status", "pressure_tier_3", "match_rule_failures",
        "match_review_reasons", "match_use_recommended", "match_decision",
        "scientific_decision", "candidate_gate_decision",
        "candidate_rejection_reasons", "candidate_review_reasons",
        "algorithmic_match_rank", "algorithmic_selection_confidence",
        "scientific_match_rank", "scientific_candidate_count",
        "scientific_runner_up_score", "scientific_score_margin",
        "scientific_selection_method",
        "route_modifier_screen_status", "evidence_strength_percent",
        "network_direction", "scientific_evidence_role",
        "candidate_direction_rule", "direction_policy",
        "tributary_acceptance_policy",
        "strong_proxy_major_tributary", "crosses_reservoir",
        "crosses_tidal_boundary", "crosses_major_abstraction",
        "evidence_match_support", "evidence_chemistry_dates_support",
        "evidence_chemistry_breadth_support", "evidence_nrfa_flow_support",
        "indicator_interpretation", "pressure_reference_cohort",
        "network_relationship", "bio_snap_distance_m", "wq_snap_distance_m",
        "wq_n_sampling_dates", "wq_n_months_sampled", "wq_n_target_determinands",
        "wq_has_core_chemistry", "wq_has_metals", "prefilter_euclidean_distance_m",
        "score_distance", "score_snap", "score_direction", "score_direction_method",
        "score_temporal", "score_chemistry", "score_transport_plausibility",
        "transport_evidence_fraction", "score_transport_plausibility_method",
        "match_score", "match_class", "candidate_count",
        "runner_up_score", "score_margin", "distance_sensitive",
        "tributary_penalty_sensitive",
        "match_class_acceptable", "sampling_dates_adequate",
        "chemistry_breadth_adequate", "biology_snap_acceptable",
        "chemistry_snap_acceptable",
    }
    tributary_prefixes = (
        "tributary_", "side_tributary_", "intervening_", "represented_shreve_",
        "unrepresented_shreve_", "estimated_flow_", "largest_tributary_",
        "largest_side_tributary_", "major_tributary_",
    )
    tributary_fields = {
        "wq_strahler", "wq_shreve", "bio_strahler", "bio_shreve",
        "match_score_before_tributary", "match_rank_before_tributary",
        "score_tributary_penalty", "score_tributary_penalty_method",
        "tributary_penalty_fraction_used", "tributary_penalty_evidence_available",
        "tributary_penalty_points_max", "winner_changed_by_tributary",
        "flow_proxy_exponent", "structural_major_tributary_count",
        "combined_tributary_influence_major", "shreve_monotonic_downstream",
        "strahler_monotonic_downstream", "network_order_monotonic",
    }
    context_columns = [
        column
        for column in site_year_analysis
        if column in context_names
        or column in tributary_fields
        or column.startswith(tributary_prefixes)
    ]
    context = site_year_analysis[context_columns].drop_duplicates(["bio_site_id", "year"])
    product = product.merge(context, on=["bio_site_id", "year"], how="left", validate="many_to_one")
    product = exclude_no_defensible_chemistry(product)
    product.insert(0, "data_origin", "connected_river_pipeline")
    product.insert(1, "processing_method", "Antecedent chemistry summaries with directed river-network matching")
    product.insert(2, "integration_unit", "biological survey")
    if product.duplicated(record_key).any():
        raise ValueError("Connected-river product must contain one row per biological survey")
    product = harmonise_connected_river_product(product)
    product = build_pressure_indicators(product, cohort_column="data_origin")
    product = _normalise_object_columns(product)
    return product.sort_values(["sample_year", "data_origin", "assemblage", "site_id"], na_position="last").reset_index(drop=True)
