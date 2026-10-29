"""Create the human-scale CSV view of the lossless survey product."""

from __future__ import annotations

import re

import pandas as pd

from connected_river.chemistry_registry import determinand_for_feature

IDENTITY = [
    "integration_record_id", "data_origin", "processing_method", "assemblage",
    "sample_id", "sample_date", "sample_year", "site_id", "bio_site_id",
    "site_label", "sample_method", "analysis_id", "integration_unit",
    "catchment", "water_body", "wfd_waterbody_id", "reporting_area",
    "easting", "northing", "longitude", "latitude", "first_sample_date",
    "last_sample_date", "source_sample_count",
]

BIOLOGY = [
    "taxon_richness", "abundance_total", "taxon_record_count", "metric_count",
    "metric_value_count", "ept_taxon_richness", "non_native_taxon_richness",
    "protected_taxon_richness",
]

MATCH_AND_CONTEXT = [
    "wq_site_id", "wq_site_name", "network_relationship", "network_direction",
    "scientific_evidence_role", "candidate_direction_rule",
    "direction_policy", "network_distance_m",
    "prefilter_euclidean_distance_m", "bio_snap_distance_m", "wq_snap_distance_m",
    "match_rank", "match_score", "match_class", "selection_confidence",
    "score_direction", "score_direction_method",
    "match_score_before_tributary", "score_transport_plausibility",
    "transport_evidence_fraction", "score_transport_plausibility_method",
    "score_tributary_penalty",
    "tributary_penalty_fraction_used", "tributary_penalty_evidence_available",
    "score_tributary_penalty_method", "tributary_penalty_points_max",
    "match_rank_before_tributary", "winner_changed_by_tributary",
    "represented_shreve_fraction", "unrepresented_shreve_fraction",
    "estimated_flow_represented_fraction", "estimated_flow_unrepresented_fraction",
    "intervening_confluence_count", "intervening_tributary_count",
    "intervening_tributary_link_ids",
    "intervening_tributary_hydrology_link_ids",
    "largest_side_tributary_estimated_flow_fraction_at_biology",
    "largest_side_tributary_estimated_flow_link_id",
    "largest_side_tributary_estimated_flow_hydrology_link_id",
    "largest_tributary_link_id", "largest_tributary_hydrology_link_id",
    "largest_side_tributary_local_flow_fraction", "major_tributary_count",
    "major_tributary_link_ids", "major_tributary_hydrology_link_ids",
    "structural_major_tributary_count", "major_influence_tributary_count",
    "major_structural_tributary_between_sites",
    "major_influence_tributary_between_sites", "combined_tributary_influence_major",
    "major_tributary_between_sites", "tributary_influence_class",
    "tributary_influence_method", "shreve_monotonic_downstream",
    "strahler_monotonic_downstream", "network_order_monotonic",
    "tributary_topology_complete", "tributary_attribute_coverage_fraction",
    "tributary_wq_link_resolution", "tributary_bio_link_resolution",
    "tributary_parallel_alias_resolved", "flow_proxy_exponent",
    "major_tributary_flow_fraction_threshold",
    "tributary_penalty_sensitive",
    "candidate_count", "runner_up_score", "score_margin", "algorithmic_match_rank",
    "algorithmic_selection_confidence", "scientific_match_rank",
    "scientific_candidate_count", "scientific_runner_up_score",
    "scientific_score_margin",
    "candidate_gate_decision", "candidate_rejection_reasons",
    "candidate_review_reasons", "scientific_selection_method",
    "scientific_decision", "match_decision", "match_rule_failures",
    "match_review_reasons", "match_use_recommended",
    "tributary_acceptance_policy",
    "strong_proxy_major_tributary", "route_modifier_screen_status",
    "crosses_reservoir", "crosses_tidal_boundary", "crosses_major_abstraction",
    "wq_n_sampling_dates",
    "wq_n_months_sampled", "wq_n_target_determinands", "wq_has_core_chemistry",
    "wq_has_metals", "chemistry_window_months", "chemistry_window_start",
    "chemistry_window_end", "chemistry_temporal_method", "below_loq_treatment",
    "chemistry_determinand_count_6mo", "chemistry_breadth_adequate",
    "chemistry_included_in_release", "chemistry_release_status",
    "path_edm_assets", "path_edm_crosswalked_assets",
    "path_edm_location_provenance", "path_counted_spills",
    "path_spill_duration_hours",
    "path_attribution_method", "path_permitted_sewage_outlets",
    "path_treatment_works", "path_water_company_outlets", "chemical_status",
    "side_tributary_edm_assets", "side_tributary_edm_crosswalked_assets",
    "side_tributary_edm_location_provenance", "side_tributary_counted_spills",
    "side_tributary_spill_duration_hours", "side_tributary_attribution_method",
    "side_tributary_permitted_sewage_outlets", "side_tributary_treatment_works",
    "side_tributary_water_company_outlets", "side_tributary_context_method",
    "side_tributary_upstream_cutoff_m",
    "chemical_status_year", "ecological_status", "ecological_status_year",
    "overall_status", "overall_status_year", "pressure_tier_3", "distance_sensitive",
    "relative_nutrient_pressure", "relative_organic_oxygen_stress",
    "relative_metals_pressure", "relative_edm_pressure", "indicator_interpretation",
    "pressure_reference_cohort",
    "evidence_match_support", "evidence_chemistry_dates_support",
    "evidence_chemistry_breadth_support", "evidence_chemistry_frequency_support",
    "evidence_nrfa_flow_support", "evidence_strength_percent",
]

FLOW = [
    "nrfa_station_id", "nrfa_station_name", "nrfa_river",
    "nrfa_network_distance_m", "nrfa_relationship", "nrfa_selection_reason",
    "nrfa_window_start", "nrfa_window_end", "nrfa_flow_n_6mo",
    "nrfa_flow_coverage_6mo", "nrfa_flow_min_6mo_m3s", "nrfa_flow_max_6mo_m3s",
    "nrfa_flow_mean_6mo_m3s", "nrfa_flow_std_6mo_m3s",
    "nrfa_flow_median_6mo_m3s", "nrfa_flow_q05_6mo_m3s",
    "nrfa_flow_q95_6mo_m3s", "flow_iha_reference_coverage",
    "flow_iha_reference_eligible", "flow_iha_reference_reason",
    "flow_iha_window_coverage", "flow_iha_seasonal_reference_coverage",
    "flow_reference_rf_mean_poi", "flow_reference_rf_q50_poi",
    "flow_reference_rf_std_poi", "flow_reference_rf_cov_poi",
    "flow_reference_rf_q5_mean_poi", "flow_reference_rf_q95_mean_poi",
    "flow_iha_survey_mean_to_poi_mean_ratio",
]


def _analysis_columns(columns: list[str]) -> list[str]:
    biology_metrics = [name for name in columns if name.startswith("biology_metric_")]
    chemistry = [
        name for name in columns
        if re.search(r"_(observation_count|median|mean)_6mo$", name)
        and not name.startswith("nrfa_")
        and not name.startswith("target_exact_")
    ]
    ordered = IDENTITY + BIOLOGY + biology_metrics + MATCH_AND_CONTEXT + chemistry + FLOW
    return list(dict.fromkeys(name for name in ordered if name in columns))


# The full product is useful for checking the method, but quite wide. This
# smaller view keeps the fields most people will need for analysis.
def build_analysis_ready_product(full: pd.DataFrame) -> pd.DataFrame:
    """Select and order interpretable fields without changing any survey row."""
    if "integration_record_id" not in full:
        raise ValueError("Full product has no integration_record_id")
    if full["integration_record_id"].isna().any() or full["integration_record_id"].duplicated().any():
        raise ValueError("integration_record_id must be complete and unique")
    selected = full.loc[:, _analysis_columns(list(full.columns))].copy()
    selected = selected.dropna(axis="columns", how="all")
    if len(selected) != len(full):
        raise AssertionError("Analysis-ready selection changed the survey row count")
    return selected


def _group(column: str) -> str:
    if column in IDENTITY:
        return "site and source details"
    if column in BIOLOGY or column.startswith("biology_metric_"):
        return "biology"
    if column in FLOW or column.startswith(("nrfa_", "flow_")):
        return "NRFA flow"
    if re.search(r"_(observation_count|median|mean)_6mo$", column):
        return "six-month chemistry"
    if column.startswith(("path_", "side_tributary_")):
        return "wastewater context"
    if column in {"chemical_status", "ecological_status", "overall_status", "pressure_tier_3"}:
        return "WFD context"
    if column.startswith("evidence_"):
        return "evidence quality"
    return "matching and interpretation"


def _definition(column: str) -> str:
    replacements = {
        "integration_record_id": "Stable unique identifier for this survey-level integrated record.",
        "data_origin": "Named source-processing cohort for this pipeline-generated row.",
        "sample_date": "Date anchoring the biological survey and its antecedent evidence windows.",
        "evidence_strength_percent": "Coverage score (0–100) combining match, chemistry timing, chemistry breadth/frequency and NRFA-flow support; it measures evidence completeness, not river health.",
        "network_distance_m": "Along-river distance between the biology site and selected chemistry site, in metres.",
        "chemistry_window_start": "Start of the inclusive six-calendar-month chemistry window before the survey.",
        "chemistry_window_end": "End of the chemistry window: the biological survey date.",
        "processing_method": "Named workflow that created the integrated row.",
        "assemblage": "Biological monitoring group represented by the survey.",
        "bio_site_id": "Assemblage-qualified biological-site identifier; use this rather than raw site_id for cross-programme counts.",
        "site_id": "Publisher site identifier within its biological monitoring programme.",
        "sample_year": "Calendar year of the biological survey date.",
        "taxon_richness": "Number of distinct taxa recorded for the survey at the available taxonomic resolution.",
        "abundance_total": "Sum of supplied organism abundance or count values for the survey.",
        "taxon_record_count": "Number of taxon rows contributing to the survey record.",
        "ept_taxon_richness": "Distinct mayfly, stonefly and caddisfly taxa recorded in an applicable macroinvertebrate survey.",
        "wq_site_id": "Identifier of the selected chemistry monitoring station; missing when no candidate passes the matching decision.",
        "network_relationship": "Directed river relationship between selected chemistry and biological sites.",
        "network_direction": "Physical direction of the chemistry station relative to biology; unlike network_relationship, same-link routes retain upstream or downstream direction explicitly.",
        "scientific_evidence_role": "Local river-chemistry context role assigned to the candidate, with upstream or downstream direction retained explicitly.",
        "score_direction": "Direction component kept for reference. It is fixed at zero because upstream and downstream candidates are both admissible and direction alone is not evidence of quality.",
        "score_direction_method": "Versioned statement of the bidirectional no-preference direction policy used in scoring.",
        "candidate_gate_decision": "Check before selection: accept, review or reject. An acceptable candidate is chosen even when another candidate has a slightly higher score.",
        "algorithmic_match_rank": "Candidate's original score rank before scientific eligibility is used to select an acceptable route.",
        "algorithmic_selection_confidence": "Score-margin confidence around the unfiltered algorithmic winner; kept for comparison and not used as confidence in the final scientific selection.",
        "scientific_match_rank": "Candidate rank within the best available scientific decision pool: acceptable first, otherwise review, otherwise rejected fallback.",
        "scientific_candidate_count": "Number of candidates competing in the selected scientific decision pool for the site-year.",
        "scientific_runner_up_score": "Score of the second candidate in the selected scientific decision pool, when present.",
        "scientific_score_margin": "Score gap between the selected candidate and runner-up in the selected scientific decision pool.",
        "scientific_decision": "Final accept, review or reject decision after available route modifiers are applied.",
        "match_review_reasons": "Semicolon-separated reasons why a visible candidate requires scientific review and cannot include chemistry automatically.",
        "route_modifier_screen_status": "Declares which optional reservoir, tidal and abstraction route layers were supplied; missing layers are not interpreted as absence.",
    "match_rank": "Candidate order under the declared matching score; rank 1 is best available, not guaranteed truth.",
    "match_score": "Transparent non-saturating 0–100 ranking score combining river-network distance, coordinate fit, monitoring support and represented-flow evidence. It orders candidates but is not an eligibility check.",
    "match_score_before_tributary": "Candidate score before the incoming-river representativeness adjustment.",
    "score_transport_plausibility": "Water-transport plausibility points based on represented-flow evidence only; along-river distance is scored separately and is not counted again here.",
    "transport_evidence_fraction": "Estimated route flow-share proxy used for transport plausibility scoring, from estimated flow share then fallback Shreve proxy.",
    "score_transport_plausibility_method": "Method used to derive transport plausibility score points.",
    "score_tributary_penalty": "Points deducted because mapped tributaries enter between the chemistry and biology sites; a transparent heuristic, not a probability.",
        "tributary_penalty_fraction_used": "Unrepresented incoming-river influence fraction used to calculate the score penalty; the calibrated flow proxy is preferred and raw Shreve share is the declared fallback.",
        "tributary_penalty_evidence_available": "Whether an incoming-river influence fraction was available for scoring. False means a zero-point penalty reflects unavailable evidence, not known zero influence.",
        "estimated_flow_represented_fraction": "Estimated fraction of network-flow influence at the biology site represented by the chemistry route, using normalized gauge-calibrated Shreve weights.",
        "estimated_flow_unrepresented_fraction": "Estimated fraction of network-flow influence entering from other mapped branches between the chemistry and biology sites; one minus the represented fraction.",
        "represented_shreve_fraction": "Raw additive Shreve-magnitude share represented by the chemistry route before flow-proxy calibration.",
        "intervening_tributary_count": "Number of mapped incoming river branches joining between the chemistry station and biology site.",
        "intervening_tributary_link_ids": "Shortest routing-link identifiers for mapped incoming branches; compare with intervening_tributary_hydrology_link_ids where parallel raw IRN links share graph nodes.",
        "intervening_tributary_hydrology_link_ids": "IRN raw link identifiers supplying the representative Shreve and Strahler attributes for intervening branches; these may differ from shortest routing-link identifiers where parallel raw links share graph nodes.",
        "largest_side_tributary_estimated_flow_fraction_at_biology": "Estimated final influence share of the single largest intervening tributary at the biology site after subsequent confluences are included.",
        "largest_tributary_link_id": "Shortest routing-link identifier for the largest tributary under the raw additive Shreve proxy.",
        "largest_tributary_hydrology_link_id": "IRN raw link identifier supplying the Shreve attributes for the largest tributary under the raw additive proxy.",
        "largest_side_tributary_estimated_flow_link_id": "Shortest routing-link identifier for the tributary with the largest estimated final flow influence.",
        "largest_side_tributary_estimated_flow_hydrology_link_id": "IRN raw link identifier supplying the Shreve attributes for the tributary with the largest estimated final flow influence.",
        "major_tributary_link_ids": "Shortest routing-link identifiers for tributaries meeting a structural or estimated-influence major rule.",
        "major_tributary_hydrology_link_ids": "IRN raw link identifiers supplying representative hydrologic attributes for tributaries meeting a structural or estimated-influence major rule.",
        "major_tributary_between_sites": "Whether at least one intervening branch meets the declared structural or estimated-influence major-tributary rule.",
        "tributary_influence_class": "Plain decision class for intervening tributary influence: none, minor, major or unknown.",
        "tributary_wq_link_resolution": "How the chemistry station's raw Integrated River Network link identifier was resolved in the routing graph, including retained parallel-link aliases.",
        "tributary_bio_link_resolution": "How the biology site's raw Integrated River Network link identifier was resolved in the routing graph, including retained parallel-link aliases.",
        "tributary_parallel_alias_resolved": "Whether either monitoring-site link was recovered from the recorded parallel-link alias map rather than the routing edge's primary identifier.",
        "flow_proxy_exponent": "Fixed exponent applied to positive Shreve magnitude before branch weights are normalized at each confluence.",
        "major_tributary_flow_fraction_threshold": "Declared represented-flow-proxy threshold at which one branch or combined incoming influence is treated as major.",
        "side_tributary_edm_assets": "Distinct annual event-duration-monitoring assets on bounded upstream reaches of tributaries entering between the matched sites.",
        "side_tributary_edm_crosswalked_assets": "Side-tributary EDM assets whose network location came from a unique exact cross-year publisher-field match rather than a grid reference in that annual return.",
        "side_tributary_edm_location_provenance": "Whether side-tributary EDM attribution uses only publisher-reported coordinates or also exact cross-year location matches.",
        "side_tributary_counted_spills": "Sum of publisher-reported annual spill events on bounded upstream reaches of tributaries entering between the matched sites.",
        "side_tributary_spill_duration_hours": "Annual reported spill asset-hours on bounded upstream reaches of tributaries entering between the matched sites; separate from the exact selected path.",
        "side_tributary_permitted_sewage_outlets": "Distinct permitted sewage outlets on bounded upstream reaches of tributaries entering between the matched sites.",
        "side_tributary_treatment_works": "Permitted treatment works on bounded upstream reaches of tributaries entering between the matched sites.",
        "match_use_recommended": "Whether the selected candidate passes all declared analytical-use gates.",
        "path_edm_assets": "Number of distinct EDM assets mapped on the selected directed route; missing when the national annual return is unavailable.",
        "path_edm_crosswalked_assets": "Selected-route EDM assets whose network location came from a unique exact cross-year publisher-field match rather than a grid reference in that annual return.",
        "path_edm_location_provenance": "Whether selected-route EDM attribution uses only publisher-reported coordinates or also exact cross-year location matches.",
        "path_counted_spills": "Sum of annual publisher-reported counted spill events across routed EDM assets.",
        "path_spill_duration_hours": "Sum of annual reported spill hours across routed assets (asset-hours, not unique elapsed exposure time).",
        "ecological_status": "Latest acquired WFD ecological classification for the linked water body; static context, not survey-year status.",
        "ecological_status_year": "Classification year associated with ecological_status.",
        "relative_nutrient_pressure": "Within-cohort 0-1 percentile composite for supplied nutrient chemistry; not a legal threshold. See pressure_reference_cohort.",
        "relative_organic_oxygen_stress": "Within-cohort 0-1 composite of organic-load and dissolved-oxygen evidence. See pressure_reference_cohort.",
        "relative_metals_pressure": "Highest within-cohort percentile among measured dissolved/total metal fields retained by the method.",
        "relative_edm_pressure": "Within-cohort percentile of log(1 + routed annual asset-hours) where EDM duration is available.",
        "pressure_reference_cohort": "Named provenance cohort within which relative pressure indicators were ranked.",
    }
    if column in replacements:
        return replacements[column]
    determinand = determinand_for_feature(column)
    if column.endswith("_observation_count_6mo"):
        label = f" for {determinand.label}" if determinand else ""
        return f"Number of observations{label} in the inclusive six-calendar-month window ending on the survey date."
    if column.endswith("_median_6mo"):
        label = f" of {determinand.label}" if determinand else ""
        return f"Median{label} in the inclusive six-calendar-month window; censored values use the declared LOQ treatment."
    if column.endswith("_mean_6mo"):
        label = f" of {determinand.label}" if determinand else ""
        return f"Arithmetic mean{label} in the inclusive six-calendar-month window; censored values use the declared LOQ treatment."
    if column.startswith("biology_metric_"):
        metric = column.removeprefix("biology_metric_").replace("_", " ").upper()
        return f"Publisher-supplied {metric} biological metric; interpret only for its applicable assemblage and survey method."
    if column.startswith("evidence_"):
        return "Visible component of the evidence-support calculation; this measures data support, not ecological condition."
    if column.startswith("nrfa_flow_") and column.endswith("_m3s"):
        statistic = column.removeprefix("nrfa_flow_").removesuffix("_6mo_m3s").replace("_", " ")
        return f"{statistic.title()} daily discharge at the selected NRFA gauge in the survey's six-month antecedent window."
    if column.startswith("flow_"):
        return "NRFA-derived hydrological-regime diagnostic using the fixed 1974-2019 reference and declared survey window."
    if column.startswith("score_"):
        return "Named component of the transparent candidate-matching score."
    if column.endswith("_status_year"):
        return "Publisher classification year for the corresponding latest WFD status field."
    return "Released field from the named data group; interpretation, applicability and temporal scope are recorded in the adjacent dictionary columns."


def _unit(column: str) -> str:
    if column.endswith(("_easting", "_northing")) or column in {"easting", "northing"}:
        return "metres (British National Grid)"
    if column.endswith("_distance_m"):
        return "metres"
    if column.endswith("_m3s"):
        return "m³/s"
    if column.endswith("_hours"):
        return "asset-hours"
    if column.endswith("_fraction") or "_fraction_" in column:
        return "proportion (0-1)"
    determinand = determinand_for_feature(column)
    if determinand is not None:
        if column.endswith((
            "_observation_count_6mo", "_below_loq_count_6mo", "_above_range_count_6mo",
        )):
            return "count"
        if column.endswith(("_below_loq_fraction_6mo", "_above_range_fraction_6mo")):
            return "proportion"
        return determinand.standard_unit
    if "_mg_l_" in column or re.match(
        r"^(arsenic|cadmium|chromium|copper|lead|mercury|nickel|zinc)(?:_total)?_",
        column,
    ):
        return "mg/L"
    if column.endswith("_percent") or column == "evidence_strength_percent":
        return "percent"
    if column.startswith("relative_"):
        return "0-1 relative rank"
    if column.endswith(("_observation_count_6mo", "_count", "_assets")):
        return "count"
    if column.endswith(("_median_6mo", "_mean_6mo")):
        return "standardised determinand unit; see determinand name"
    return "not applicable or publisher-defined"


def _temporal_scope(column: str) -> str:
    if column.endswith("_6mo") or "window" in column or column.startswith("nrfa_flow_"):
        return "six calendar months anchored to the biological survey"
    if column.startswith("path_edm") or column in {"path_counted_spills", "path_spill_duration_hours"}:
        return "EDM reporting year matching the biological year"
    if column.startswith("side_tributary_") and column in {
        "side_tributary_edm_assets", "side_tributary_counted_spills",
        "side_tributary_spill_duration_hours", "side_tributary_attribution_method",
        "side_tributary_edm_crosswalked_assets",
        "side_tributary_edm_location_provenance",
    }:
        return "EDM reporting year matching the biological year"
    if column.startswith("side_tributary_"):
        return "static discharge extract or route-method metadata for the output"
    if column in {"chemical_status", "ecological_status", "overall_status"} or column.endswith("_status_year"):
        return "latest acquired WFD classification and its recorded vintage"
    if column.startswith(("flow_reference_", "flow_iha_reference")):
        return "fixed 1974-2019 reference plus survey-anchored window"
    return "biological survey row or source-static context"


def _applicability(column: str) -> str:
    if column == "tributary_penalty_fraction_used":
        return "automated matched rows where a calibrated-flow or raw-Shreve fraction is available"
    if column == "tributary_penalty_evidence_available":
        return "automated rows with a selected directed-network candidate"
    if column in {
        "transport_evidence_fraction",
        "score_transport_plausibility",
        "score_transport_plausibility_method",
    }:
        return "automated rows with a selected directed-network candidate and transport evidence"
    if column in {
        "tributary_wq_link_resolution", "tributary_bio_link_resolution",
        "tributary_parallel_alias_resolved", "flow_proxy_exponent",
        "major_tributary_flow_fraction_threshold",
    }:
        return "automated rows with a selected directed-network candidate"
    if column.startswith((
        "tributary_", "intervening_", "represented_shreve_",
        "unrepresented_shreve_", "estimated_flow_", "largest_tributary_",
        "largest_side_tributary_", "major_tributary_",
    )) or column in {
        "structural_major_tributary_count",
        "combined_tributary_influence_major",
        "shreve_monotonic_downstream",
        "strahler_monotonic_downstream",
        "network_order_monotonic",
    }:
        return "automated rows with a selected directed-network candidate"
    if column.startswith("biology_metric_"):
        return "only the biological groups and methods supplying this metric"
    if column.startswith(("wq_", "chemistry_")) or re.search(r"_(observation_count|median|mean)_6mo$", column):
        return "rows with a defensible chemistry match and relevant observations"
    if column.startswith(("nrfa_", "flow_")):
        return "rows with an eligible selected NRFA gauge; newer years may be unavailable"
    if column.startswith("path_edm") or column in {"path_counted_spills", "path_spill_duration_hours"}:
        return "automated rows in a year with a valid national EDM return and routed match"
    if column.startswith("side_tributary_"):
        return "automated rows with a routed match; zero indicates no attributed side-tributary evidence"
    return "all rows unless missing in the originating source"


def _lineage(column: str) -> str:
    group = _group(column)
    return {
        "biology": "publisher biology survey field",
        "six-month chemistry": "selected-station observations aggregated to the survey window",
        "NRFA flow": "quality-controlled NRFA daily flow and fixed reference",
        "wastewater context": "annual EDM/static discharge evidence routed on the selected path",
        "WFD context": "latest acquired WFD classification and RNAG context",
        "evidence quality": "repeatable output calculation",
        "matching and interpretation": "directed network candidate/decision workflow",
        "site and source details": "site IDs and fields used to join the source tables",
    }[group]


# Build the dictionary from the columns actually present, so it stays in step
# when an optional source is added or left out.
def build_data_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    """Describe every column in the public analysis-ready CSV."""
    origin = data.get("data_origin", pd.Series(index=data.index, dtype="object"))
    rows = []
    for column in data.columns:
        values = data[column]
        rows.append({
            "column": column,
            "group": _group(column),
            "data_type": str(values.dtype),
            "definition": _definition(column),
            "unit": _unit(column),
            "temporal_scope": _temporal_scope(column),
            "applicability": _applicability(column),
            "lineage": _lineage(column),
            "non_null_rows": int(values.notna().sum()),
            "coverage_percent": round(float(values.notna().mean() * 100), 2),
            "pipeline_non_null_rows": int(values[origin.eq("connected_river_pipeline")].notna().sum()),
        })
    return pd.DataFrame(rows)
