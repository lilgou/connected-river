"""Build the analysis-ready site-year table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _percentile(series: pd.Series, *, reverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranked = values.rank(pct=True, method="average")
    return 1 - ranked if reverse else ranked


def build_pressure_indicators(
    table: pd.DataFrame,
    *,
    cohort_column: str | None = None,
) -> pd.DataFrame:
    """Add transparent relative pressure dimensions; do not collapse them into one score.

    ``cohort_column`` keeps relative ranks comparable within a declared period or
    provenance group. The released research table retains release-wide ranks by
    default.
    """
    out = table.copy()
    if cohort_column is not None and cohort_column not in out:
        raise KeyError(f"Pressure reference cohort column is missing: {cohort_column}")

    def rank(column: str, *, reverse: bool = False) -> pd.Series:
        if cohort_column is None:
            return _percentile(out[column], reverse=reverse)
        values = pd.to_numeric(out[column], errors="coerce")
        ranked = values.groupby(out[cohort_column], dropna=False).rank(
            pct=True, method="average"
        )
        return 1 - ranked if reverse else ranked

    nutrient = [c for c in ["ammoniacal_n_mg_l_median", "nitrate_n_mg_l_median", "orthophosphate_p_mg_l_median"] if c in out]
    oxygen = [c for c in ["bod_5day_atu_mg_l_median", "dissolved_oxygen_mg_l_median"] if c in out]
    metal_candidates = [
        ("arsenic_median", "Arsenic"),
        ("cadmium_median", "Cadmium"),
        ("chromium_median", "Chromium"),
        ("copper_median", "Copper"),
        ("lead_median", "Lead"),
        ("mercury_total_median", "Mercury"),
        ("mercury_median", "Mercury"),
        ("nickel_median", "Nickel"),
        ("zinc_median", "Zinc"),
    ]
    # Accept both mercury aliases and count mercury once if both are present.
    metals: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for column, label in metal_candidates:
        if column in out and label not in seen_labels:
            metals.append((column, label))
            seen_labels.add(label)
    if nutrient:
        out["relative_nutrient_pressure"] = pd.concat(
            [rank(c) for c in nutrient], axis=1
        ).mean(axis=1)
    if oxygen:
        components = [rank("bod_5day_atu_mg_l_median")] if "bod_5day_atu_mg_l_median" in out else []
        if "dissolved_oxygen_mg_l_median" in out:
            components.append(rank("dissolved_oxygen_mg_l_median", reverse=True))
        out["relative_organic_oxygen_stress"] = pd.concat(components, axis=1).mean(axis=1)
    if metals:
        metal_ranks = pd.concat(
            {label: rank(column) for column, label in metals}, axis=1
        )
        metal_values = pd.concat(
            {
                label: pd.to_numeric(out[column], errors="coerce")
                for column, label in metals
            },
            axis=1,
        )
        out["metals_measured_count"] = metal_values.notna().sum(axis=1).astype("Int64")
        has_metal_rank = metal_ranks.notna().any(axis=1)
        dominant_metal = pd.Series(pd.NA, index=out.index, dtype="string")
        dominant_metal.loc[has_metal_rank] = metal_ranks.loc[has_metal_rank].idxmax(axis=1)
        out["dominant_metal"] = dominant_metal
        out["relative_metals_pressure"] = metal_ranks.max(axis=1)
    if "path_spill_duration_hours" in out:
        out["_edm_log_duration"] = np.log1p(
            pd.to_numeric(out["path_spill_duration_hours"], errors="coerce")
        )
        out["relative_edm_pressure"] = rank("_edm_log_duration")
        out = out.drop(columns="_edm_log_duration")
    out["pressure_reference_cohort"] = (
        out[cohort_column].astype("string")
        if cohort_column is not None
        else "whole_release"
    )
    if "flow_relative_to_q95" in out:
        ratio = pd.to_numeric(out["flow_relative_to_q95"], errors="coerce")
        out["survey_flow_context"] = pd.cut(
            ratio, bins=[-np.inf, 1, 2, 10, np.inf], labels=["at_or_below_q95", "very_low", "within_typical_range", "high_relative_flow"],
        ).astype("string")
    # Evidence strength is a fixed-denominator support score, not a river-health
    # score. Keeping the four components beside it shows where the score came
    # from and whether any supporting evidence was missing.
    match_support = pd.Series(False, index=out.index, dtype="bool")
    if "match_use_recommended" in out:
        recommended = out["match_use_recommended"]
        if pd.api.types.is_string_dtype(recommended) or recommended.dtype == object:
            recommended = recommended.astype("string").str.lower().map({"true": True, "false": False})
        match_support = recommended.fillna(False).astype(bool)

    sampling_frequency = pd.to_numeric(
        out.get("wq_observation_count_max_6mo", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    sampling_frequency = sampling_frequency.combine_first(pd.to_numeric(
        out.get("wq_n_sampling_dates", pd.Series(np.nan, index=out.index)), errors="coerce",
    )).fillna(0)
    determinand_breadth = pd.to_numeric(
        out.get("chemistry_determinand_count_6mo", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    determinand_breadth = determinand_breadth.combine_first(pd.to_numeric(
        out.get("wq_n_target_determinands", pd.Series(np.nan, index=out.index)), errors="coerce",
    )).fillna(0)
    if "flow_evidence_available" in out:
        flow_available = out["flow_evidence_available"].fillna(False).astype(bool)
    elif "nrfa_flow_mean_6mo_m3s" in out:
        flow_available = out["nrfa_flow_mean_6mo_m3s"].notna()
    elif "flow_mean_30d_m3s" in out:
        flow_available = out["flow_mean_30d_m3s"].notna()
    else:
        flow_available = pd.Series(False, index=out.index, dtype="bool")
    out["evidence_match_support"] = match_support.astype(float)
    out["evidence_chemistry_frequency_support"] = (sampling_frequency / 12).clip(lower=0, upper=1)
    # Keep the older column name as an alias; the frequency name is preferred.
    out["evidence_chemistry_dates_support"] = out["evidence_chemistry_frequency_support"]
    out["evidence_chemistry_breadth_support"] = (determinand_breadth / 9).clip(lower=0, upper=1)
    out["evidence_nrfa_flow_support"] = flow_available.astype(float)
    evidence_columns = [
        "evidence_match_support", "evidence_chemistry_frequency_support",
        "evidence_chemistry_breadth_support", "evidence_nrfa_flow_support",
    ]
    out["evidence_strength_percent"] = 100 * out[evidence_columns].mean(axis=1)
    reference_text = (
        f"within each {cohort_column} cohort"
        if cohort_column is not None
        else "within this release"
    )
    out["indicator_interpretation"] = (
        f"Pressure indicators are empirical relative positions {reference_text}; evidence strength is a "
        "0-100 data-support score. Neither is a regulatory threshold or causal river-health score."
    )
    return out


def build_analysis_table(
    integrated: pd.DataFrame, match_decisions: pd.DataFrame, flow_features: pd.DataFrame,
    discharge_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    decision_columns = [c for c in match_decisions if c not in integrated or c in {"bio_site_id", "year"}]
    out = integrated.merge(match_decisions[decision_columns], on=["bio_site_id", "year"], how="left", validate="one_to_one")
    if "match_decision" in out:
        out["match_decision"] = out["match_decision"].fillna("no_candidate_match")
        out["match_use_recommended"] = out["match_use_recommended"].fillna(False)
    out = out.merge(flow_features, on=["bio_site_id", "year"], how="left", validate="one_to_one", suffixes=("", "_flow"))
    if discharge_context is not None:
        discharge_context = discharge_context.copy()
        cutoff = "side_tributary_upstream_cutoff_m"
        static_cutoff = f"{cutoff}_static"
        if cutoff in out and cutoff in discharge_context:
            discharge_context = discharge_context.rename(columns={cutoff: static_cutoff})
        out = out.merge(
            discharge_context, on=["bio_site_id", "year", "wq_site_id"], how="left", validate="one_to_one",
        )
        if static_cutoff in out:
            routed = pd.to_numeric(out[cutoff], errors="coerce")
            static = pd.to_numeric(out[static_cutoff], errors="coerce")
            inconsistent = routed.notna() & static.notna() & ~np.isclose(routed, static)
            if inconsistent.any():
                raise ValueError("EDM and static-discharge side-tributary cutoffs disagree")
            out[cutoff] = routed.combine_first(static)
            out = out.drop(columns=static_cutoff)
    return build_pressure_indicators(out)
