"""Build one monitoring-summary row per WFD river water body."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _join(values: pd.Series) -> str:
    return "; ".join(sorted(set(values.dropna().astype(str))))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce"
    )


# Collect the main monitoring and regulatory signals at water-body level. This
# is a descriptive summary for exploration, not a combined condition score.
def build_waterbody_evidence_dossier(
    evidence: pd.DataFrame,
    *,
    status: pd.DataFrame,
    pressures: pd.DataFrame,
    protected_areas: pd.DataFrame,
    objectives: pd.DataFrame | None = None,
    rsn_evidence: pd.DataFrame | None = None,
    recent_start_year: int = 2022,
) -> pd.DataFrame:
    """Join official context to site-balanced monitoring evidence.

    Regulatory attributes use one vote per water body; repeat monitoring visits
    use one vote per site, biological group and year. Concentrations remain
    descriptive medians, never regulatory pass/fail classifications.
    """
    base = status.rename(columns={"Water Body ID": "wfd_waterbody_id"}).copy()
    base["wfd_waterbody_id"] = base["wfd_waterbody_id"].astype("string")

    if objectives is not None and not objectives.empty:
        context_columns = [
            column for column in [
                "Water Body ID", "Water Body", "River Basin District",
                "Management Catchment", "Operational Catchment",
            ] if column in objectives
        ]
        context = objectives[context_columns].drop_duplicates("Water Body ID").rename(
            columns={
                "Water Body ID": "wfd_waterbody_id",
                "Water Body": "water_body_name",
                "River Basin District": "river_basin_district",
                "Management Catchment": "management_catchment",
                "Operational Catchment": "operational_catchment",
            }
        )
        context["wfd_waterbody_id"] = context["wfd_waterbody_id"].astype("string")
        base = base.merge(context, on="wfd_waterbody_id", how="outer", validate="one_to_one")

    pressure = pressures.rename(columns={"Water Body ID": "wfd_waterbody_id"}).copy()
    pressure["wfd_waterbody_id"] = pressure["wfd_waterbody_id"].astype("string")
    base = base.merge(pressure, on="wfd_waterbody_id", how="outer", validate="one_to_one")

    if not protected_areas.empty:
        protected = protected_areas.groupby("Water Body ID", as_index=False).agg(
            protected_area_count=("Protected Area Code", "nunique"),
            protected_area_codes=("Protected Area Code", _join),
            protected_area_types=("Protected Area Type", _join),
        ).rename(columns={"Water Body ID": "wfd_waterbody_id"})
        protected["wfd_waterbody_id"] = protected["wfd_waterbody_id"].astype("string")
        base = base.merge(protected, on="wfd_waterbody_id", how="left", validate="one_to_one")
    base["protected_area_count"] = _numeric(base, "protected_area_count").fillna(0).astype("Int64")

    x = evidence[evidence.get("wfd_waterbody_id", pd.Series(pd.NA, index=evidence.index)).notna()].copy()
    if not x.empty:
        x["wfd_waterbody_id"] = x["wfd_waterbody_id"].astype("string")
        fallback = x.get("site_id", pd.Series(pd.NA, index=x.index)).astype("string")
        x["_site_key"] = x.get("bio_site_id", fallback).astype("string").fillna(fallback)
        x["sample_year"] = _numeric(x, "sample_year").astype("Int64")
        survey_counts = x.groupby("wfd_waterbody_id", as_index=False).agg(
            survey_records=("integration_record_id", "size")
        )

        median_columns = [
            "evidence_strength_percent", "relative_nutrient_pressure",
            "relative_organic_oxygen_stress", "relative_metals_pressure",
            "relative_edm_pressure", "metals_measured_count",
            "ammoniacal_n_mg_l_median", "nitrate_n_mg_l_median",
            "nitrite_n_mg_l_median", "ammonia_unionised_n_mg_l_median",
            "orthophosphate_p_mg_l_median", "bod_5day_atu_mg_l_median",
            "dissolved_oxygen_mg_l_median", "dissolved_oxygen_percent_median",
            "turbidity_ntu_median", "path_edm_assets",
            "path_permitted_sewage_outlets", "path_treatment_works",
        ]
        aggregations: dict[str, tuple[str, str]] = {
            "integration_record_id": ("integration_record_id", "first"),
            "data_origin": ("data_origin", "first"),
        }
        for column in median_columns:
            if column in x:
                aggregations[column] = (column, "median")
        for column in ["wq_n_sampling_dates", "flow_evidence_available", "match_use_recommended"]:
            if column in x:
                aggregations[column] = (column, "max")
        site_year = x.groupby(
            ["wfd_waterbody_id", "_site_key", "assemblage", "sample_year"],
            as_index=False, dropna=False,
        ).agg(**aggregations)
        site_year["chemistry_supported"] = _numeric(site_year, "wq_n_sampling_dates").gt(0)
        site_year["flow_supported"] = site_year.get(
            "flow_evidence_available", pd.Series(False, index=site_year.index)
        ).fillna(False).astype(bool)
        site_year["recommended_route"] = site_year.get(
            "match_use_recommended", pd.Series(False, index=site_year.index)
        ).fillna(False).astype(bool)
        site_year["recent"] = site_year["sample_year"].ge(recent_start_year)
        site_year["edm_context_known"] = _numeric(site_year, "path_edm_assets").notna()
        site_year["edm_exposed"] = _numeric(site_year, "path_edm_assets").gt(0)
        site_year["permitted_outlet_context"] = _numeric(
            site_year, "path_permitted_sewage_outlets"
        ).gt(0)
        site_year["treatment_works_context"] = _numeric(
            site_year, "path_treatment_works"
        ).gt(0)

        named: dict[str, tuple[str, object]] = {
            "first_biology_year": ("sample_year", "min"),
            "latest_biology_year": ("sample_year", "max"),
            "site_years": ("integration_record_id", "size"),
            "biology_sites": ("_site_key", "nunique"),
            "biological_groups": ("assemblage", _join),
            "recent_site_years": ("recent", "sum"),
            "chemistry_supported_site_years": ("chemistry_supported", "sum"),
            "flow_supported_site_years": ("flow_supported", "sum"),
            "recommended_route_site_years": ("recommended_route", "sum"),
            "edm_context_site_years": ("edm_context_known", "sum"),
            "edm_exposed_site_years": ("edm_exposed", "sum"),
            "permitted_outlet_context_site_years": ("permitted_outlet_context", "sum"),
            "treatment_works_context_site_years": ("treatment_works_context", "sum"),
        }
        for source, target in [
            ("evidence_strength_percent", "median_evidence_strength"),
            ("relative_nutrient_pressure", "median_relative_nutrient_pressure"),
            ("relative_organic_oxygen_stress", "median_relative_organic_oxygen_stress"),
            ("relative_metals_pressure", "median_relative_metals_pressure"),
            ("relative_edm_pressure", "median_relative_edm_pressure"),
            ("metals_measured_count", "median_metals_measured"),
            ("ammoniacal_n_mg_l_median", "median_ammoniacal_n_mg_l"),
            ("ammonia_unionised_n_mg_l_median", "median_ammonia_unionised_n_mg_l"),
            ("nitrate_n_mg_l_median", "median_nitrate_n_mg_l"),
            ("nitrite_n_mg_l_median", "median_nitrite_n_mg_l"),
            ("orthophosphate_p_mg_l_median", "median_orthophosphate_p_mg_l"),
            ("bod_5day_atu_mg_l_median", "median_bod_5day_atu_mg_l"),
            ("dissolved_oxygen_mg_l_median", "median_dissolved_oxygen_mg_l"),
            ("dissolved_oxygen_percent_median", "median_dissolved_oxygen_percent"),
            ("turbidity_ntu_median", "median_turbidity_ntu"),
        ]:
            if source in site_year:
                named[target] = (source, "median")
                named[target + "_n"] = (source, "count")
        monitoring = site_year.groupby("wfd_waterbody_id", as_index=False).agg(**named)
        monitoring = monitoring.merge(survey_counts, on="wfd_waterbody_id", validate="one_to_one")
        base = base.merge(monitoring, on="wfd_waterbody_id", how="outer", validate="one_to_one")

    if rsn_evidence is not None and not rsn_evidence.empty and "waterbody_id" in rsn_evidence:
        rsn = rsn_evidence[rsn_evidence["waterbody_id"].notna()].copy()
        rsn["waterbody_id"] = rsn["waterbody_id"].astype("string")
        rsn_summary = rsn.groupby("waterbody_id", as_index=False).agg(
            rsn_site_years=("evidence_id", "size"),
            rsn_sites=("network_id", "nunique"),
            latest_rsn_year=("year", "max"),
            rsn_wq_observations=("wq_numeric_observations", "sum"),
            rsn_habitat_surveys=("habitat_surveys", "sum"),
        ).rename(columns={"waterbody_id": "wfd_waterbody_id"})
        base = base.merge(rsn_summary, on="wfd_waterbody_id", how="left", validate="one_to_one")

    base["below_good"] = base.get("ecological_status", pd.Series(pd.NA, index=base.index)).isin(
        ["Moderate", "Poor", "Bad"]
    )
    recent = _numeric(base, "recent_site_years").fillna(0)
    base["recent_biology_available"] = recent.gt(0)
    base["monitoring_gap_since_2022"] = ~base["recent_biology_available"]
    base["below_good_without_recent_biology"] = base["below_good"] & ~base["recent_biology_available"]
    return base.sort_values("wfd_waterbody_id").reset_index(drop=True)
