"""Exhaustive automated-product chemistry-coverage accounting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUTCOMES = (
    "released_chemistry_match",
    "no_chemistry_candidate",
    "candidates_present_but_all_rejected",
    "insufficient_temporal_coverage",
    "network_attachment_failure",
    "missing_or_ambiguous_route",
    "quality_exclusion",
    "other_documented_reason",
)


def _bool(values: pd.Series) -> pd.Series:
    return values.astype("boolean").fillna(False)


def _distance_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-float("inf"), 50, 100, 250, 500, 1000, float("inf")],
        labels=["0-50m", "50-100m", "100-250m", "250-500m", "500-1000m", ">1000m"],
    ).astype("string").fillna("unavailable")


def _monitoring_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-float("inf"), 0, 1, 2, 5, 12, float("inf")],
        labels=["0", "1", "2", "3-5", "6-12", ">12"],
        include_lowest=True,
    ).astype("string").fillna("no_candidate")


# Give every automated survey exactly one chemistry outcome. This makes gaps
# countable instead of leaving missing rows open to interpretation.
def build_chemistry_coverage_ledger(
    specialist: pd.DataFrame,
    *,
    biology_snaps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign exactly one documented chemistry outcome to every automated row."""
    required = {
        "integration_record_id", "data_origin", "bio_site_id", "sample_date",
        "sample_year", "assemblage", "chemistry_release_status", "match_decision",
    }
    missing = sorted(required - set(specialist))
    if missing:
        raise ValueError("Chemistry coverage table requires columns: " + ", ".join(missing))

    automated = specialist.loc[specialist["data_origin"].eq("connected_river_pipeline")].copy()
    if automated["integration_record_id"].duplicated().any():
        raise ValueError("Automated specialist rows do not have unique integration_record_id values")

    snap_available = pd.Series(False, index=automated.index, dtype="boolean")
    if biology_snaps is not None:
        snap_required = {"bio_site_id", "snapped_link_id"}
        if not snap_required.issubset(biology_snaps):
            raise ValueError("Biology snaps require bio_site_id and snapped_link_id")
        snap = (
            biology_snaps[["bio_site_id", "snapped_link_id"]]
            .assign(network_attached=lambda frame: frame["snapped_link_id"].notna())
            .groupby("bio_site_id", as_index=False)["network_attached"].max()
            .set_index("bio_site_id")["network_attached"]
        )
        snap_available = automated["bio_site_id"].map(snap).fillna(False).astype("boolean")
    elif "bio_snap_distance_m" in automated:
        snap_available = automated["bio_snap_distance_m"].notna().astype("boolean")

    release_status = automated["chemistry_release_status"].astype("string")
    decision = automated["match_decision"].astype("string")
    failures = automated.get(
        "match_rule_failures", pd.Series(pd.NA, index=automated.index, dtype="string")
    ).astype("string").fillna("")
    included = _bool(automated.get(
        "chemistry_included_in_release",
        release_status.eq("included"),
    ))

    outcome = pd.Series("other_documented_reason", index=automated.index, dtype="string")
    reason = pd.Series("unrecognised release state", index=automated.index, dtype="string")

    no_candidate = release_status.eq("no_candidate_match") | decision.eq("no_candidate_match")
    attachment_failure = no_candidate & ~snap_available
    outcome.loc[attachment_failure] = "network_attachment_failure"
    reason.loc[attachment_failure] = "biological site has no valid IRN link attachment"
    nearby_absent = no_candidate & snap_available
    outcome.loc[nearby_absent] = "no_chemistry_candidate"
    reason.loc[nearby_absent] = "site attached to IRN but no directed candidate passed candidate generation"

    insufficient = release_status.eq("no_observations_in_six_month_window")
    outcome.loc[insufficient] = "insufficient_temporal_coverage"
    reason.loc[insufficient] = "selected station has no exact-code observation in the antecedent window"

    rejected = release_status.eq("excluded_no_defensible_match") | decision.eq(
        "no_defensible_match"
    )
    only_temporal = rejected & failures.str.fullmatch(
        r"insufficient_sampling_dates(?:; insufficient_target_determinand_breadth)?",
        na=False,
    )
    outcome.loc[only_temporal] = "insufficient_temporal_coverage"
    reason.loc[only_temporal] = "candidates existed but antecedent monitoring support was inadequate"
    rejected_other = rejected & ~only_temporal
    outcome.loc[rejected_other] = "candidates_present_but_all_rejected"
    reason.loc[rejected_other] = failures.loc[rejected_other].where(
        failures.loc[rejected_other].ne(""), "all candidates failed declared acceptance gates"
    )

    route_ambiguous = decision.isin({
        "provisional_scientific_review",
        "provisional_major_tributary",
        "provisional_intervening_tributary",
        "provisional_intervening_discharge",
        "provisional_intervening_abstraction",
        "provisional_distance_sensitive",
        "provisional_tributary_influence_unavailable",
        "provisional_tributary_influence_unknown",
    }) | release_status.eq("excluded_pending_scientific_review")
    outcome.loc[route_ambiguous] = "missing_or_ambiguous_route"
    review_reason = automated.get(
        "match_review_reasons",
        pd.Series(pd.NA, index=automated.index, dtype="string"),
    ).astype("string")
    reason.loc[route_ambiguous] = review_reason.loc[route_ambiguous].fillna(
        decision.loc[route_ambiguous]
    )

    released = included & ~route_ambiguous
    outcome.loc[released] = "released_chemistry_match"
    reason.loc[released] = "accepted candidate with at least one exact-code antecedent observation"

    quality = release_status.eq("quality_exclusion")
    outcome.loc[quality] = "quality_exclusion"
    reason.loc[quality] = "chemistry observations failed declared quality checks"

    if not set(outcome.unique()).issubset(OUTCOMES):
        raise AssertionError("Coverage outcome is outside the closed vocabulary")

    exact_count_columns = [
        column for column in automated
        if column.startswith("target_exact_") and column.endswith("_observation_count_6mo")
    ]
    observed_determinand_count = (
        automated[exact_count_columns].apply(pd.to_numeric, errors="coerce").gt(0).sum(axis=1)
        if exact_count_columns else pd.Series(0, index=automated.index)
    )
    ledger = automated[[
        "integration_record_id", "bio_site_id", "sample_date", "sample_year", "assemblage"
    ]].copy()
    for column in ("site_id", "site_label", "catchment", "water_body"):
        ledger[column] = automated.get(column, pd.NA)
    ledger["network_attached"] = snap_available
    ledger["chemistry_coverage_outcome"] = outcome
    ledger["chemistry_coverage_reason"] = reason
    ledger["chemistry_included_in_release"] = included
    for column in (
        "match_decision", "match_rule_failures", "candidate_count", "network_distance_m",
        "bio_snap_distance_m", "wq_snap_distance_m", "wq_n_sampling_dates",
        "wq_n_months_sampled", "wq_n_target_determinands", "selection_confidence",
        "tributary_influence_class",
    ):
        ledger[column] = automated.get(column, pd.NA)
    ledger["attachment_distance_band"] = _distance_band(ledger["bio_snap_distance_m"])
    ledger["network_distance_band"] = _distance_band(ledger["network_distance_m"])
    ledger["monitoring_density_band"] = _monitoring_band(ledger["wq_n_sampling_dates"])
    ledger["observed_exact_determinand_count"] = observed_determinand_count.astype("Int64")
    ledger["coverage_classification_version"] = "1.0"
    ledger = ledger.reset_index(drop=True)
    if len(ledger) != len(automated) or ledger["chemistry_coverage_outcome"].isna().any():
        raise AssertionError("Coverage accounting is not exhaustive")
    return ledger


def build_coverage_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return tidy outcome counts across the requested explanatory dimensions."""
    dimensions = {
        "geography_catchment": "catchment",
        "geography_water_body": "water_body",
        "biological_programme": "assemblage",
        "survey_year": "sample_year",
        "attachment_distance": "attachment_distance_band",
        "network_distance": "network_distance_band",
        "network_gaps": "match_rule_failures",
        "monitoring_density": "monitoring_density_band",
    }
    parts = []
    for label, column in dimensions.items():
        values = ledger[column].astype("string").fillna("unavailable")
        grouped = (
            ledger.assign(dimension_value=values)
            .groupby(["dimension_value", "chemistry_coverage_outcome"], dropna=False)
            .size().rename("survey_rows").reset_index()
        )
        grouped.insert(0, "dimension", label)
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


# Write row-level outcomes and a smaller summary together so headline
# percentages can always be traced back to individual surveys.
def write_chemistry_coverage_outputs(
    specialist_path: Path,
    output_dir: Path,
    *,
    biology_snaps_path: Path | None = None,
) -> dict[str, Path]:
    specialist = pd.read_parquet(specialist_path)
    snaps = pd.read_parquet(biology_snaps_path) if biology_snaps_path else None
    ledger = build_chemistry_coverage_ledger(specialist, biology_snaps=snaps)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "chemistry_coverage_ledger.parquet"
    summary_path = output_dir / "chemistry_coverage_summary.csv"
    report_path = output_dir / "2018-onward-chemistry-coverage.md"
    ledger.to_parquet(ledger_path, index=False)
    build_coverage_summary(ledger).to_csv(summary_path, index=False)
    counts = ledger["chemistry_coverage_outcome"].value_counts().reindex(OUTCOMES, fill_value=0)
    total = len(ledger)
    lines = [
        "# Chemistry coverage for the 2018-onward automated product",
        "",
        f"Every one of the `{total:,}` automated survey rows has exactly one outcome.",
        "",
        "| Outcome | Survey rows | Percent |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{outcome}` | {int(count):,} | {100 * int(count) / total:.1f}% |"
        for outcome, count in counts.items()
    )
    lines.extend([
        "",
        "This table does not relax the matching checks. `missing_or_ambiguous_route` includes released "
        "provisional matches whose tributary evidence is unavailable or unknown; use those rows only "
        "with that uncertainty visible. `insufficient_temporal_coverage` includes both a selected "
        "station with no observation in the six-calendar-month antecedent window and candidates "
        "rejected solely for inadequate monitoring support.",
        "",
        "The tidy summary reports outcomes by geography, assemblage, year, attachment/network "
        "distance, rule failures and monitoring density. Determinand-level observation counts remain "
        "available in the specialist product and its 41-code acquisition/coverage ledgers.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ledger": ledger_path, "summary": summary_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specialist", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--biology-snaps", type=Path)
    args = parser.parse_args()
    print(write_chemistry_coverage_outputs(
        args.specialist, args.output_dir, biology_snaps_path=args.biology_snaps
    ))


if __name__ == "__main__":
    main()
