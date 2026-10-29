"""Apply user-defined ranking recipes to retained matching candidates.

These helpers leave the default match unchanged. They create named
alternative selections from the candidate table and can produce a site-year
dataset when the corresponding chemistry features are available.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

MATCH_KEYS = ["bio_site_id", "year"]

# These are the maximum points available in the standard 100-point score. A
# custom recipe rescales each component from its original maximum to the number
# of points chosen below. For example, a distance score of 17.5 out of 35 earns
# half of the custom distance points.
BASE_COMPONENT_MAXIMA = {
    "score_distance": 35.0,
    "score_snap": 15.0,
    "score_temporal": 20.0,
    "score_chemistry": 20.0,
    "score_transport_plausibility": 10.0,
}
RECIPE_TOTAL_POINTS = 100.0  # Every recipe must add up to exactly 100 points.

# This example gives chemistry more weight and distance a little less. It does
# not replace the standard match; it creates a named alternative that can be
# compared with it.
EXAMPLE_CHEMISTRY_EMPHASIS_RECIPE = {
    # A short label added to the output columns. Use lowercase characters only.
    "name": "chemistry_emphasis",
    # Optional year limits are useful when a source is not available throughout
    # the full study period. Leave either limit out if it is not needed.
    "year_min": 2018,
    # Reassign the five standard score components. These values, plus any
    # evidence terms below, must total 100.
    "base_component_points": {
        "score_distance": 25,
        "score_snap": 15,
        "score_temporal": 20,
        "score_chemistry": 30,
        "score_transport_plausibility": 10,
    },
    # Extra candidate columns can be added to the score here. The empty list
    # means this example uses only the five standard components. An extra term
    # would look like this:
    # {
    #     "column": "example_evidence_column",
    #     "transform": "linear_cap",
    #     "cap": 10,
    #     "points": 5,
    #     "missing": "exclude",
    # }
    "evidence_terms": [],
    # Review candidates remain excluded unless every one of their review reasons
    # appears here. An empty list is the cautious option: accept rows only.
    "allowed_review_reasons": [],
}

# Evidence columns need to be turned into a value between 0 and 1 before points
# are assigned. These are the four supported ways to do that:
# - binary:    zero = 0, anything above zero = 1
# - identity:  use a value already expressed between 0 and 1
# - linear_cap: divide by a chosen cap, with anything above the cap set to 1
# - log1p_cap: as above, but reduce the influence of very large values
TRANSFORMS = {"binary", "identity", "linear_cap", "log1p_cap"}

# If an extra evidence value is missing, "exclude" removes that candidate from
# this custom recipe. "zero" keeps it but awards zero points for that term.
MISSING_POLICIES = {"exclude", "zero"}


def _validated_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Check a recipe and return a plain dictionary ready for scoring."""

    out = dict(recipe)

    # Restrict names to characters that are safe to use in filenames and column
    # labels later in the workflow.
    name = str(out.get("name", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Recipe name must use lowercase letters, numbers, underscores or hyphens")

    base = out.get("base_component_points")
    if not isinstance(base, Mapping) or not base:
        raise ValueError("Recipe must declare base_component_points")
    unknown = sorted(set(base) - set(BASE_COMPONENT_MAXIMA))
    if unknown:
        raise ValueError("Unknown base score components: " + ", ".join(unknown))
    for column, points in base.items():
        value = float(points)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Base component {column!r} must have finite non-negative points")

    # Extra evidence is optional, but each term needs one unique source column,
    # a supported scaling method and a clear points value.
    terms = out.get("evidence_terms", [])
    if not isinstance(terms, list):
        raise ValueError("evidence_terms must be a list")
    seen_columns: set[str] = set()
    for term in terms:
        if not isinstance(term, Mapping):
            raise ValueError("Each evidence term must be a mapping")
        column = str(term.get("column", ""))
        if not column or column in seen_columns:
            raise ValueError("Evidence-term columns must be present and unique")
        seen_columns.add(column)
        transform = str(term.get("transform", ""))
        if transform not in TRANSFORMS:
            raise ValueError(f"Unsupported transform {transform!r} for {column!r}")
        missing = str(term.get("missing", "exclude"))
        if missing not in MISSING_POLICIES:
            raise ValueError(f"Unsupported missing policy {missing!r} for {column!r}")
        points = float(term.get("points", float("nan")))
        if not math.isfinite(points) or points < 0:
            raise ValueError(
                f"Evidence term {column!r} must have finite non-negative points"
            )
        if transform in {"linear_cap", "log1p_cap"}:
            cap = float(term.get("cap", float("nan")))
            if not math.isfinite(cap) or cap <= 0:
                raise ValueError(f"Evidence term {column!r} requires a positive finite cap")

    # Review reasons are compared as exact strings. This makes any relaxation of
    # the standard evidence checks deliberate and easy to see in the recipe.
    allowed = out.get("allowed_review_reasons", [])
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError("allowed_review_reasons must be a list of exact reason strings")
    for bound in ("year_min", "year_max"):
        if out.get(bound) is not None:
            out[bound] = int(out[bound])
    if (
        out.get("year_min") is not None
        and out.get("year_max") is not None
        and out["year_min"] > out["year_max"]
    ):
        raise ValueError("year_min cannot exceed year_max")
    # Keeping every recipe on a 100-point scale makes comparisons much easier to
    # explain, even when the balance of components changes.
    total_points = sum(float(points) for points in base.values()) + sum(
        float(term["points"]) for term in terms
    )
    if not math.isclose(total_points, RECIPE_TOTAL_POINTS, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Recipe points must total exactly {RECIPE_TOTAL_POINTS:g}; "
            f"received {total_points:g}"
        )
    out["total_points"] = total_points
    return out


def _scale(values: pd.Series, *, transform: str, cap: float | None) -> pd.Series:
    """Convert one evidence column to the common zero-to-one scale."""

    numeric = pd.to_numeric(values, errors="coerce")
    if transform == "binary":
        return numeric.gt(0).astype("float64").where(numeric.notna())
    if transform == "identity":
        return numeric.clip(0.0, 1.0)
    if transform == "linear_cap":
        return (numeric / float(cap)).clip(0.0, 1.0)
    if transform == "log1p_cap":
        return (np.log1p(numeric.clip(lower=0)) / np.log1p(float(cap))).clip(0.0, 1.0)
    raise AssertionError(f"Unreachable transform: {transform}")


def _reason_set(value: object) -> set[str]:
    """Split the semicolon-separated review reasons stored on a candidate."""

    if pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(";") if item.strip()}


def apply_custom_matching_recipe(
    candidates: pd.DataFrame,
    recipe: Mapping[str, Any],
) -> pd.DataFrame:
    """Score and rank candidates under a declared alternative research recipe.

    Rejected default candidates are never eligible. Review candidates become
    eligible only when every recorded review reason is explicitly allow-listed.
    This lets a user change the research estimand without silently bypassing
    unrelated tributary, topology, temporal, chemistry or snap-quality gates.
    """

    # Validate first so a partly specified recipe cannot produce a plausible-
    # looking result with the wrong weights.
    config = _validated_recipe(recipe)
    required = {
        *MATCH_KEYS,
        "wq_site_id",
        "network_distance_m",
        "candidate_gate_decision",
        "candidate_review_reasons",
        *config["base_component_points"],
        *(str(term["column"]) for term in config.get("evidence_terms", [])),
    }
    missing_columns = sorted(required - set(candidates.columns))
    if missing_columns:
        raise KeyError("Candidate table is missing recipe fields: " + ", ".join(missing_columns))

    out = candidates.copy()
    contribution_columns: list[str] = []
    missing_evidence = pd.Series(False, index=out.index)

    # Rescale each standard component to its new points allowance. The separate
    # contribution columns make it possible to see exactly where a score came
    # from instead of returning only a final total.
    for column, points in config["base_component_points"].items():
        values = pd.to_numeric(out[column], errors="coerce")
        contribution = (
            values.clip(0.0, BASE_COMPONENT_MAXIMA[column])
            / BASE_COMPONENT_MAXIMA[column]
            * float(points)
        )
        contribution_column = f"custom_component_{column.removeprefix('score_')}"
        out[contribution_column] = contribution.fillna(0.0)
        contribution_columns.append(contribution_column)
        missing_evidence |= values.isna()

    # Add any study-specific evidence terms using the same zero-to-one approach.
    for term in config.get("evidence_terms", []):
        column = str(term["column"])
        scaled = _scale(
            out[column],
            transform=str(term["transform"]),
            cap=float(term["cap"]) if term.get("cap") is not None else None,
        )
        contribution_column = f"custom_component_{column}"
        out[contribution_column] = scaled.fillna(0.0) * float(term["points"])
        contribution_columns.append(contribution_column)
        if str(term.get("missing", "exclude")) == "exclude":
            missing_evidence |= scaled.isna()

    out["custom_score"] = out[contribution_columns].sum(axis=1)

    # A custom score is not a way around the scientific checks. Standard accepts
    # stay eligible; review rows are allowed only when their reasons were named
    # in the recipe; standard rejects always remain excluded.
    gate = out["candidate_gate_decision"].astype("string")
    allowed_review_reasons = set(config.get("allowed_review_reasons", []))
    reason_sets = out["candidate_review_reasons"].map(_reason_set)
    review_allowed = gate.eq("review") & reason_sets.map(
        lambda reasons: bool(reasons) and reasons.issubset(allowed_review_reasons)
    )
    eligible = gate.eq("accept") | review_allowed

    ineligible_reason = pd.Series("", index=out.index, dtype="string")
    ineligible_reason.loc[gate.eq("reject")] = "default_reject_gate"
    disallowed_review = gate.eq("review") & ~review_allowed
    ineligible_reason.loc[disallowed_review] = "review_reason_not_allow_listed"
    unknown_gate = ~gate.isin(["accept", "review", "reject"])
    ineligible_reason.loc[unknown_gate] = "unknown_default_gate"
    eligible &= ~unknown_gate

    # Apply the optional study-period limits after the evidence checks.
    if config.get("year_min") is not None:
        before = pd.to_numeric(out["year"], errors="coerce").lt(config["year_min"])
        eligible &= ~before
        ineligible_reason.loc[before] = "before_recipe_year_range"
    if config.get("year_max") is not None:
        after = pd.to_numeric(out["year"], errors="coerce").gt(config["year_max"])
        eligible &= ~after
        ineligible_reason.loc[after] = "after_recipe_year_range"
    eligible &= ~missing_evidence
    ineligible_reason.loc[missing_evidence] = "required_recipe_evidence_missing"

    # Keep the eligibility decision and reason beside the score. This is useful
    # when comparing why two recipes selected different stations.
    out["custom_recipe_name"] = config["name"]
    out["custom_eligible"] = eligible
    out["custom_ineligible_reason"] = ineligible_reason.where(~eligible, "")
    out["custom_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    # Highest score wins within each biological site-year. Distance and station
    # ID provide consistent tie-breaks when two candidates have the same score.
    ranked = out.loc[eligible].sort_values(
        [*MATCH_KEYS, "custom_score", "network_distance_m", "wq_site_id"],
        ascending=[True, True, False, True, True],
    )
    ranks = ranked.groupby(MATCH_KEYS).cumcount() + 1
    out.loc[ranked.index, "custom_rank"] = ranks.astype("Int64")
    out["custom_selected"] = out["custom_rank"].eq(1).fillna(False)
    return out


def select_custom_matches(scored_candidates: pd.DataFrame) -> pd.DataFrame:
    """Return one selected candidate per biology-site year."""

    required = {*MATCH_KEYS, "custom_selected"}
    missing = sorted(required - set(scored_candidates.columns))
    if missing:
        raise KeyError("Scored candidate table is missing: " + ", ".join(missing))
    # The scoring function marks rank one, so this helper returns those
    # rows and checks that there is no accidental duplication.
    selected = scored_candidates.loc[scored_candidates["custom_selected"]].copy()
    if selected.duplicated(MATCH_KEYS).any():
        raise ValueError("Custom selection produced duplicate biology-site years")
    return selected.sort_values(MATCH_KEYS).reset_index(drop=True)


def materialize_custom_site_year_dataset(
    biology_site_year: pd.DataFrame,
    selected_matches: pd.DataFrame,
    wq_site_year_features: pd.DataFrame,
    *,
    recipe_name: str,
) -> pd.DataFrame:
    """Join a custom station selection to biology and station-year chemistry."""

    # Start with all biological site-years so unmatched rows are retained rather
    # than disappearing from the finished table.
    biology = biology_site_year.copy()
    if "bio_site_id" not in biology:
        required = {"assemblage", "site_id"}
        if not required.issubset(biology.columns):
            raise KeyError("Biology table needs bio_site_id or assemblage and site_id")
        biology["bio_site_id"] = (
            biology["assemblage"].astype("string")
            + ":"
            + biology["site_id"].astype("string")
        )
    if "year" not in biology:
        if "sample_year" not in biology:
            raise KeyError("Biology table needs year or sample_year")
        biology["year"] = pd.to_numeric(biology["sample_year"], errors="coerce").astype("Int64")

    if selected_matches.duplicated(MATCH_KEYS).any():
        raise ValueError("Selected matches must be unique by biology-site year")
    selection = selected_matches.copy()

    # Prefix match fields before joining so they cannot be confused with fields
    # describing the biological survey itself.
    selection = selection.rename(
        columns={
            column: f"custom_match_{column}"
            for column in selection.columns
            if column not in MATCH_KEYS
        }
    )
    out = biology.merge(selection, on=MATCH_KEYS, how="left", validate="many_to_one")

    # Chemistry features are joined using the station selected by this recipe,
    # not the station chosen by the standard recipe.
    features = wq_site_year_features.copy().rename(
        columns={"wq_site_id": "custom_match_wq_site_id"}
    )
    if features.duplicated(["custom_match_wq_site_id", "year"]).any():
        raise ValueError("Water-quality features must be unique by station-year")
    out = out.merge(
        features,
        on=["custom_match_wq_site_id", "year"],
        how="left",
        validate="many_to_one",
    )
    out["custom_recipe_name"] = recipe_name
    out["custom_recipe_match_available"] = out["custom_match_wq_site_id"].notna()
    return out


def recipe_terms_table(recipe: Mapping[str, Any]) -> pd.DataFrame:
    """Return a compact human-readable representation of a validated recipe."""

    # This small table is intended for display in the notebook and records the
    # weighting choices alongside the results.
    config = _validated_recipe(recipe)
    rows = []
    for column, points in config["base_component_points"].items():
        rows.append(
            {
                "term": column,
                "source": "default score component",
                "transform": f"fraction of {BASE_COMPONENT_MAXIMA[column]:g}-point maximum",
                "points": float(points),
                "missing": "exclude",
            }
        )
    for term in config.get("evidence_terms", []):
        rows.append(
            {
                "term": str(term["column"]),
                "source": str(term.get("source", "user-supplied candidate evidence")),
                "transform": str(term["transform"])
                + (f"; cap={float(term['cap']):g}" if term.get("cap") is not None else ""),
                "points": float(term["points"]),
                "missing": str(term.get("missing", "exclude")),
            }
        )
    return pd.DataFrame(rows)
