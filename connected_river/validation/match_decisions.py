"""Scientific eligibility and use decisions for biology-chemistry links.

Ranking and scientific acceptability are handled separately. A candidate may
rank first but still be held for review or rejected. If none of the candidates
pass the checks, the method returns no usable chemistry match.
"""

from __future__ import annotations

import pandas as pd

KEYS = ["bio_site_id", "year"]


def _boolean(
    table: pd.DataFrame,
    column: str,
    *,
    default: bool = False,
) -> pd.Series:
    if column not in table:
        return pd.Series(default, index=table.index, dtype="boolean")
    return table[column].fillna(default).astype(bool)


def _number(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(float("nan"), index=table.index, dtype="float64")
    return pd.to_numeric(table[column], errors="coerce")


def _route_modifier_masks(
    table: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Return explicit reject/review masks for available route evidence."""
    treatment_works = _number(table, "path_treatment_works").gt(0)
    permitted_outlets = _number(table, "path_permitted_sewage_outlets").gt(0)
    actual_spills = _number(table, "path_counted_spills").gt(0)
    crosses_reservoir = _boolean(table, "crosses_reservoir")
    crosses_tidal_boundary = _boolean(table, "crosses_tidal_boundary")
    crosses_major_abstraction = _boolean(table, "crosses_major_abstraction")
    physical_break = crosses_reservoir | crosses_tidal_boundary
    return (
        {
            "treatment_works_between_sites": treatment_works,
            "reservoir_between_sites": crosses_reservoir,
            "tidal_boundary_between_sites": crosses_tidal_boundary,
        },
        {
            "intervening_permitted_or_spilling_sewage_outlet": (
                (permitted_outlets | actual_spills) & ~treatment_works
            ),
            "major_abstraction_between_sites": (
                crosses_major_abstraction & ~physical_break
            ),
        },
    )


def _append_reason(
    reasons: pd.Series,
    mask: pd.Series,
    reason: str,
) -> pd.Series:
    """Append one semicolon-delimited reason without duplicating it."""
    out = reasons.fillna("").astype(str).copy()
    mask = mask.fillna(False).astype(bool)
    missing_reason = mask & ~out.str.split("; ").map(lambda values: reason in values)
    out.loc[missing_reason] = out.loc[missing_reason].str.cat(
        pd.Series(reason, index=out.index).loc[missing_reason], sep="; "
    ).str.strip("; ")
    return out


# Apply evidence checks before choosing a winner. The score orders suitable
# candidates; it cannot turn a failed route or weak evidence into an acceptance.
def classify_candidate_matches(
    candidates: pd.DataFrame,
    *,
    min_sampling_dates: int = 3,
    min_target_determinands: int = 3,
    max_snap_distance_m: float = 250.0,
    allow_proxy_minor_tributaries: bool = False,
    allow_downstream_wq: bool = True,
) -> pd.DataFrame:
    """Apply physical and evidence-quality gates before choosing a candidate.

    The current tributary quantity is a gauge-calibrated network-size proxy,
    not observed flow or chemistry mixing.  Until the chemistry-coherence and
    expert-review programme validates a minor-tributary acceptance threshold,
    a proxy-only intervening tributary is review-only by default.
    """
    out = candidates.copy()
    required = {
        "bio_site_id", "year", "wq_site_id", "network_relationship",
        "network_distance_m", "bio_snap_distance_m", "wq_snap_distance_m",
        "wq_n_sampling_dates", "wq_n_target_determinands",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError("Candidate table missing: " + ", ".join(missing))

    direction = out.get("network_direction", out["network_relationship"]).astype(
        "string"
    )
    downstream = direction.eq("wq_downstream_of_bio")

    checks = {
        "sampling_dates_adequate": _number(out, "wq_n_sampling_dates").ge(
            min_sampling_dates
        ),
        "chemistry_breadth_adequate": _number(
            out, "wq_n_target_determinands"
        ).ge(min_target_determinands),
        "biology_snap_acceptable": _number(out, "bio_snap_distance_m").le(
            max_snap_distance_m
        ),
        "chemistry_snap_acceptable": _number(out, "wq_snap_distance_m").le(
            max_snap_distance_m
        ),
    }
    for name, values in checks.items():
        out[name] = values.fillna(False)
    if "match_class" in out:
        # Score classes describe an ordinal ranking aid.  They are deliberately
        # not a scientific eligibility gate because the score is not a
        # calibrated probability of ecological representativeness.
        out["match_class_acceptable"] = out["match_class"].isin(["A", "B"])

    tributary_count = _number(out, "intervening_tributary_count")
    tributary_present = tributary_count.gt(0)
    influence_class = out.get(
        "tributary_influence_class",
        pd.Series(pd.NA, index=out.index, dtype="string"),
    ).astype("string").str.strip().str.lower()
    evidence_available = _boolean(
        out, "tributary_penalty_evidence_available", default=False
    )
    evidence_unavailable = (
        influence_class.isna()
        | influence_class.eq("")
        | influence_class.eq("unavailable")
        | ~evidence_available
    )
    evidence_unknown = influence_class.eq("unknown").fillna(False)
    structural_major = _boolean(
        out, "major_structural_tributary_between_sites"
    )
    individual_major = _boolean(
        out, "major_influence_tributary_between_sites"
    )
    combined_major = _boolean(out, "combined_tributary_influence_major")
    major = _boolean(out, "major_tributary_between_sites")
    complete = _boolean(out, "tributary_topology_complete")
    coverage_complete = _number(
        out, "tributary_attribute_coverage_fraction"
    ).eq(1.0)
    strong_major = (
        structural_major
        & individual_major
        & combined_major
        & complete
        & coverage_complete
    )

    reject_masks = {
        "insufficient_sampling_dates": ~checks["sampling_dates_adequate"],
        "insufficient_target_determinand_breadth": ~checks[
            "chemistry_breadth_adequate"
        ],
        "biology_snap_exceeds_limit": ~checks["biology_snap_acceptable"],
        "chemistry_snap_exceeds_limit": ~checks["chemistry_snap_acceptable"],
        "downstream_station_excluded_by_direction_policy": (
            downstream & (not allow_downstream_wq)
        ),
    }
    review_masks = {
        "strong_proxy_major_tributary_requires_expert_review": strong_major,
        "proxy_major_tributary_requires_expert_review": major & ~strong_major,
        "proxy_minor_tributary_threshold_not_empirically_validated": (
            tributary_present & ~major & (not allow_proxy_minor_tributaries)
        ),
        "tributary_influence_evidence_unavailable": evidence_unavailable,
        "tributary_influence_evidence_unknown": evidence_unknown,
    }
    route_reject_masks, route_review_masks = _route_modifier_masks(out)
    reject_masks.update(route_reject_masks)
    review_masks.update(route_review_masks)

    reject = pd.Series(False, index=out.index)
    review = pd.Series(False, index=out.index)
    for values in reject_masks.values():
        reject |= values.fillna(True).astype(bool)
    for values in review_masks.values():
        review |= values.fillna(False).astype(bool)

    out["candidate_gate_decision"] = "accept"
    out.loc[review, "candidate_gate_decision"] = "review"
    out.loc[reject, "candidate_gate_decision"] = "reject"
    safe_reject_masks = {
        name: values.fillna(True).astype(bool) for name, values in reject_masks.items()
    }
    safe_review_masks = {
        name: values.fillna(False).astype(bool) for name, values in review_masks.items()
    }
    out["candidate_rejection_reasons"] = [
        "; ".join(
            name for name, values in safe_reject_masks.items() if values.iloc[index]
        )
        for index in range(len(out))
    ]
    out["candidate_review_reasons"] = [
        "; ".join(
            name for name, values in safe_review_masks.items() if values.iloc[index]
        )
        for index in range(len(out))
    ]
    out["candidate_use_recommended"] = out["candidate_gate_decision"].eq("accept")
    out["strong_proxy_major_tributary"] = strong_major
    out["tributary_acceptance_policy"] = (
        "proxy_minor_allowed_after_external_validation"
        if allow_proxy_minor_tributaries
        else "any_proxy_only_intervening_tributary_review_until_validated"
    )
    out["direction_policy"] = (
        "bidirectional_connected_river_context"
        if allow_downstream_wq
        else "upstream_only"
    )
    optional_modifier_fields = [
        column
        for column in (
            "crosses_reservoir", "crosses_tidal_boundary", "crosses_major_abstraction"
        )
        if column in out
    ]
    route_fields_present = any(
        column in out
        for column in (
            "path_treatment_works", "path_permitted_sewage_outlets",
            "path_counted_spills", *optional_modifier_fields,
        )
    )
    if route_fields_present:
        out["route_modifier_screen_status"] = (
            "tributary_and_sewage_only; reservoir_tidal_abstraction_layers_not_supplied"
            if not optional_modifier_fields
            else "screened_available_fields:" + ",".join(optional_modifier_fields)
        )
    return out


# Select the best accepted candidate first, falling back to review only when no
# accepted option exists. Rejected rows remain available for checking.
def select_candidates_by_scientific_gate(
    classified_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Choose the best acceptable candidate, retaining review/reject fallbacks.

    An accepted lower-ranked candidate is preferred to a review-only top score.
    If no candidate is acceptable, the best review case remains visible but is
    not recommended for analysis.  A rejected row is retained only when every
    candidate failed, while keeping the reasons for no match.
    """
    required = {*KEYS, "wq_site_id", "candidate_gate_decision"}
    missing = sorted(required - set(classified_candidates.columns))
    if missing:
        raise ValueError("Classified candidate table missing: " + ", ".join(missing))
    out = classified_candidates.copy()
    if "selection_confidence" in out:
        out["algorithmic_selection_confidence"] = out["selection_confidence"]
    priority = out["candidate_gate_decision"].map(
        {"accept": 0, "review": 1, "reject": 2}
    ).fillna(3)
    out["_scientific_gate_priority"] = priority
    sort_columns = KEYS + ["_scientific_gate_priority"]
    ascending = [True, True, True]
    if "match_score" in out:
        sort_columns.extend(["match_score", "network_distance_m", "wq_site_id"])
        ascending.extend([False, True, True])
    else:
        sort_columns.append("match_rank")
        ascending.append(True)
    out = out.sort_values(sort_columns, ascending=ascending).copy()
    pool_keys = KEYS + ["candidate_gate_decision"]
    out["scientific_match_rank"] = out.groupby(pool_keys).cumcount() + 1
    out["scientific_candidate_count"] = out.groupby(pool_keys)["wq_site_id"].transform(
        "size"
    )
    if "match_score" in out:
        out["scientific_runner_up_score"] = out.groupby(pool_keys)[
            "match_score"
        ].transform(lambda values: values.iloc[1] if len(values) > 1 else pd.NA)
        out["scientific_score_margin"] = (
            pd.to_numeric(out["match_score"], errors="coerce")
            - pd.to_numeric(out["scientific_runner_up_score"], errors="coerce")
        ).where(out["scientific_match_rank"].eq(1))
    else:
        out["scientific_runner_up_score"] = pd.NA
        out["scientific_score_margin"] = pd.NA
    selected = out.drop_duplicates(KEYS, keep="first").copy()
    selected["algorithmic_match_rank"] = selected.get(
        "match_rank", pd.Series(pd.NA, index=selected.index)
    )
    selected["selection_confidence"] = "ambiguous"
    unopposed = selected["scientific_candidate_count"].eq(1)
    margin = pd.to_numeric(selected["scientific_score_margin"], errors="coerce")
    selected.loc[unopposed, "selection_confidence"] = "unopposed"
    selected.loc[~unopposed & margin.ge(15), "selection_confidence"] = "high"
    selected.loc[
        ~unopposed & margin.ge(7.5) & margin.lt(15), "selection_confidence"
    ] = "moderate"
    selected["scientific_selection_method"] = (
        "best_route_screened_accepted_else_review_else_rejected_audit_fallback"
    )
    selected = selected.drop(columns="_scientific_gate_priority")
    return selected


def classify_selected_matches(
    selected: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    route_context: pd.DataFrame | None = None,
    min_sampling_dates: int = 3,
    min_target_determinands: int = 3,
    max_snap_distance_m: float = 250,
    allow_proxy_minor_tributaries: bool = False,
    allow_downstream_wq: bool = True,
) -> pd.DataFrame:
    """Make the final accept/review/reject decision with route modifiers."""
    if "candidate_gate_decision" not in selected:
        out = classify_candidate_matches(
            selected,
            min_sampling_dates=min_sampling_dates,
            min_target_determinands=min_target_determinands,
            max_snap_distance_m=max_snap_distance_m,
            allow_proxy_minor_tributaries=allow_proxy_minor_tributaries,
            allow_downstream_wq=allow_downstream_wq,
        )
    else:
        out = selected.copy()

    sensitivity_columns = KEYS + [
        column
        for column in ("distance_sensitive", "tributary_penalty_sensitive")
        if column in sensitivity
    ]
    out = out.merge(
        sensitivity[sensitivity_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if route_context is not None and not route_context.empty:
        context_columns = [
            column
            for column in route_context
            if column not in out or column in {*KEYS, "wq_site_id"}
        ]
        join_keys = [
            column for column in (*KEYS, "wq_site_id")
            if column in route_context and column in out
        ]
        out = out.merge(
            route_context[context_columns],
            on=join_keys,
            how="left",
            validate="one_to_one",
        )

    decision = out["candidate_gate_decision"].astype("string").copy()
    rejection_reasons = out.get(
        "candidate_rejection_reasons", pd.Series("", index=out.index)
    ).fillna("").astype(str)
    review_reasons = out.get(
        "candidate_review_reasons", pd.Series("", index=out.index)
    ).fillna("").astype(str)

    route_reject_masks, route_review_masks = _route_modifier_masks(out)
    for reason, mask in route_reject_masks.items():
        decision.loc[mask] = "reject"
        rejection_reasons = _append_reason(rejection_reasons, mask, reason)
    for reason, mask in route_review_masks.items():
        decision.loc[mask & decision.eq("accept")] = "review"
        review_reasons = _append_reason(review_reasons, mask, reason)

    ambiguous = out.get(
        "selection_confidence", pd.Series("", index=out.index)
    ).astype("string").eq("ambiguous")
    distance_sensitive = _boolean(out, "distance_sensitive")
    methodological_review = (ambiguous | distance_sensitive) & decision.eq("accept")
    decision.loc[methodological_review] = "review"
    review_reasons.loc[ambiguous] = review_reasons.loc[ambiguous].str.cat(
        pd.Series("ambiguous_candidate_ranking", index=out.index)[ambiguous], sep="; "
    ).str.strip("; ")
    review_reasons.loc[distance_sensitive] = review_reasons.loc[
        distance_sensitive
    ].str.cat(
        pd.Series("distance_threshold_sensitive", index=out.index)[distance_sensitive],
        sep="; ",
    ).str.strip("; ")

    out["scientific_decision"] = decision
    optional_modifier_fields = [
        column
        for column in (
            "crosses_reservoir", "crosses_tidal_boundary", "crosses_major_abstraction"
        )
        if column in out
    ]
    out["route_modifier_screen_status"] = (
        "tributary_and_sewage_only; reservoir_tidal_abstraction_layers_not_supplied"
        if not optional_modifier_fields
        else "screened_available_fields:" + ",".join(optional_modifier_fields)
    )
    out["match_rule_failures"] = rejection_reasons
    out["match_review_reasons"] = review_reasons
    out["match_use_recommended"] = decision.eq("accept")
    out["match_decision"] = "accepted"
    out.loc[decision.eq("reject"), "match_decision"] = "no_defensible_match"
    review = decision.eq("review")
    out.loc[review, "match_decision"] = "provisional_scientific_review"
    out.loc[
        review & review_reasons.str.contains("major_tributary", na=False),
        "match_decision",
    ] = "provisional_major_tributary"
    out.loc[
        review & review_reasons.str.contains("proxy_minor_tributary", na=False),
        "match_decision",
    ] = "provisional_intervening_tributary"
    out.loc[
        review & review_reasons.str.contains("evidence_unavailable", na=False),
        "match_decision",
    ] = "provisional_tributary_influence_unavailable"
    out.loc[
        review & review_reasons.str.contains("evidence_unknown", na=False),
        "match_decision",
    ] = "provisional_tributary_influence_unknown"
    out.loc[
        review & review_reasons.str.contains("permitted_or_spilling", na=False),
        "match_decision",
    ] = "provisional_intervening_discharge"
    out.loc[
        review & review_reasons.str.contains("major_abstraction", na=False),
        "match_decision",
    ] = "provisional_intervening_abstraction"
    out.loc[
        review & review_reasons.str.contains("distance_threshold_sensitive", na=False),
        "match_decision",
    ] = "provisional_distance_sensitive"
    return out
