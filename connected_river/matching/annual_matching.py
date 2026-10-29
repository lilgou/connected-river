"""Default ranking and uncertainty summaries for annual candidate matches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from connected_river.matching.score_matches import score_candidate_matches

KEYS = ["bio_site_id", "year"]


# Ranking is done within each biological site and year. This stops a well-
# monitored station in one year from automatically winning in every year.
def rank_annual_matches(
    candidates: pd.DataFrame, *, tributary_penalty_points: float = 0.0,
    transport_plausibility_decay_scale_m: float = 10000.0,
) -> pd.DataFrame:
    """Score every candidate and retain information needed to audit the choice.

    Confidence describes separation among candidates under this algorithm. It is
    not the probability that a pairing is ecologically correct.
    """
    required = {
        "bio_site_id", "year", "wq_site_id", "network_distance_m",
        "bio_snap_distance_m", "wq_snap_distance_m", "network_relationship",
        "wq_n_sampling_dates", "wq_n_months_sampled", "wq_n_target_determinands",
        "wq_has_core_chemistry", "wq_has_metals",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError("Candidate table missing: " + ", ".join(missing))
    out = score_candidate_matches(
        candidates,
        tributary_penalty_points=tributary_penalty_points,
        transport_plausibility_decay_scale_m=transport_plausibility_decay_scale_m,
    ).reset_index(drop=True)
    pre_tributary = out.sort_values(
        KEYS + ["match_score_before_tributary", "network_distance_m", "wq_site_id"],
        ascending=[True, True, False, True, True],
    ).copy()
    pre_tributary["match_rank_before_tributary"] = pre_tributary.groupby(KEYS).cumcount() + 1
    out["match_rank_before_tributary"] = pre_tributary[
        "match_rank_before_tributary"
    ].reindex(out.index)
    out = out.sort_values(KEYS + ["match_score", "network_distance_m", "wq_site_id"], ascending=[True, True, False, True, True]).copy()
    out["match_rank"] = out.groupby(KEYS).cumcount() + 1
    pre_winners = (
        out.loc[out["match_rank_before_tributary"].eq(1), KEYS + ["wq_site_id"]]
        .rename(columns={"wq_site_id": "_pre_tributary_winner"})
    )
    final_winners = (
        out.loc[out["match_rank"].eq(1), KEYS + ["wq_site_id"]]
        .rename(columns={"wq_site_id": "_final_tributary_winner"})
    )
    winner_comparison = pre_winners.merge(final_winners, on=KEYS, how="outer", validate="one_to_one")
    winner_comparison["winner_changed_by_tributary"] = (
        winner_comparison["_pre_tributary_winner"].astype("string")
        != winner_comparison["_final_tributary_winner"].astype("string")
    ).fillna(False)
    out = out.merge(
        winner_comparison[KEYS + ["winner_changed_by_tributary"]],
        on=KEYS,
        how="left",
        validate="many_to_one",
    )
    out["candidate_count"] = out.groupby(KEYS)["wq_site_id"].transform("size")
    out["runner_up_score"] = out.groupby(KEYS)["match_score"].transform(lambda s: s.iloc[1] if len(s) > 1 else np.nan)
    out["score_margin"] = np.where(out["match_rank"].eq(1), out["match_score"] - out["runner_up_score"], np.nan)

    def confidence(row: pd.Series) -> str:
        if row["match_rank"] != 1:
            return "candidate"
        if row["candidate_count"] == 1:
            return "unopposed"
        margin = row["score_margin"]
        if row["match_class"] == "A" and margin >= 15:
            return "high"
        if row["match_class"] in {"A", "B"} and margin >= 7.5:
            return "moderate"
        return "ambiguous"

    out["selection_confidence"] = out.apply(confidence, axis=1)
    return out


# Re-run the recipe with small changes to show how much the chosen settings
# affect coverage and station choice.
def sensitivity_summary(
    ranked: pd.DataFrame,
    distance_limits: tuple[int, ...] = (5000, 7500, 10000),
    tributary_penalty_limits: tuple[int, ...] = (0, 5, 10, 15),
) -> pd.DataFrame:
    """Report whether distance or tributary-penalty choices change the winner.

    Whole-table sorts avoid a Python loop and repeated small DataFrame sorts for
    every biology site-year while preserving the declared winner and tie-break rules.
    """
    if ranked.empty:
        return pd.DataFrame()

    sort_columns = KEYS + ["match_score", "network_distance_m", "wq_site_id"]
    sort_ascending = [True, True, False, True, True]
    base = (
        ranked.sort_values(sort_columns, ascending=sort_ascending)
        .drop_duplicates(KEYS)
        [KEYS + ["wq_site_id"]]
        .rename(columns={"wq_site_id": "selected_wq_site_id"})
    )
    out = base.copy()

    distance_columns: list[str] = []
    network_distance = pd.to_numeric(ranked["network_distance_m"], errors="coerce")
    for limit in distance_limits:
        column = f"winner_at_{limit}m"
        distance_columns.append(column)
        eligible = ranked.loc[network_distance.le(limit)]
        winners = (
            eligible.sort_values(sort_columns, ascending=sort_ascending)
            .drop_duplicates(KEYS)
            [KEYS + ["wq_site_id"]]
            .rename(columns={"wq_site_id": column})
        )
        out = out.merge(winners, on=KEYS, how="left", validate="one_to_one")

    if distance_columns:
        normalised_distance_winners = out[distance_columns].astype("string")
        out["distance_sensitive"] = (
            normalised_distance_winners.nunique(axis=1, dropna=True).gt(1)
            | normalised_distance_winners.notna().sum(axis=1).lt(len(distance_columns))
        )
    else:
        out["distance_sensitive"] = False

    flow_fraction = pd.to_numeric(
        ranked.get(
            "estimated_flow_unrepresented_fraction",
            pd.Series(np.nan, index=ranked.index),
        ),
        errors="coerce",
    )
    shreve_fraction = pd.to_numeric(
        ranked.get(
            "unrepresented_shreve_fraction",
            pd.Series(np.nan, index=ranked.index),
        ),
        errors="coerce",
    )
    unrepresented = (
        flow_fraction.combine_first(shreve_fraction).fillna(0.0).clip(0.0, 1.0)
    )
    score_before = pd.to_numeric(
        ranked.get("match_score_before_tributary", ranked["match_score"]),
        errors="coerce",
    )
    penalty_columns: list[str] = []
    for configured_points in tributary_penalty_limits:
        penalty_points = float(configured_points)
        if not np.isfinite(penalty_points) or penalty_points < 0:
            raise ValueError(
                "tributary_penalty_limits must contain finite non-negative values"
            )
        column = f"winner_at_tributary_penalty_{penalty_points:g}pt"
        penalty_columns.append(column)
        scored = ranked[KEYS + ["network_distance_m", "wq_site_id"]].copy()
        scored["_tributary_sensitivity_score"] = (
            score_before - unrepresented * penalty_points
        )
        winners = (
            scored.sort_values(
                KEYS
                + [
                    "_tributary_sensitivity_score",
                    "network_distance_m",
                    "wq_site_id",
                ],
                ascending=[True, True, False, True, True],
            )
            .drop_duplicates(KEYS)
            [KEYS + ["wq_site_id"]]
            .rename(columns={"wq_site_id": column})
        )
        out = out.merge(winners, on=KEYS, how="left", validate="one_to_one")

    if penalty_columns:
        out["tributary_penalty_sensitive"] = (
            out[penalty_columns]
            .astype("string")
            .nunique(axis=1, dropna=True)
            .gt(1)
        )
    else:
        out["tributary_penalty_sensitive"] = False
    return out


def scientific_sensitivity_summary(
    classified: pd.DataFrame,
    distance_limits: tuple[int, ...] = (5000, 7500, 10000),
    tributary_penalty_limits: tuple[int, ...] = (0, 5, 10, 15),
) -> pd.DataFrame:
    """Measure stability inside the decision pool used for scientific selection.

    Candidate gates can move the selected station away from the algorithmic
    rank-one row.  Sensitivity must therefore compare the acceptable candidates
    when any exist, otherwise the review pool, and only then rejected audit
    fallbacks.  Applying an all-candidate sensitivity flag to a lower-ranked
    scientific selection would describe the wrong competition.
    """
    if classified.empty:
        return pd.DataFrame()
    if "candidate_gate_decision" not in classified:
        raise ValueError("Scientific sensitivity requires candidate_gate_decision")
    priority = classified["candidate_gate_decision"].map(
        {"accept": 0, "review": 1, "reject": 2}
    ).fillna(3)
    best_priority = priority.groupby(
        [classified[column] for column in KEYS]
    ).transform("min")
    pool = classified.loc[priority.eq(best_priority)].copy()
    return sensitivity_summary(
        pool,
        distance_limits=distance_limits,
        tributary_penalty_limits=tributary_penalty_limits,
    )


def rank_matches_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    tributary_penalty_points: float = 0.0,
) -> dict[str, Path]:
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path, low_memory=False)
    ranked = rank_annual_matches(
        candidates, tributary_penalty_points=tributary_penalty_points
    )
    paths = {"ranked": output_dir / "ranked_matches.parquet", "selected": output_dir / "selected_matches.parquet", "sensitivity": output_dir / "match_sensitivity.parquet"}
    ranked.to_parquet(paths["ranked"], index=False)
    ranked[ranked["match_rank"].eq(1)].to_parquet(paths["selected"], index=False)
    sensitivity_summary(ranked).to_parquet(paths["sensitivity"], index=False)
    return paths
