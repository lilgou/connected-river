from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from connected_river.utils.config import resolve_path


def distance_score(distance_m: float | None, max_distance_m: float = 10000) -> float:
    if pd.isna(distance_m):
        return 0.0
    return max(0.0, 35.0 * (1.0 - min(float(distance_m), max_distance_m) / max_distance_m))


def snap_score(bio_snap_m: float | None, wq_snap_m: float | None, max_snap_m: float = 250) -> float:
    vals = [v for v in [bio_snap_m, wq_snap_m] if pd.notna(v)]
    if not vals:
        return 0.0
    mean_snap = sum(vals) / len(vals)
    return max(0.0, 15.0 * (1.0 - min(mean_snap, max_snap_m) / max_snap_m))


def direction_score(relationship: str) -> float:
    """Retain direction as evidence without adding a constant ranking bonus.

    The default link is river-chemistry context, not a claim that the
    chemistry station measures antecedent exposure at the biology location.
    Every admitted candidate is already river-connected, so awarding every
    direction the same positive score adds no information and can make weak
    candidates appear stronger than they are.
    """
    return 0.0


def temporal_score(n_sampling_dates: float | None, n_months_sampled: float | None) -> float:
    dates = 0 if pd.isna(n_sampling_dates) else float(n_sampling_dates)
    months = 0 if pd.isna(n_months_sampled) else float(n_months_sampled)

    date_component = min(dates / 6.0, 1.0) * 10.0
    month_component = min(months / 6.0, 1.0) * 10.0

    return date_component + month_component


def chemistry_score(n_target_determinands: float | None, has_core: bool, has_metals: bool) -> float:
    n = 0 if pd.isna(n_target_determinands) else float(n_target_determinands)
    score = min(n / 12.0, 1.0) * 10.0

    if bool(has_core):
        score += 7.0
    if bool(has_metals):
        score += 3.0

    return min(score, 20.0)


def transport_plausibility_score(
    distance_m: float | None,
    represented_fraction: float | None,
    *,
    decay_scale_m: float | None = None,
    max_points: float = 10.0,
) -> float:
    """Score the represented-flow evidence without scoring distance twice.

    Network distance already contributes up to 35 points.  The tributary
    transport proxy also carries its own configured decay where available, so
    applying another whole-route exponential decay here would double count
    proximity.  ``decay_scale_m`` remains accepted for API compatibility but is
    deliberately ignored.
    """
    represented = 0.0 if pd.isna(represented_fraction) else float(represented_fraction)
    if not math.isfinite(represented):
        return 0.0
    represented = max(0.0, min(represented, 1.0))
    if represented <= 0:
        return 0.0

    return max(0.0, max_points * represented)


def classify(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


# The score ranks candidates; it does not decide whether a match is suitable.
# The evidence checks are applied separately so a high score cannot hide a gap.
def score_candidate_matches(
    df: pd.DataFrame, *,
    tributary_penalty_points: float = 0.0,
    transport_plausibility_decay_scale_m: float = 10000.0,
) -> pd.DataFrame:
    out = df.copy()
    tributary_penalty_points = float(tributary_penalty_points)
    if not np.isfinite(tributary_penalty_points) or tributary_penalty_points < 0:
        raise ValueError("tributary_penalty_points must be a finite non-negative number")
    transport_plausibility_decay_scale_m = float(transport_plausibility_decay_scale_m)
    if (
        not np.isfinite(transport_plausibility_decay_scale_m)
        or transport_plausibility_decay_scale_m <= 0
    ):
        raise ValueError(
            "transport_plausibility_decay_scale_m must be finite and greater than zero"
        )

    out["score_distance"] = out["network_distance_m"].apply(distance_score)
    out["score_snap"] = [
        snap_score(b, w)
        for b, w in zip(out["bio_snap_distance_m"], out["wq_snap_distance_m"])
    ]
    direction = out.get("network_direction", out["network_relationship"])
    out["score_direction"] = direction.apply(direction_score)
    out["score_direction_method"] = (
        "direction_retained_for_audit_no_constant_ranking_points"
    )
    out["score_temporal"] = [
        temporal_score(d, m)
        for d, m in zip(out["wq_n_sampling_dates"], out["wq_n_months_sampled"])
    ]
    out["score_chemistry"] = [
        chemistry_score(n, core, metals)
        for n, core, metals in zip(
            out["wq_n_target_determinands"],
            out["wq_has_core_chemistry"],
            out["wq_has_metals"],
        )
    ]
    estimated_transport = pd.to_numeric(
        out.get("estimated_transport_represented_fraction", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    estimated_flow = pd.to_numeric(
        out.get("estimated_flow_represented_fraction", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    shreve_fraction = pd.to_numeric(
        out.get("represented_shreve_fraction", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["transport_evidence_fraction"] = (
        estimated_transport
        .combine_first(estimated_flow)
        .combine_first(shreve_fraction)
    )
    out["score_transport_plausibility"] = [
        transport_plausibility_score(
            distance,
            rep,
            decay_scale_m=scale,
        )
        for distance, rep, scale in zip(
            out["network_distance_m"],
            out["transport_evidence_fraction"],
            pd.to_numeric(
                out.get(
                    "transport_decay_scale_m",
                    pd.Series(
                        transport_plausibility_decay_scale_m,
                        index=out.index,
                    ),
                ),
                errors="coerce",
            ).fillna(transport_plausibility_decay_scale_m),
        )
    ]
    out["score_transport_plausibility_method"] = np.select(
        [
            out.get("estimated_transport_represented_fraction", pd.Series(np.nan, index=out.index)).notna(),
            out.get("estimated_flow_represented_fraction", pd.Series(np.nan, index=out.index)).notna(),
            out.get("represented_shreve_fraction", pd.Series(np.nan, index=out.index)).notna(),
        ],
        [
            "estimated_transport_flow_proxy_no_additional_route_decay",
            "estimated_flow_proxy_no_additional_route_decay",
            "shreve_proxy_no_additional_route_decay_fallback",
        ],
        default="not_applied_transport_evidence_unavailable",
    )

    out["match_score_before_tributary"] = (
        out["score_distance"]
        + out["score_snap"]
        + out["score_direction"]
        + out["score_temporal"]
        + out["score_chemistry"]
        + out["score_transport_plausibility"]
    ).clip(0.0, 100.0)
    if "unrepresented_shreve_fraction" in out:
        unrepresented_shreve = pd.to_numeric(
            out["unrepresented_shreve_fraction"], errors="coerce"
        )
    else:
        unrepresented_shreve = pd.Series(np.nan, index=out.index, dtype="float64")
    if "estimated_transport_unrepresented_fraction" in out:
        unrepresented_transport_proxy = pd.to_numeric(
            out["estimated_transport_unrepresented_fraction"], errors="coerce"
        )
    else:
        unrepresented_transport_proxy = pd.Series(np.nan, index=out.index, dtype="float64")
    if "estimated_flow_unrepresented_fraction" in out:
        unrepresented_flow_proxy = pd.to_numeric(
            out["estimated_flow_unrepresented_fraction"], errors="coerce"
        )
    else:
        unrepresented_flow_proxy = pd.Series(np.nan, index=out.index, dtype="float64")
    penalty_fraction = (
        unrepresented_transport_proxy
        .combine_first(unrepresented_flow_proxy)
        .combine_first(unrepresented_shreve)
    )
    out["tributary_penalty_fraction_used"] = penalty_fraction.clip(0.0, 1.0)
    out["tributary_penalty_evidence_available"] = penalty_fraction.notna()
    out["score_tributary_penalty"] = (
        out["tributary_penalty_fraction_used"] * tributary_penalty_points
    ).fillna(0.0)
    out["tributary_penalty_points_max"] = tributary_penalty_points
    out["score_tributary_penalty_method"] = np.select(
        [
            unrepresented_transport_proxy.notna(),
            unrepresented_flow_proxy.notna(),
            unrepresented_shreve.notna(),
        ],
        [
            "calibrated_transport_flow_proxy_fraction_x_configured_penalty_points",
            "calibrated_flow_proxy_fraction_x_configured_penalty_points",
            "shreve_fraction_x_configured_penalty_points_fallback",
        ],
        default="not_applied_tributary_fraction_unavailable",
    )
    out["match_score"] = (
        out["match_score_before_tributary"] - out["score_tributary_penalty"]
    ).clip(0.0, 100.0)

    out["match_class"] = out["match_score"].apply(classify)

    out["match_rank"] = (
        out.sort_values(
            ["bio_site_id", "match_score", "network_distance_m"],
            ascending=[True, False, True],
        )
        .groupby("bio_site_id")
        .cumcount()
        + 1
    )
    return out


def score_matches_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tributary_penalty_points: float = 0.0,
) -> Path:
    input_path = resolve_path(input_path)
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    out = score_candidate_matches(df, tributary_penalty_points=tributary_penalty_points)
    out.to_parquet(output_path, index=False)
    return output_path


def best_matches_file(scored_path: str | Path, output_path: str | Path) -> Path:
    scored_path = resolve_path(scored_path)
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(scored_path)
    best = df[df["match_rank"] == 1].copy()
    best = (
        best.sort_values(["bio_site_id", "match_score", "network_distance_m"], ascending=[True, False, True])
        .drop_duplicates("bio_site_id", keep="first")
        .copy()
    )
    best.to_parquet(output_path, index=False)
    return output_path
