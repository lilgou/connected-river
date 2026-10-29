# Editable matching recipe

import pandas as pd
import pytest

from connected_river.analysis.custom_matching_recipe import (
    apply_custom_matching_recipe,
    select_custom_matches,
)
from connected_river.features.chemistry_features import (
    build_antecedent_chemistry_features,
)
from connected_river.matching.irn_matching import _antecedent_candidate_support
from connected_river.matching.prefilter import apply_table_filters, parse_filter
from connected_river.validation.match_decisions import (
    classify_candidate_matches,
    select_candidates_by_scientific_gate,
)


def test_custom_recipe_selects_the_highest_eligible_candidate():
    candidates = pd.DataFrame(
        {
            "bio_site_id": ["A", "A", "B"],
            "year": [2024, 2024, 2024],
            "wq_site_id": ["W1", "W2", "W3"],
            "network_distance_m": [1000, 500, 100],
            "candidate_gate_decision": ["accept", "accept", "reject"],
            "candidate_review_reasons": ["", "", ""],
            "score_distance": [20, 30, 35],
            "score_chemistry": [20, 5, 20],
        }
    )
    recipe = {
        "name": "chemistry_first",
        "base_component_points": {
            "score_distance": 20,
            "score_chemistry": 80,
        },
        "evidence_terms": [],
        "allowed_review_reasons": [],
    }

    scored = apply_custom_matching_recipe(candidates, recipe)
    selected = select_custom_matches(scored)

    assert selected[["bio_site_id", "wq_site_id"]].to_dict("records") == [
        {"bio_site_id": "A", "wq_site_id": "W1"}
    ]
    assert scored["custom_score"].between(0, 100).all()


def test_custom_recipe_requires_exactly_100_points():
    candidates = pd.DataFrame(
        {
            "bio_site_id": ["A"],
            "year": [2024],
            "wq_site_id": ["W1"],
            "network_distance_m": [1000],
            "candidate_gate_decision": ["accept"],
            "candidate_review_reasons": [""],
            "score_distance": [20],
            "score_chemistry": [20],
        }
    )
    recipe = {
        "name": "invalid_total",
        "base_component_points": {
            "score_distance": 20,
            "score_chemistry": 70,
        },
        "evidence_terms": [],
        "allowed_review_reasons": [],
    }

    with pytest.raises(ValueError, match="must total exactly 100"):
        apply_custom_matching_recipe(candidates, recipe)


def test_evidence_terms_are_included_in_the_100_point_total():
    candidates = pd.DataFrame(
        {
            "bio_site_id": ["A"],
            "year": [2024],
            "wq_site_id": ["W1"],
            "network_distance_m": [1000],
            "candidate_gate_decision": ["accept"],
            "candidate_review_reasons": [""],
            "score_distance": [35],
            "wq_has_metals": [True],
        }
    )
    recipe = {
        "name": "distance_and_metals",
        "base_component_points": {"score_distance": 80},
        "evidence_terms": [{
            "column": "wq_has_metals", "transform": "binary", "points": 20,
        }],
        "allowed_review_reasons": [],
    }

    scored = apply_custom_matching_recipe(candidates, recipe)

    assert scored.loc[0, "custom_score"] == pytest.approx(100)


# Candidate pre-filters


def test_filters_are_applied_with_an_audit():
    candidates = pd.DataFrame(
        {
            "wq_has_metals": [True, False, True],
            "network_direction": [
                "wq_upstream_of_bio",
                "wq_upstream_of_bio",
                "wq_downstream_of_bio",
            ],
        }
    )

    retained, audit = apply_table_filters(
        candidates,
        [
            "wq_has_metals:eq:true",
            "network_direction:eq:wq_upstream_of_bio",
        ],
        label="candidate routes",
    )

    assert retained.index.tolist() == [0]
    assert audit["logic"] == "AND"
    assert audit["rows_before"] == 3
    assert audit["rows_after"] == 1
    assert len(audit["steps"]) == 2


def test_numeric_filter_requires_a_number():
    rule = parse_filter("network_distance_m:le:10000")
    assert rule.value == 10000

    with pytest.raises(ValueError, match="numeric value"):
        apply_table_filters(
            pd.DataFrame({"network_distance_m": [100]}),
            ["network_distance_m:le:nearby"],
        )


# Scientific acceptance decisions


def candidate(station, rank, *, tributaries=0, major=False, dates=6):
    return {
        "bio_site_id": "BIO-1",
        "year": 2024,
        "wq_site_id": station,
        "match_rank": rank,
        "match_score": 90.0 - rank,
        "match_class": "A",
        "selection_confidence": "high",
        "network_relationship": "wq_upstream_of_bio",
        "network_direction": "wq_upstream_of_bio",
        "network_distance_m": 100.0 * rank,
        "bio_snap_distance_m": 10.0,
        "wq_snap_distance_m": 10.0,
        "wq_n_sampling_dates": dates,
        "wq_n_target_determinands": 8,
        "intervening_tributary_count": tributaries,
        "tributary_influence_class": "major" if major else "none",
        "tributary_penalty_evidence_available": True,
        "tributary_topology_complete": True,
        "tributary_attribute_coverage_fraction": 1.0,
        "major_tributary_between_sites": major,
        "major_structural_tributary_between_sites": major,
        "major_influence_tributary_between_sites": major,
        "combined_tributary_influence_major": major,
    }


def test_acceptance_gate_can_override_algorithmic_rank():
    candidates = pd.DataFrame(
        [
            candidate("WQ-MAJOR", 1, tributaries=1, major=True),
            candidate("WQ-CLEAN", 2),
        ]
    )

    classified = classify_candidate_matches(candidates)
    selected = select_candidates_by_scientific_gate(classified).iloc[0]

    assert selected["wq_site_id"] == "WQ-CLEAN"
    assert selected["algorithmic_match_rank"] == 2
    assert selected["candidate_gate_decision"] == "accept"


def test_treatment_works_on_route_rejects_the_candidate():
    row = {
        **candidate("WQ-TREATMENT", 1),
        "path_treatment_works": 1,
        "path_permitted_sewage_outlets": 1,
        "path_counted_spills": 0,
    }

    classified = classify_candidate_matches(pd.DataFrame([row])).iloc[0]

    assert classified["candidate_gate_decision"] == "reject"
    assert "treatment_works_between_sites" in classified[
        "candidate_rejection_reasons"
    ]


# Temporal matching window


def observations():
    frame = pd.DataFrame(
        {
            "wq_site_id": ["WQ1", "WQ1"],
            "sample_datetime": ["2020-04-01", "2020-10-01"],
            "determinand_key": ["zinc", "zinc"],
            "target_determinand_key": [pd.NA, pd.NA],
            "result_standardized_loq_half": [2.0, 4.0],
            "result_qualifier": ["=", "="],
            "is_matching_determinand": [True, True],
        }
    )
    frame["sample_datetime"] = pd.to_datetime(frame["sample_datetime"])
    return frame


def test_six_and_twelve_month_windows_use_different_evidence():
    candidates = pd.DataFrame(
        {
            "bio_site_id": ["BIO1"],
            "year": [2020],
            "wq_site_id": ["WQ1"],
            "survey_date": ["2020-12-31"],
        }
    )

    six_month = _antecedent_candidate_support(
        candidates,
        observations(),
        window_months=6,
    )
    twelve_month = _antecedent_candidate_support(
        candidates,
        observations(),
        window_months=12,
    )

    assert six_month.loc[0, "wq_n_sampling_dates"] == 1
    assert twelve_month.loc[0, "wq_n_sampling_dates"] == 2
    assert twelve_month.loc[0, "candidate_support_window_months"] == 12


def test_joined_columns_record_the_requested_window():
    surveys = pd.DataFrame(
        {
            "site_id": ["1"],
            "sample_id": ["S1"],
            "sample_date": ["2020-12-31"],
            "sample_year": [2020],
            "assemblage": ["macroinvertebrate"],
        }
    )
    selected = pd.DataFrame(
        {
            "bio_site_id": ["macroinvertebrate:1"],
            "year": [2020],
            "wq_site_id": ["WQ1"],
        }
    )

    result = build_antecedent_chemistry_features(
        surveys,
        selected,
        observations(),
        window_months=12,
        years=[2020],
    )

    assert result.loc[0, "chemistry_window_months"] == 12
    assert result.loc[0, "zinc_median_window"] == 3.0
    assert "zinc_median_6mo" not in result


@pytest.mark.parametrize("months", [0, 13])
def test_window_must_match_the_acquisition_contract(months):
    with pytest.raises(ValueError, match="between 1 and 12"):
        _antecedent_candidate_support(
            pd.DataFrame(),
            observations(),
            window_months=months,
        )
