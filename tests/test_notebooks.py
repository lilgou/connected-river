# Notebook structure

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from connected_river.figures import (
    compare_recipe_weights,
    plot_temporal_join,
    plot_workflow_overview,
    prepare_evidence_flow,
)

ROOT = Path(__file__).resolve().parents[1]
BUILD_NOTEBOOK = ROOT / "The_Connected_River_Part_1_Building.ipynb"
EXPLORE_NOTEBOOK = ROOT / "The_Connected_River_Part_2_Exploring.ipynb"
PROFILE = ROOT / "config/connected_river.yaml"
WORKFLOW = ROOT / "connected_river/workflow.py"
RELEASE = ROOT / "connected_river/pipeline/release.py"


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sources(path: Path) -> list[str]:
    return ["".join(cell.get("source", [])) for cell in _document(path)["cells"]]


def test_notebooks_are_valid_documents():
    for path in (BUILD_NOTEBOOK, EXPLORE_NOTEBOOK):
        document = _document(path)
        assert document["nbformat"] == 4
        assert any(cell["cell_type"] == "markdown" for cell in document["cells"])
        assert any(cell["cell_type"] == "code" for cell in document["cells"])
        assert not any(
            output.get("output_type") == "error"
            for cell in document["cells"]
            for output in cell.get("outputs", [])
        )


def test_notebooks_include_clear_opening_sections_and_cross_links():
    build = "\n".join(_sources(BUILD_NOTEBOOK))
    explore = "\n".join(_sources(EXPLORE_NOTEBOOK))

    assert "## What this notebook does" in build
    assert "## Before running" in build
    assert "## What this notebook explores" in explore
    assert "## What this analysis does not establish" in explore

    assert "./The_Connected_River_Part_2_Exploring.ipynb" in build
    assert "./The_Connected_River_Part_1_Building.ipynb" in explore


def test_build_notebook_calls_the_reusable_workflow():
    sources = _sources(BUILD_NOTEBOOK)
    joined = "\n".join(sources)
    tags = {
        tag
        for cell in _document(BUILD_NOTEBOOK)["cells"]
        for tag in cell.get("metadata", {}).get("tags", [])
    }

    assert "build_or_resume" in joined
    assert "antecedent_window_months" in joined
    assert "apply_table_filters" in joined
    assert "CUSTOM_MATCH_RECIPE" in joined
    assert {"user-settings", "advanced-settings"}.issubset(tags)


def test_explore_notebook_uses_the_completed_output():
    joined = "\n".join(_sources(EXPLORE_NOTEBOOK))
    required_calls = {
        "plot_spatial_support",
        "plot_river_stories",
        "plot_evidence_fingerprint",
        "plot_pressure_response_curves",
        "build_historical_continuation",
    }

    assert "build_or_resume" not in joined
    assert all(call in joined for call in required_calls)
    assert "display(interactive_map)" in joined
    assert "IFrame" not in joined
    assert "as_uri()" not in joined


def test_worked_release_starts_with_downloaded_2018_surveys():
    profile = PROFILE.read_text(encoding="utf-8")

    assert "online_matching_start_year: 2018" in profile
    assert "public_extension_start_year: 2018" in profile


def test_core_release_does_not_append_the_historical_deposit():
    delivery_code = WORKFLOW.read_text(encoding="utf-8") + RELEASE.read_text(
        encoding="utf-8"
    )

    assert "23f194b0" not in delivery_code
    assert "connected_river_integrated_data.parquet" in delivery_code


# Notebook figures


def test_evidence_flow_categories_are_mutually_exclusive():
    records = pd.DataFrame({
        "assemblage": ["freshwater_fish", "diatom", "macrophyte", "macroinvertebrate"],
        "candidate_count": [2, 1, 1, 0],
        "match_decision": [
            "accepted", "provisional_intervening_discharge",
            "no_defensible_match", "no_candidate_match",
        ],
        "chemistry_included_in_release": [True, False, False, False],
        "flow_evidence_available": [True, False, False, False],
    })

    flow = prepare_evidence_flow(records)

    assert flow["Biological programme"].tolist()[0] == "Freshwater fish"
    assert flow["Network connection"].value_counts().to_dict() == {
        "Candidate route found": 3,
        "No candidate route": 1,
    }
    assert flow["Scientific decision"].tolist() == [
        "Accepted route", "Review: discharge on route",
        "No defensible route", "No route to assess",
    ]
    assert flow["Joined data"].tolist() == [
        "Biology + chemistry + flow", "Biology + review record",
        "Biology retained; no defensible match",
        "Biology retained; no defensible match",
    ]


def test_notebook_overview_figures_render():
    workflow = plot_workflow_overview()
    temporal = plot_temporal_join(6)

    assert len(workflow.axes) == 1
    assert len(temporal.axes) == 1
    assert "quality-checked" in workflow.axes[0].get_title(loc="left")
    assert "Chemistry window" in temporal.axes[0].get_title(loc="left")


def test_recipe_weight_comparison_reports_station_changes():
    candidates = pd.DataFrame({
        "bio_site_id": ["A", "A"],
        "year": [2024, 2024],
        "wq_site_id": ["W1", "W2"],
        "network_distance_m": [500, 1500],
        "candidate_gate_decision": ["accept", "accept"],
        "candidate_review_reasons": ["", ""],
        "match_score": [90, 80],
        "score_distance": [35, 10],
        "score_snap": [15, 15],
        "score_temporal": [20, 20],
        "score_chemistry": [5, 20],
        "score_transport_plausibility": [10, 10],
    })
    recipe = {
        "name": "chemistry_emphasis",
        "base_component_points": {
            "score_distance": 5,
            "score_snap": 15,
            "score_temporal": 20,
            "score_chemistry": 40,
            "score_transport_plausibility": 20,
        },
        "evidence_terms": [],
        "allowed_review_reasons": [],
    }

    figure, summary, changes = compare_recipe_weights(
        candidates, recipe, total_site_years=1,
    )

    assert len(figure.axes) == 2
    assert summary.loc[1, "station changes vs default"] == 1
    assert changes[["default station", "alternative station"]].iloc[0].tolist() == [
        "W1", "W2",
    ]
