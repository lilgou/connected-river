"""Method and current-evidence figures built from the joined output.

Long-term continuation figures remain with their data preparation in
``historical_bridge.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from PIL import Image as PILImage
from shapely import wkb
from shapely.geometry import LineString, MultiLineString, Point

from connected_river.analysis.custom_matching_recipe import (
    BASE_COMPONENT_MAXIMA,
    apply_custom_matching_recipe,
    recipe_terms_table,
    select_custom_matches,
)
from connected_river.chemistry_registry import determinand_registry_records
from connected_river.matching.prefilter import apply_table_filters
from connected_river.validation.match_decisions import (
    select_candidates_by_scientific_gate,
)

ASSEMBLAGE_LABELS = {
    "macroinvertebrate": "Macroinvertebrates",
    "freshwater_fish": "Freshwater fish",
    "diatom": "Diatoms",
    "macrophyte": "Macrophytes",
    "transitional_coastal_fish": "Transitional/coastal fish",
}


# Shared presentation helpers


def dataframe_markdown(frame: pd.DataFrame, *, include_index: bool = True) -> str:
    """Return a small dependency-free Markdown table for notebook summaries."""

    table = frame.copy()
    if include_index:
        index_name = table.index.name or ""
        table.insert(0, index_name, table.index)

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in table.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _save_figure(
    figure: plt.Figure,
    output_directory: str | Path | None,
    filename: str,
) -> Path | None:
    """Save a notebook figure when an output directory is supplied."""

    if output_directory is None:
        return None
    destination = Path(output_directory) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    return destination


# Part 1: matching method


# --- Method figures -------------------------------------------------------
# These first plots explain how the pipeline works and how changing a matching
# recipe affects the available evidence.
def plot_workflow_overview() -> plt.Figure:
    """Show the main steps from source data to the joined output."""

    figure, axis = plt.subplots(figsize=(15, 4.2))
    axis.set(xlim=(0, 15), ylim=(0, 4.2))
    axis.axis("off")
    stages = [
        (0.15, "1  ACQUIRE", "Public monitoring\n+ licensed network", "#657786"),
        (2.30, "2  RECORD", "Source dates, versions\nand file details", "#5F8F9D"),
        (4.45, "3  FORMAT", "IDs, dates, units\nand coordinates", "#5BC0BE"),
        (6.60, "4  FILTER", "Choose relevant sites\nand route context", "#F2B134"),
        (8.75, "5  RANK + ACCEPT", "Score eligible routes,\nthen apply checks", "#087E8B"),
        (10.90, "6  JOIN", "Pre-survey chemistry\nwindow", "#467A58"),
        (13.05, "7  EXPORT", "Joined tables, data\ndictionary and checks", "#D95D39"),
    ]
    for index, (x_value, title, description, colour) in enumerate(stages):
        axis.add_patch(FancyBboxPatch(
            (x_value, 1.20), 1.80, 1.75,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=colour, edgecolor="none", alpha=0.96,
        ))
        axis.text(
            x_value + 0.90, 2.55, title, ha="center", va="center",
            color="white", weight="bold", fontsize=9.5,
        )
        axis.text(
            x_value + 0.90, 1.86, description, ha="center", va="center",
            color="white", fontsize=8.3, linespacing=1.25,
        )
        if index < len(stages) - 1:
            axis.annotate(
                "", xy=(x_value + 2.10, 2.08), xytext=(x_value + 1.84, 2.08),
                arrowprops={"arrowstyle": "->", "color": "#17324D", "lw": 1.5},
            )
    for left, right, label in [
        (0.15, 6.25, "DATA PREPARATION"),
        (6.60, 10.55, "ADAPTABLE NETWORK MATCHING"),
        (10.90, 14.85, "JOINED DATASET"),
    ]:
        axis.plot([left, right], [0.78, 0.78], color="#B8C7CD", lw=2)
        axis.text(
            (left + right) / 2, 0.48, label, ha="center", va="center",
            color="#526876", fontsize=8, weight="bold",
        )
    axis.set_title(
        "From source data to a quality-checked joined dataset",
        loc="left", pad=12, weight="bold",
    )
    axis.text(
        0.15, 0.05,
        "Settings change the rules; saved tables show what was included, excluded and selected.",
        color="#526876", fontsize=8.5,
    )
    return figure


def plot_network_coverage(release_directory: str | Path) -> plt.Figure:
    """Map the candidate routes and summarise the network checks."""

    release_directory = Path(release_directory)
    network_qa = json.loads(
        (release_directory / "network_integration_qa.json").read_text()
    )
    map_geometry = gpd.read_file(
        release_directory / "catchment_map_geometry.geojson"
    ).to_crs("EPSG:4326")
    coverage = pd.read_parquet(
        release_directory / "river_connection_routes.parquet",
        columns=[
            "bio_site_id", "wq_site_id", "match_rank",
            "biology_longitude", "biology_latitude",
            "wq_longitude", "wq_latitude",
        ],
        filters=[("match_rank", "==", 1)],
    )
    biology = coverage.drop_duplicates("bio_site_id")
    chemistry = coverage.drop_duplicates("wq_site_id")

    figure, (map_axis, qa_axis) = plt.subplots(
        1, 2, figsize=(13.5, 6.7), gridspec_kw={"width_ratios": [1.25, 0.75]},
    )
    map_geometry.plot(
        ax=map_axis, color="#EDF3F4", edgecolor="#B7C8CD", linewidth=0.35,
    )
    map_axis.scatter(
        chemistry["wq_longitude"], chemistry["wq_latitude"],
        s=6, color="#087E8B", alpha=0.40, linewidths=0,
        label=f"Chemistry stations ({len(chemistry):,})",
    )
    map_axis.scatter(
        biology["biology_longitude"], biology["biology_latitude"],
        s=5, color="#D95D39", alpha=0.25, linewidths=0,
        label=f"Biology sites ({len(biology):,})",
    )
    map_axis.set_axis_off()
    map_axis.set_title("A. Monitoring coverage searched across England", loc="left")
    map_axis.legend(loc="lower left", frameon=False, fontsize=8)

    topology = network_qa["irn_topology"]
    rows = [
        ("IRN links checked", topology["raw_link_count"], None),
        ("Biology sites snapped", network_qa["biology_snapping"]["snapped_points"],
         network_qa["biology_snapping"]["snap_rate"]),
        ("Chemistry sites snapped", network_qa["water_quality_snapping"]["snapped_points"],
         network_qa["water_quality_snapping"]["snap_rate"]),
        ("Candidate routes retained", network_qa["candidate_rows"], None),
        ("Target site-years assessed", network_qa["biology_site_year_rows"], None),
    ]
    qa_axis.set(xlim=(0, 1), ylim=(0, 1))
    qa_axis.axis("off")
    qa_axis.set_title("B. Checks for the national run", loc="left")
    for index, (label, value, rate) in enumerate(rows):
        y_value = 0.84 - index * 0.16
        qa_axis.add_patch(FancyBboxPatch(
            (0.02, y_value - 0.075), 0.94, 0.115,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#F6F8FA", edgecolor="#D7E0E3",
        ))
        qa_axis.text(0.06, y_value, label, va="center", color="#465A65", fontsize=9)
        value_label = f"{int(value):,}" + (f"  ({100 * rate:.1f}%)" if rate is not None else "")
        qa_axis.text(
            0.92, y_value, value_label, va="center", ha="right",
            color="#17324D", weight="bold", fontsize=10,
        )
    figure.suptitle(
        "England-wide monitoring coverage and network checks",
        x=0.01, ha="left", weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_candidate_decision_example(
    ranked_candidates: pd.DataFrame,
    release_directory: str | Path,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Show why ranking and scientific acceptance are separate decisions."""

    top_review = ranked_candidates.loc[
        ranked_candidates["match_rank"].eq(1)
        & ranked_candidates["candidate_gate_decision"].eq("review")
        & ranked_candidates["bio_site_id"].astype(str).str.startswith("macroinvertebrate:")
        & ranked_candidates["year"].ge(2018)
        & ranked_candidates["network_distance_m"].between(500, 8_000)
    ].copy()
    alternatives = (
        ranked_candidates.loc[
            ranked_candidates["candidate_gate_decision"].eq("accept")
            & ranked_candidates["match_rank"].between(2, 5)
            & ranked_candidates["network_distance_m"].between(500, 8_000)
        ]
        .sort_values(["bio_site_id", "year", "match_rank"])
        .drop_duplicates(["bio_site_id", "year"])
        [["bio_site_id", "year", "match_rank"]]
        .rename(columns={"match_rank": "accepted_rank"})
    )
    options = top_review.merge(alternatives, on=["bio_site_id", "year"], how="inner")
    if options.empty:
        raise RuntimeError("No top-ranked review followed by an accepted candidate was found")
    choice = options.sort_values(
        ["year", "network_distance_m"], ascending=[False, False],
    ).iloc[0]
    candidates = ranked_candidates.loc[
        ranked_candidates["bio_site_id"].eq(choice["bio_site_id"])
        & ranked_candidates["year"].eq(choice["year"])
        & ranked_candidates["match_rank"].isin([1, int(choice["accepted_rank"])])
    ].sort_values("match_rank").copy()
    routes = pd.read_parquet(
        Path(release_directory) / "river_connection_routes.parquet",
        filters=[
            ("bio_site_id", "==", choice["bio_site_id"]),
            ("year", "==", int(choice["year"])),
        ],
    )
    routes = routes.loc[routes["match_rank"].isin(candidates["match_rank"])].sort_values("match_rank")
    if len(routes) != 2 or routes["route_wkb"].isna().any():
        raise RuntimeError("The worked example does not contain two renderable routes")

    accepted_rank = int(choice["accepted_rank"])
    colours = {1: "#F2B134", accepted_rank: "#087E8B"}
    figure, (map_axis, evidence_axis) = plt.subplots(
        1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.2, 1]},
    )
    nearby = wkb.loads(routes.iloc[0]["neighbouring_links_wkb"])
    gpd.GeoSeries([nearby], crs="EPSG:4326").plot(
        ax=map_axis, color="#CBDADD", linewidth=0.7, alpha=0.8,
    )
    geometries = []
    for route in routes.itertuples():
        geometry = wkb.loads(route.route_wkb)
        geometries.append(geometry)
        colour = colours[int(route.match_rank)]
        gpd.GeoSeries([geometry], crs="EPSG:4326").plot(
            ax=map_axis, color=colour, linewidth=4.5, zorder=4,
        )
        map_axis.scatter(
            route.wq_longitude, route.wq_latitude, s=120, color=colour,
            edgecolor="white", linewidth=1, zorder=7,
        )
        map_axis.text(
            route.wq_longitude, route.wq_latitude, str(int(route.match_rank)),
            ha="center", va="center", color="white", weight="bold", zorder=8,
        )
    biology = Point(routes.iloc[0]["biology_longitude"], routes.iloc[0]["biology_latitude"])
    map_axis.scatter(
        biology.x, biology.y, marker="*", s=210, color="#D95D39",
        edgecolor="white", linewidth=1, zorder=8,
    )
    bounds = [geometry.bounds for geometry in geometries]
    min_x = min(item[0] for item in bounds); max_x = max(item[2] for item in bounds)
    min_y = min(item[1] for item in bounds); max_y = max(item[3] for item in bounds)
    dx = max(max_x - min_x, 0.01); dy = max(max_y - min_y, 0.01)
    map_axis.set_xlim(min_x - 0.15 * dx, max_x + 0.15 * dx)
    map_axis.set_ylim(min_y - 0.15 * dy, max_y + 0.15 * dy)
    map_axis.set_axis_off()
    map_axis.set_title("A. Two connected candidates", loc="left")

    evidence_axis.set(xlim=(0, 1), ylim=(0, 1))
    evidence_axis.axis("off")
    evidence_axis.set_title("B. Ranking and acceptance are separate steps", loc="left")
    for index, row in enumerate(candidates.itertuples(index=False)):
        y_value = 0.72 - index * 0.43
        accepted = row.candidate_gate_decision == "accept"
        colour = "#087E8B" if accepted else "#F2B134"
        evidence_axis.add_patch(FancyBboxPatch(
            (0.03, y_value - 0.16), 0.93, 0.31,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="#F6F8FA", edgecolor=colour, linewidth=2,
        ))
        outcome = "USED" if accepted else "HELD FOR REVIEW"
        reason = "All acceptance checks pass" if accepted else str(row.candidate_review_reasons).replace("_", " ")
        evidence_axis.text(
            0.08, y_value + 0.075,
            f"Candidate {int(row.match_rank)}  {outcome}",
            weight="bold", color=colour, fontsize=10,
        )
        evidence_axis.text(
            0.08, y_value - 0.01,
            f"Score {row.match_score:.1f} · {row.network_distance_m / 1000:.2f} km river-network distance",
            color="#17324D", fontsize=9,
        )
        evidence_axis.text(0.08, y_value - 0.09, reason[:68], color="#657786", fontsize=8)
    figure.suptitle(
        "Worked example of ranking and acceptance",
        x=0.01, ha="left", weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure, candidates


def compare_matching_recipes(
    ranked_candidates: pd.DataFrame,
    total_site_years: int,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Compare coverage and station stability under alternative matching rules."""

    definitions = {
        "Default": [],
        "Metals": ["wq_has_metals:eq:true"],
        "Upstream": ["network_direction:eq:wq_upstream_of_bio"],
        "≤5 km": ["network_distance_m:le:5000"],
        "Richer chemistry": ["wq_n_sampling_dates:ge:6", "wq_n_target_determinands:ge:5"],
        "No major tributary": ["major_tributary_between_sites:eq:false"],
    }

    selections: dict[str, pd.DataFrame] = {}
    for label, rules in definitions.items():
        candidates, _ = apply_table_filters(ranked_candidates, rules, label=label)
        candidates = candidates.sort_values(
            ["bio_site_id", "year", "match_score", "network_distance_m", "wq_site_id"],
            ascending=[True, True, False, True, True],
        ).copy()
        candidates["match_rank"] = candidates.groupby(["bio_site_id", "year"]).cumcount() + 1
        selections[label] = select_candidates_by_scientific_gate(candidates).loc[
            lambda frame: frame["candidate_gate_decision"].eq("accept")
        ].drop_duplicates(["bio_site_id", "year"])

    default = selections["Default"][["bio_site_id", "year", "wq_site_id"]].rename(
        columns={"wq_site_id": "default_wq_site_id"}
    )
    rows = []
    for label, accepted in selections.items():
        comparison = accepted[["bio_site_id", "year", "wq_site_id"]].merge(
            default, on=["bio_site_id", "year"], how="inner",
        )
        rows.append({
            "recipe": label,
            "accepted site-years": len(accepted),
            "accepted coverage (%)": 100 * len(accepted) / max(total_site_years, 1),
            "retained from default (%)": 100 * len(comparison) / max(len(default), 1),
            "same station among shared (%)": (
                100 * comparison["wq_site_id"].eq(comparison["default_wq_site_id"]).mean()
                if len(comparison) else np.nan
            ),
            "station switches": int(
                comparison["wq_site_id"].ne(comparison["default_wq_site_id"]).sum()
            ),
        })
    summary = pd.DataFrame(rows)

    figure, (coverage_axis, stability_axis) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    positions = np.arange(len(summary))
    coverage_axis.barh(positions, summary["accepted coverage (%)"], color="#17324D")
    coverage_axis.set_yticks(positions, summary["recipe"])
    coverage_axis.invert_yaxis()
    coverage_axis.set(xlabel="Target site-years accepted (%)", xlim=(0, 100))
    coverage_axis.set_title("A. Each question changes usable coverage", loc="left")
    for position, value in zip(positions, summary["accepted coverage (%)"]):
        coverage_axis.text(value + 1, position, f"{value:.1f}%", va="center", fontsize=8)

    near_default = (
        summary["retained from default (%)"].gt(95)
        & summary["same station among shared (%)"].gt(99.5)
    )
    stability_groups = summary.loc[~near_default].rename(columns={
        "retained from default (%)": "x",
        "same station among shared (%)": "y",
        "accepted site-years": "accepted_site_years",
        "station switches": "station_switches",
    })[["recipe", "x", "y", "accepted_site_years", "station_switches"]]
    if near_default.any():
        cluster = summary.loc[near_default]
        clustered_label = " / ".join(cluster["recipe"])
        if len(clustered_label) > 28:
            clustered_label = clustered_label.replace(" / No major", " /\nNo major")
        stability_groups = pd.concat([
            stability_groups,
            pd.DataFrame([{
                "recipe": clustered_label,
                "x": cluster["retained from default (%)"].mean(),
                "y": cluster["same station among shared (%)"].mean(),
                "accepted_site_years": cluster["accepted site-years"].max(),
                "station_switches": cluster["station switches"].max(),
            }]),
        ], ignore_index=True)
    stability_axis.scatter(
        stability_groups["x"], stability_groups["y"],
        s=70 + np.sqrt(stability_groups["accepted_site_years"]) * 2,
        c=stability_groups["station_switches"], cmap="viridis", edgecolor="white",
    )
    for _, row in stability_groups.iterrows():
        top_right = row["x"] > 95 and row["y"] > 99.5
        stability_axis.annotate(
            row["recipe"],
            (row["x"], row["y"]),
            xytext=(-7, 8) if top_right else (4, 5),
            textcoords="offset points", fontsize=8,
            ha="right" if top_right else "left",
        )
    stability_axis.set(
        xlabel="Default accepted site-years retained (%)",
        ylabel="Same station among shared site-years (%)",
        xlim=(0, 103), ylim=(94.5, 100.7),
    )
    stability_axis.set_title("B. Retained coverage and station choice", loc="left")
    for axis in (coverage_axis, stability_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.16)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Effect of alternative matching rules",
        x=0.01, ha="left", fontsize=16, weight="bold",
    )
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.82, wspace=0.23)
    _save_figure(figure, output_directory, "matching_recipe_sensitivity.png")
    return figure, summary


def compare_recipe_weights(
    ranked_candidates: pd.DataFrame,
    alternative_recipe: Mapping[str, object],
    *,
    total_site_years: int,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Compare the default 100-point ranking with one alternative allocation."""

    default_candidates = ranked_candidates.sort_values(
        ["bio_site_id", "year", "match_score", "network_distance_m", "wq_site_id"],
        ascending=[True, True, False, True, True],
    ).copy()
    default_candidates["match_rank"] = (
        default_candidates.groupby(["bio_site_id", "year"]).cumcount() + 1
    )
    default_selected = select_candidates_by_scientific_gate(default_candidates).loc[
        lambda frame: frame["candidate_gate_decision"].eq("accept")
    ].copy()

    alternative_scored = apply_custom_matching_recipe(
        ranked_candidates, alternative_recipe
    )
    alternative_selected = select_custom_matches(alternative_scored)

    default_view = default_selected[[
        "bio_site_id", "year", "wq_site_id", "network_distance_m",
        "score_chemistry",
    ]].rename(columns={
        "wq_site_id": "default station",
        "network_distance_m": "default distance (m)",
        "score_chemistry": "default chemistry component",
    })
    alternative_view = alternative_selected[[
        "bio_site_id", "year", "wq_site_id", "network_distance_m",
        "score_chemistry", "custom_score",
    ]].rename(columns={
        "wq_site_id": "alternative station",
        "network_distance_m": "alternative distance (m)",
        "score_chemistry": "alternative chemistry component",
        "custom_score": "alternative score",
    })
    comparison = default_view.merge(
        alternative_view, on=["bio_site_id", "year"], how="outer",
        validate="one_to_one",
    )
    shared = comparison["default station"].notna() & comparison["alternative station"].notna()
    changed = shared & comparison["default station"].ne(comparison["alternative station"])
    same = shared & ~changed

    recipe_name = str(alternative_recipe["name"])
    default_chemistry_support = (
        pd.to_numeric(default_selected["score_chemistry"], errors="coerce")
        / BASE_COMPONENT_MAXIMA["score_chemistry"]
        * 100
    )
    alternative_chemistry_support = (
        pd.to_numeric(alternative_selected["score_chemistry"], errors="coerce")
        / BASE_COMPONENT_MAXIMA["score_chemistry"]
        * 100
    )
    summary = pd.DataFrame([
        {
            "recipe": "Default",
            "points": 100,
            "usable site-years": len(default_selected),
            "coverage (%)": 100 * len(default_selected) / max(total_site_years, 1),
            "median river-network distance (km)": pd.to_numeric(
                default_selected["network_distance_m"], errors="coerce"
            ).median() / 1_000,
            "median chemistry support (%)": default_chemistry_support.median(),
            "shared with default": len(default_selected),
            "same station among shared (%)": 100.0,
            "station changes vs default": 0,
        },
        {
            "recipe": recipe_name,
            "points": 100,
            "usable site-years": len(alternative_selected),
            "coverage (%)": 100 * len(alternative_selected) / max(total_site_years, 1),
            "median river-network distance (km)": pd.to_numeric(
                alternative_selected["network_distance_m"], errors="coerce"
            ).median() / 1_000,
            "median chemistry support (%)": alternative_chemistry_support.median(),
            "shared with default": int(shared.sum()),
            "same station among shared (%)": 100 * int(same.sum()) / max(int(shared.sum()), 1),
            "station changes vs default": int(changed.sum()),
        },
    ])

    changed_examples = comparison.loc[changed].copy()
    for column in ("default distance (m)", "alternative distance (m)"):
        changed_examples[column.replace("(m)", "(km)")] = (
            pd.to_numeric(changed_examples[column], errors="coerce") / 1_000
        )
    changed_examples = changed_examples[[
        "bio_site_id", "year", "default station", "alternative station",
        "default distance (km)", "alternative distance (km)",
        "default chemistry component", "alternative chemistry component",
    ]].sort_values(["bio_site_id", "year"])

    allocation = recipe_terms_table(alternative_recipe)
    alternative_points = dict(zip(allocation["term"], allocation["points"]))
    component_order = list(BASE_COMPONENT_MAXIMA)
    additional_evidence_points = allocation.loc[
        allocation["source"].ne("default score component"), "points"
    ].sum()
    if additional_evidence_points:
        component_order.append("additional_evidence")
        alternative_points["additional_evidence"] = float(additional_evidence_points)
    component_labels = {
        "score_distance": "River-network distance",
        "score_snap": "Snapping quality",
        "score_temporal": "Temporal support",
        "score_chemistry": "Chemistry support",
        "score_transport_plausibility": "Transport plausibility",
        "additional_evidence": "Additional evidence",
    }
    colours = ["#17324D", "#4E79A7", "#59A14F", "#F2B134", "#B07AA1", "#E15759"]

    figure, (points_axis, impact_axis) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    left_positions = np.zeros(2)
    for component, colour in zip(component_order, colours):
        values = np.array([
            BASE_COMPONENT_MAXIMA.get(component, 0.0),
            float(alternative_points.get(component, 0.0)),
        ])
        points_axis.barh(
            [0, 1], values, left=left_positions, color=colour,
            label=component_labels[component], height=0.56,
        )
        for row, (left, value) in enumerate(zip(left_positions, values)):
            if value >= 8:
                points_axis.text(
                    left + value / 2, row, f"{value:g}", ha="center", va="center",
                    color="white", fontsize=8, weight="bold",
                )
        left_positions += values
    points_axis.set(
        xlim=(0, 100), xlabel="Points", yticks=[0, 1],
        yticklabels=["Default", recipe_name.replace("_", " ").title()],
    )
    points_axis.invert_yaxis()
    points_axis.set_title("A. Allocation of the 100 points", loc="left")
    points_axis.legend(frameon=False, fontsize=8, loc="lower center", ncol=2,
                       bbox_to_anchor=(0.5, -0.34))

    shared_count = int(shared.sum())
    impact_labels = [
        "Default coverage", "Alternative coverage",
        "Same station\namong shared", "Changed station\namong shared",
    ]
    impact_values = [
        100 * len(default_selected) / max(total_site_years, 1),
        100 * len(alternative_selected) / max(total_site_years, 1),
        100 * int(same.sum()) / max(shared_count, 1),
        100 * int(changed.sum()) / max(shared_count, 1),
    ]
    impact_counts = [
        len(default_selected), len(alternative_selected), int(same.sum()), int(changed.sum()),
    ]
    bars = impact_axis.barh(
        np.arange(4), impact_values,
        color=["#17324D", "#F2B134", "#59A14F", "#D95D39"], height=0.58,
    )
    impact_axis.set(
        xlim=(0, 100), xlabel="Site-years (%)", yticks=np.arange(4),
        yticklabels=impact_labels,
    )
    impact_axis.invert_yaxis()
    impact_axis.set_title("B. Effect on coverage and station choice", loc="left")
    for bar, value, count in zip(bars, impact_values, impact_counts):
        impact_axis.text(
            min(value + 1.0, 93), bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%  (n={count:,})", va="center", fontsize=8,
        )

    for axis in (points_axis, impact_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.16)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Effect of changing the 100-point weights",
        x=0.01, ha="left", fontsize=16, weight="bold",
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.82, wspace=0.34)
    _save_figure(figure, output_directory, "matching_recipe_weight_comparison.png")
    return figure, summary, changed_examples


def plot_matching_outcomes(
    ranked_candidates: pd.DataFrame,
    decisions: pd.DataFrame,
) -> plt.Figure:
    """Summarise national decisions and candidate river-network distances."""

    counts = decisions["match_decision"].fillna("missing").value_counts()
    distances = pd.to_numeric(
        ranked_candidates["network_distance_m"], errors="coerce",
    ) / 1000
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colours = [
        "#087E8B" if label == "accepted"
        else "#F2B134" if label.startswith("provisional")
        else "#657786"
        for label in counts.index
    ]
    axes[0].barh(range(len(counts)), counts.values, color=colours)
    axes[0].set_yticks(
        range(len(counts)), [label.replace("_", " ").title() for label in counts.index],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Target site-years")
    axes[0].set_title("A. Outcomes for all target site-years", loc="left")
    axes[1].hist(distances.dropna(), bins=35, color="#087E8B", alpha=0.9)
    axes[1].set(xlabel="Candidate river-network distance (km)", ylabel="Candidate routes")
    axes[1].set_title("B. Candidate river-network distances", loc="left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.16)
        axis.set_axisbelow(True)
    figure.suptitle(
        "England-wide matching outcomes",
        x=0.01, ha="left", weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def plot_temporal_join(window_months: int) -> plt.Figure:
    """Illustrate the target-specific antecedent chemistry window."""

    figure, axis = plt.subplots(figsize=(13, 3.9))
    axis.set(xlim=(-window_months - 2.0, 2.0), ylim=(-0.15, 2.65))
    axis.set_yticks([])
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0.72))
    axis.set_xlabel("Months relative to the biological survey")
    axis.axvspan(-window_months, 0, ymin=0.23, ymax=0.78, color="#5BC0BE", alpha=0.22)
    axis.axvline(-window_months, ymin=0.22, ymax=0.82, color="#087E8B", lw=1.8, linestyle="--")
    axis.axvline(0, ymin=0.18, ymax=0.88, color="#D95D39", lw=2.2)
    included = np.array([-0.86, -0.56, -0.22]) * window_months
    axis.scatter(included, np.full(3, 1.35), s=105, color="#087E8B", edgecolor="white", linewidth=1.2)
    axis.scatter([-window_months - 1.15], [1.35], s=90, color="#9AA9B0", edgecolor="white", linewidth=1.2)
    axis.scatter([1.0], [1.35], s=100, color="#D95D39", marker="x", linewidth=2.4)
    axis.text(-window_months / 2, 2.18, f"Included {window_months}-month chemistry window", ha="center", color="#087E8B", weight="bold")
    axis.text(-window_months - 1.15, 1.70, "Too early", ha="center", color="#657786", fontsize=8.5)
    axis.text(1.0, 1.70, "Future sample\nexcluded", ha="center", color="#D95D39", fontsize=8.5)
    axis.text(-window_months / 2, 0.93, "Included samples → dates, breadth and summary statistics", ha="center", color="#17324D", fontsize=8.5)
    ticks = sorted({-window_months, -window_months / 2, 0, 1})
    axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
    axis.set_title(
        "Chemistry window for each biological survey",
        loc="left", pad=12, weight="bold",
    )
    return figure


# Part 2: evidence coverage and geography


# --- Dataset coverage figures --------------------------------------------
# Coverage plots describe what made it into the joined data before any response
# analysis is attempted.
def plot_dataset_overview(
    specialist_path: str | Path,
    data_dictionary: pd.DataFrame,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Summarise the fields, biological programmes and matching outcomes."""

    group_order = [
        "six-month chemistry", "matching and interpretation", "biology",
        "NRFA flow", "site and source details", "wastewater context",
        "evidence quality", "WFD context",
    ]
    group_labels = {
        "six-month chemistry": "Antecedent chemistry",
        "matching and interpretation": "Network match",
        "biology": "Biology",
        "NRFA flow": "NRFA flow",
        "site and source details": "Site and source details",
        "wastewater context": "Wastewater context",
        "evidence quality": "Evidence quality",
        "WFD context": "WFD context",
    }
    colours = ["#3A86A8", "#735DA5", "#3A7D44", "#4C78A8", "#6C757D", "#C56A2D", "#D4A72C", "#7A9E3A"]
    dictionary_groups = data_dictionary["group"].replace(
        {"identity and provenance": "site and source details"}
    )
    field_counts = dictionary_groups.value_counts().reindex(group_order, fill_value=0)

    columns = [
        "data_origin", "assemblage", "site_id", "sample_year", "match_decision",
        "chemistry_included_in_release", "chemistry_determinand_count_6mo",
        "taxon_richness",
    ]
    records = pd.read_parquet(specialist_path, columns=columns)
    records = records.loc[records["data_origin"].eq("connected_river_pipeline")].copy()
    records["chemistry_included_in_release"] = records[
        "chemistry_included_in_release"
    ].fillna(False).astype(bool)
    decisions = records["match_decision"].fillna("").astype(str)
    records["matching outcome"] = np.select(
        [
            records["chemistry_included_in_release"],
            decisions.str.startswith("provisional"),
            decisions.str.startswith("no_"),
        ],
        ["Chemistry included", "Retained for review", "No defensible match"],
        default="Other excluded",
    )

    assemblage_order = list(ASSEMBLAGE_LABELS)
    disposition_order = [
        "Chemistry included", "Retained for review", "No defensible match", "Other excluded",
    ]
    disposition = pd.crosstab(
        records["assemblage"], records["matching outcome"],
    ).reindex(index=assemblage_order, columns=disposition_order, fill_value=0)
    disposition_percent = disposition.div(disposition.sum(axis=1), axis=0) * 100

    summary_rows = []
    for assemblage in assemblage_order:
        frame = records.loc[records["assemblage"].eq(assemblage)]
        released = frame.loc[frame["chemistry_included_in_release"]]
        years = pd.to_numeric(frame["sample_year"], errors="coerce").dropna()
        summary_rows.append({
            "biological group": ASSEMBLAGE_LABELS[assemblage],
            "survey rows": len(frame),
            "biology sites": frame["site_id"].nunique(dropna=True),
            "year range": f"{int(years.min())}-{int(years.max())}" if len(years) else "not available",
            "chemistry included (%)": 100 * len(released) / max(len(frame), 1),
            "median determinands": released["chemistry_determinand_count_6mo"].median(),
            "rows with taxon richness": frame["taxon_richness"].notna().sum(),
        })
    summary = pd.DataFrame(summary_rows).set_index("biological group")

    figure, (field_axis, outcome_axis) = plt.subplots(1, 2, figsize=(15, 6.2))
    plot_order = list(reversed(group_order))
    field_axis.barh(
        [group_labels[group] for group in plot_order],
        field_counts.reindex(plot_order).values,
        color=list(reversed(colours)), height=0.66,
    )
    field_axis.set_xlabel("Documented fields")
    field_axis.set_title("A. Contents of the finished dataset", loc="left")
    field_axis.grid(axis="x", alpha=0.18)

    left = np.zeros(len(assemblage_order))
    disposition_colours = {
        "Chemistry included": "#2A9D8F", "Retained for review": "#E9C46A",
        "No defensible match": "#E76F51", "Other excluded": "#8D99AE",
    }
    for label in disposition_order:
        values = disposition_percent[label].to_numpy()
        outcome_axis.barh(
            np.arange(len(assemblage_order)), values, left=left,
            color=disposition_colours[label], label=label,
        )
        left += values
    outcome_axis.set_yticks(
        np.arange(len(assemblage_order)),
        [ASSEMBLAGE_LABELS[value] for value in assemblage_order],
    )
    outcome_axis.invert_yaxis()
    outcome_axis.set(xlabel="Share of survey rows (%)", xlim=(0, 100))
    outcome_axis.set_title("B. Matching outcomes by biological group", loc="left")
    outcome_axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    for axis in (field_axis, outcome_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    years = pd.to_numeric(records["sample_year"], errors="coerce").dropna()
    figure.suptitle(
        f"{len(records):,} surveys · {records['site_id'].nunique():,} biological sites · "
        f"{int(years.min())}-{int(years.max())} · {len(data_dictionary):,} documented fields",
        x=0.01, ha="left", fontsize=15, weight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    _save_figure(figure, output_directory, "joined_dataset_overview.png")
    return figure, summary


@dataclass(frozen=True)
# --- Spatial evidence figures --------------------------------------------
# Prepare map layers once and reuse them across the static and interactive maps.
class SpatialEvidence:
    """Prepared hydrometric-area evidence reused by several report figures."""

    assignments: pd.DataFrame
    areas: gpd.GeoDataFrame
    records: pd.DataFrame
    support: gpd.GeoDataFrame
    observed: gpd.GeoDataFrame
    bounds: tuple[float, float, float, float]


def prepare_spatial_evidence(
    release_directory: str | Path,
    specialist_path: str | Path,
) -> SpatialEvidence:
    """Prepare the shared hydrometric-area tables once."""

    release_directory = Path(release_directory)
    assignments = pd.read_parquet(
        release_directory / "hydrological_area_assignments.parquet"
    )
    areas = gpd.read_file(
        release_directory / "hydrological_area_map_geometry.geojson"
    ).to_crs("EPSG:4326")
    areas["map_area_id"] = areas["id"].astype(str)
    areas["area_km2"] = areas.to_crs("EPSG:27700").geometry.area / 1_000_000

    columns = [
        "integration_record_id", "data_origin", "assemblage", "site_id",
        "sample_year", "chemistry_included_in_release",
    ]
    records = pd.read_parquet(specialist_path, columns=columns).merge(
        assignments, on="integration_record_id", how="left",
    )
    records = records.loc[
        records["data_origin"].eq("connected_river_pipeline") & records["map_area_id"].notna()
    ].copy()
    records["map_area_id"] = records["map_area_id"].astype(str)
    records["site key"] = records["assemblage"].astype(str) + "::" + records["site_id"].astype(str)
    records["chemistry included"] = records["chemistry_included_in_release"].fillna(False).astype(bool)

    site_years = records.groupby(["map_area_id", "site key"])["sample_year"].nunique().rename("observed years").reset_index()
    site_support = site_years.groupby("map_area_id").agg(
        biological_sites=("site key", "size"),
        repeated_sites=("observed years", lambda values: values.ge(5).sum()),
        repeated_site_percent=("observed years", lambda values: 100 * values.ge(5).mean()),
    ).reset_index()
    row_support = records.groupby("map_area_id").agg(
        survey_rows=("integration_record_id", "size"),
        chemistry_release_percent=("chemistry included", lambda values: 100 * values.mean()),
    ).reset_index()
    support = areas.merge(site_support, on="map_area_id", how="left").merge(
        row_support, on="map_area_id", how="left",
    )
    support["sites per 1,000 km²"] = 1_000 * support["biological_sites"] / support["area_km2"]
    observed = support.loc[support["map_area_id"].isin(set(records["map_area_id"]))].copy()
    min_x, min_y, max_x, max_y = observed.total_bounds
    x_pad, y_pad = (max_x - min_x) * 0.04, (max_y - min_y) * 0.04
    return SpatialEvidence(
        assignments=assignments,
        areas=areas,
        records=records,
        support=support,
        observed=observed,
        bounds=(min_x - x_pad, min_y - y_pad, max_x + x_pad, max_y + y_pad),
    )


def plot_spatial_support(
    spatial: SpatialEvidence,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Map monitoring density, repeat coverage and included chemistry."""

    specifications = [
        ("sites per 1,000 km²", "A. Biological-site density", "YlGnBu", "Sites per 1,000 km²"),
        ("repeated_site_percent", "B. Longitudinal monitoring support", "PuBu", "Sites observed in at least five years (%)"),
        ("chemistry_release_percent", "C. Chemistry included", "YlGn", "Survey rows with chemistry included (%)"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 6.5))
    min_x, min_y, max_x, max_y = spatial.bounds
    for axis, (column, title, colour_map, legend_label) in zip(axes, specifications):
        spatial.areas.plot(ax=axis, color="#E4E8EA", edgecolor="white", linewidth=0.35)
        spatial.observed.plot(
            ax=axis, column=column, cmap=colour_map, legend=True,
            edgecolor="white", linewidth=0.4,
            legend_kwds={"label": legend_label, "shrink": 0.68},
        )
        axis.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
        axis.set_axis_off()
        axis.set_title(title, loc="left", fontsize=11)
    figure.suptitle(
        "Monitoring and matched chemistry by hydrometric area",
        x=0.01, ha="left", fontsize=15, weight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save_figure(figure, output_directory, "spatial_monitoring_and_match_support.png")
    table = spatial.observed[[
        "catchment_name", "biological_sites", "sites per 1,000 km²",
        "repeated_site_percent", "chemistry_release_percent",
    ]].copy()
    table.columns = [
        "hydrometric area", "biological sites", "sites per 1,000 km²",
        "sites observed in at least five years (%)", "chemistry included rows (%)",
    ]
    return figure, table


def plot_river_stories(
    specialist_path: str | Path,
    release_directory: str | Path,
    spatial: SpatialEvidence,
    output_directory: str | Path | None = None,
) -> plt.Figure:
    """Contrast three reproducibly selected site histories and their retained routes."""

    columns = [
        "data_origin", "assemblage", "site_id", "bio_site_id", "sample_year",
        "taxon_richness", "relative_nutrient_pressure", "chemistry_included_in_release",
        "match_decision", "wq_site_id", "longitude", "latitude",
    ]
    records = pd.read_parquet(specialist_path, columns=columns)
    records = records.loc[records["data_origin"].eq("connected_river_pipeline")].copy()
    decisions = records["match_decision"].fillna("").astype(str)
    records["status"] = np.select(
        [
            records["chemistry_included_in_release"].fillna(False),
            decisions.str.startswith("provisional"),
            decisions.str.startswith("no_"),
        ],
        ["Chemistry included", "Review", "No match"],
        default="Other",
    )
    site_year = records.groupby(
        ["assemblage", "site_id", "bio_site_id", "sample_year"],
        observed=True, as_index=False,
    ).agg(
        taxon_richness=("taxon_richness", "median"),
        nutrient_pressure=("relative_nutrient_pressure", "median"),
        chemistry_released=("chemistry_included_in_release", "max"),
        status=("status", lambda values: values.value_counts().index[0]),
        wq_site_id=("wq_site_id", lambda values: values.dropna().iloc[0] if values.notna().any() else pd.NA),
        longitude=("longitude", "median"),
        latitude=("latitude", "median"),
    )
    site_summary = site_year.groupby(
        ["assemblage", "site_id", "bio_site_id"], observed=True,
    ).agg(
        years=("sample_year", "nunique"),
        release_fraction=("chemistry_released", "mean"),
        pressure_years=("nutrient_pressure", "count"),
        status_count=("status", "nunique"),
    ).reset_index()
    specifications = [
        ("Consistent evidence", "macroinvertebrate", site_summary["years"].ge(6) & site_summary["release_fraction"].ge(0.8) & site_summary["pressure_years"].ge(4)),
        ("Mixed evidence", "diatom", site_summary["years"].ge(6) & site_summary["release_fraction"].between(0.25, 0.75)),
        ("Persistent chemistry gap", "freshwater_fish", site_summary["years"].ge(6) & site_summary["release_fraction"].eq(0)),
    ]
    choices = []
    for label, preferred_group, criterion in specifications:
        candidates = site_summary.loc[
            criterion & site_summary["assemblage"].eq(preferred_group)
        ].sort_values(
            ["years", "pressure_years", "status_count", "bio_site_id"],
            ascending=[False, False, False, True],
        )
        if candidates.empty:
            candidates = site_summary.loc[criterion].sort_values(
                ["years", "pressure_years", "status_count", "bio_site_id"],
                ascending=[False, False, False, True],
            )
        if candidates.empty:
            raise RuntimeError(f"No example is available for {label}")
        selected = candidates.iloc[0].copy()
        selected["story_label"] = label
        choices.append(selected)
    choices = pd.DataFrame(choices)

    status_colours = {
        "Chemistry included": "#138A72", "Review": "#E9B949",
        "No match": "#D95D39", "Other": "#8D99AE",
    }
    figure = plt.figure(figsize=(16, 9.8))
    grid = figure.add_gridspec(2, 3, height_ratios=[0.82, 1.18], hspace=0.30, wspace=0.28)
    release_directory = Path(release_directory)
    for column_index, choice in enumerate(choices.itertuples(index=False)):
        frame = site_year.loc[site_year["bio_site_id"].eq(choice.bio_site_id)].sort_values("sample_year")
        map_axis = figure.add_subplot(grid[0, column_index])
        time_axis = figure.add_subplot(grid[1, column_index])
        site_lon = float(frame["longitude"].median())
        site_lat = float(frame["latitude"].median())
        spatial.areas.plot(ax=map_axis, color="#EDF1F2", edgecolor="white", linewidth=0.35)
        renderable_year = None
        route_status = "Other"
        route_frame = pd.DataFrame()
        for year in sorted(frame["sample_year"].astype(int).unique(), reverse=True):
            candidates = pd.read_parquet(
                release_directory / "river_connection_routes.parquet",
                filters=[("bio_site_id", "==", choice.bio_site_id), ("year", "==", int(year))],
            ).dropna(subset=["route_wkb", "neighbouring_links_wkb"])
            if not candidates.empty:
                record = frame.loc[frame["sample_year"].eq(year)].iloc[0]
                preferred = candidates.loc[candidates["wq_site_id"].eq(record["wq_site_id"])]
                route_frame = preferred if not preferred.empty else candidates.head(1)
                renderable_year = int(year)
                route_status = str(record["status"])
                break
        if not route_frame.empty:
            route = route_frame.iloc[0]
            nearby = wkb.loads(route["neighbouring_links_wkb"])
            geometry = wkb.loads(route["route_wkb"])
            gpd.GeoSeries([nearby], crs="EPSG:4326").plot(ax=map_axis, color="#C8D5D8", linewidth=0.8, zorder=2)
            gpd.GeoSeries([geometry], crs="EPSG:4326").plot(ax=map_axis, color=status_colours[route_status], linewidth=4, zorder=4)
            map_axis.scatter(route["wq_longitude"], route["wq_latitude"], s=90, color=status_colours[route_status], edgecolor="white", zorder=6)
            min_x, min_y, max_x, max_y = geometry.bounds
            dx = max(max_x - min_x, 0.02); dy = max(max_y - min_y, 0.02)
            map_axis.set_xlim(min_x - 0.30 * dx, max_x + 0.30 * dx)
            map_axis.set_ylim(min_y - 0.30 * dy, max_y + 0.30 * dy)
        else:
            map_axis.set_xlim(site_lon - 0.16, site_lon + 0.16)
            map_axis.set_ylim(site_lat - 0.13, site_lat + 0.13)
        map_axis.scatter(site_lon, site_lat, marker="*", s=190, color="#17324D", edgecolor="white", linewidth=0.9, zorder=7)
        map_axis.set_axis_off()
        group_label = choice.assemblage.replace("_", " ").title()
        route_note = f"route shown: {renderable_year}" if renderable_year else "no retained route"
        map_axis.set_title(f"{choice.story_label}\n{group_label} · {route_note}", loc="left", fontsize=11, weight="bold")

        time_axis.plot(frame["sample_year"], frame["taxon_richness"], color="#526876", linewidth=1.6, zorder=1)
        for status, colour in status_colours.items():
            points = frame.loc[frame["status"].eq(status)]
            time_axis.scatter(points["sample_year"], points["taxon_richness"], color=colour, s=55, edgecolor="white", linewidth=0.7, zorder=3)
        pressure_axis = time_axis.twinx()
        pressure_axis.plot(frame["sample_year"], frame["nutrient_pressure"], color="#735DA5", linestyle="--", marker="s", markersize=4, linewidth=1.2, alpha=0.75)
        pressure_axis.set_ylim(-0.05, 1.05)
        pressure_axis.set_ylabel("Relative nutrient pressure" if column_index == 2 else "", color="#735DA5", fontsize=8)
        pressure_axis.tick_params(axis="y", labelsize=7, colors="#735DA5", labelright=column_index == 2)
        time_axis.set(xlabel="Year", ylabel="Taxon richness" if column_index == 0 else "", xticks=sorted(frame["sample_year"].astype(int).unique()))
        time_axis.tick_params(axis="x", rotation=45, labelsize=8)
        time_axis.grid(axis="y", alpha=0.15)
        time_axis.spines[["top"]].set_visible(False)
        pressure_axis.spines[["top"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colour, markeredgecolor="white", markersize=7, label=label)
        for label, colour in status_colours.items()
    ] + [Line2D([0], [0], color="#735DA5", linestyle="--", marker="s", label="Relative nutrient pressure")]
    figure.legend(handles=handles, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.005), fontsize=8.5)
    figure.suptitle(
        "Three examples from the joined dataset",
        x=0.01, y=0.985, ha="left", fontsize=15.5, weight="bold",
    )
    figure.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.12)
    _save_figure(figure, output_directory, "three_catchment_storyboards.png")
    return figure


def plot_evidence_fingerprint(
    specialist_path: str | Path,
    spatial: SpatialEvidence,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Compare the breadth of integrated evidence between hydrometric areas."""

    columns = [
        "integration_record_id", "data_origin", "assemblage", "site_id",
        "sample_year", "chemistry_included_in_release",
        "chemistry_determinand_count_6mo", "flow_evidence_available",
        "wfd_waterbody_id", "path_edm_assets",
    ]
    records = pd.read_parquet(specialist_path, columns=columns).merge(
        spatial.assignments, on="integration_record_id", how="left",
    )
    records = records.loc[
        records["data_origin"].eq("connected_river_pipeline") & records["map_area_id"].notna()
    ].copy()
    records["map_area_id"] = records["map_area_id"].astype(str)
    records["site key"] = records["assemblage"].astype(str) + "::" + records["site_id"].astype(str)
    records["chemistry included"] = records["chemistry_included_in_release"].fillna(False).astype(bool)
    records["flow available"] = records["flow_evidence_available"].fillna(False).astype(bool)
    records["WFD context available"] = records["wfd_waterbody_id"].notna()
    records["wastewater context available"] = records["path_edm_assets"].notna()
    records["included chemistry breadth"] = pd.to_numeric(
        records["chemistry_determinand_count_6mo"], errors="coerce",
    ).where(records["chemistry included"])
    names = spatial.areas[["map_area_id", "catchment_name"]].drop_duplicates("map_area_id")
    rows = []
    for area_id, frame in records.groupby("map_area_id"):
        site_years = frame.groupby("site key")["sample_year"].nunique()
        rows.append({
            "map_area_id": area_id,
            "assemblage breadth": 100 * frame["assemblage"].nunique() / 5,
            "repeat-site support": 100 * site_years.ge(5).mean(),
            "chemistry included": 100 * frame["chemistry included"].mean(),
            "chemistry breadth": 100 * frame["included chemistry breadth"].median() / 41,
            "flow available": 100 * frame["flow available"].mean(),
            "WFD context": 100 * frame["WFD context available"].mean(),
            "wastewater context": 100 * frame["wastewater context available"].mean(),
            "survey rows": len(frame),
        })
    fingerprints = pd.DataFrame(rows).merge(names, on="map_area_id", how="left", validate="one_to_one")
    metrics = [
        "assemblage breadth", "repeat-site support", "chemistry included",
        "chemistry breadth", "flow available", "WFD context", "wastewater context",
    ]
    fingerprints[metrics] = fingerprints[metrics].clip(0, 100)
    fingerprints = fingerprints.sort_values(
        ["chemistry included", "repeat-site support", "survey rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    matrix_columns = metrics
    matrix = fingerprints[matrix_columns].to_numpy(float)
    figure, axis = plt.subplots(figsize=(13.5, max(10, 0.27 * len(fingerprints))))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(matrix_columns)), [
        "Assemblages\nrepresented", "Sites in ≥5\nyears", "Chemistry\nincluded",
        "Median target\nbreadth", "Flow\navailable", "WFD\ncontext",
        "Wastewater\ncontext",
    ])
    axis.set_yticks(range(len(fingerprints)), fingerprints["catchment_name"].fillna(fingerprints["map_area_id"]), fontsize=7.5)
    axis.tick_params(axis="x", labelsize=9, top=True, labeltop=True, bottom=False, labelbottom=False)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(column_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=5.8, color="white" if value < 35 or value > 78 else "#17324D")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Evidence support (%)")
    axis.set_title(
        "Evidence coverage by hydrometric area",
        loc="left", fontsize=15, weight="bold", pad=24,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save_figure(figure, output_directory, "hydrometric_area_evidence_fingerprints.png")
    return figure, fingerprints


def plot_chemistry_coverage(
    specialist_path: str | Path,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Show target-determinand availability within included survey windows."""

    registry = pd.DataFrame(determinand_registry_records())
    count_columns = {
        row.feature_key: f"target_exact_{row.feature_key}_observation_count_6mo"
        for row in registry.itertuples()
    }
    columns = [
        "data_origin", "sample_year", "chemistry_included_in_release",
        *count_columns.values(),
    ]
    records = pd.read_parquet(specialist_path, columns=columns)
    records = records.loc[
        records["data_origin"].eq("connected_river_pipeline")
        & records["chemistry_included_in_release"].fillna(False)
    ].copy()
    years = sorted(pd.to_numeric(records["sample_year"], errors="coerce").dropna().astype(int).unique())
    rows = []
    for registry_row in registry.itertuples():
        counts = pd.to_numeric(records[count_columns[registry_row.feature_key]], errors="coerce").fillna(0)
        row = {"label": registry_row.determinand_label}
        for year in years:
            in_year = pd.to_numeric(records["sample_year"], errors="coerce").eq(year)
            row[year] = 100 * counts[in_year].gt(0).mean() if in_year.any() else np.nan
        rows.append(row)
    coverage = pd.DataFrame(rows).set_index("label")
    coverage["mean coverage"] = coverage.mean(axis=1)
    coverage = coverage.sort_values("mean coverage", ascending=False)
    plot_data = coverage[years]

    figure, axis = plt.subplots(figsize=(12.5, 13.5))
    image = axis.imshow(plot_data.to_numpy(), cmap="magma", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(years)), years)
    axis.set_yticks(range(len(plot_data)), plot_data.index, fontsize=7.5)
    for row_index in range(plot_data.shape[0]):
        for column_index in range(plot_data.shape[1]):
            value = plot_data.iat[row_index, column_index]
            if pd.notna(value) and value >= 1:
                axis.text(
                    column_index, row_index, f"{value:.0f}", ha="center", va="center",
                    fontsize=5.4, color="white" if value < 55 else "#261447",
                )
    figure.colorbar(
        image, ax=axis, fraction=0.028, pad=0.02,
        label="Included survey windows with ≥1 observation (%)",
    )
    axis.set(xlabel="Biological survey year", ylabel="")
    axis.set_title(
        "Target-chemistry coverage by determinand and year",
        loc="left", fontsize=15, weight="bold",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save_figure(figure, output_directory, "target_determinand_year_coverage.png")
    return figure, coverage


def _geometry_paths(geometry: object) -> list[list[list[float]]]:
    """Convert line geometry to pydeck path coordinates."""

    if isinstance(geometry, LineString):
        return [[[float(x), float(y)] for x, y in geometry.coords]]
    if isinstance(geometry, MultiLineString):
        return [
            [[float(x), float(y)] for x, y in line.coords]
            for line in geometry.geoms
        ]
    return []


def build_interactive_monitoring_map(
    specialist_path: str | Path,
    release_directory: str | Path,
    output_directory: str | Path,
    *,
    year: int,
) -> tuple[object, Path, pd.DataFrame]:
    """Build a zoomable monitoring map with saved river routes behind the sites."""

    import pydeck as pdk

    columns = [
        "data_origin", "assemblage", "site_id", "site_label", "sample_year",
        "longitude", "latitude", "match_decision", "chemistry_included_in_release",
    ]
    records = pd.read_parquet(specialist_path, columns=columns)
    records = records.loc[
        records["data_origin"].eq("connected_river_pipeline")
        & records["sample_year"].eq(year)
    ].copy()
    decisions = records["match_decision"].fillna("").astype(str)
    records["status"] = np.select(
        [
            records["chemistry_included_in_release"].fillna(False),
            decisions.str.startswith("provisional"),
            decisions.str.startswith("no_"),
        ],
        ["Chemistry included", "Retained for review", "No defensible match"],
        default="Other excluded",
    )
    priority = {
        "Chemistry included": 0, "Retained for review": 1,
        "No defensible match": 2, "Other excluded": 3,
    }
    colours = {
        "Chemistry included": [19, 138, 114, 175],
        "Retained for review": [233, 185, 73, 175],
        "No defensible match": [217, 93, 57, 170],
        "Other excluded": [141, 153, 174, 150],
    }
    records["status priority"] = records["status"].map(priority)
    records = records.sort_values("status priority").drop_duplicates(
        ["assemblage", "site_id"]
    )
    records["colour"] = records["status"].map(colours)
    records["assemblage label"] = records["assemblage"].map(ASSEMBLAGE_LABELS)
    records["site label"] = records["site_label"].fillna(records["site_id"])

    topology_path = (
        Path(release_directory).parent
        / "data/network_integration/network/irn_edges.parquet"
    )
    river_paths: list[dict[str, object]] = []
    if topology_path.is_file():
        from pyproj import Transformer

        edges = pd.read_parquet(
            topology_path,
            columns=["STRAHLER", "from_x", "from_y", "to_x", "to_y"],
            filters=[("STRAHLER", ">=", 5)],
        ).dropna()
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        from_lon, from_lat = transformer.transform(edges["from_x"], edges["from_y"])
        to_lon, to_lat = transformer.transform(edges["to_x"], edges["to_y"])
        river_paths = [
            {"path": [[float(x1), float(y1)], [float(x2), float(y2)]]}
            for x1, y1, x2, y2 in zip(from_lon, from_lat, to_lon, to_lat)
        ]
    else:
        routes = pd.read_parquet(
            Path(release_directory) / "river_connection_routes.parquet",
            columns=["year", "match_rank", "neighbouring_links_wkb"],
            filters=[("year", "==", int(year))],
        )
        routes = routes.loc[routes["match_rank"].eq(1)].dropna(
            subset=["neighbouring_links_wkb"]
        ).drop_duplicates("neighbouring_links_wkb").head(800)
        for value in routes["neighbouring_links_wkb"]:
            geometry = wkb.loads(value).simplify(0.0004, preserve_topology=False)
            river_paths.extend({"path": path} for path in _geometry_paths(geometry))

    layers = [
        pdk.Layer(
            "PathLayer", river_paths, get_path="path",
            get_color=[102, 145, 158, 95], get_width=18,
            width_min_pixels=0.45, pickable=False,
        ),
        pdk.Layer(
            "ScatterplotLayer", records,
            get_position="[longitude, latitude]", get_fill_color="colour",
            get_radius=85, radius_min_pixels=2.2, radius_max_pixels=8,
            stroked=True, get_line_color=[255, 255, 255, 210],
            line_width_min_pixels=0.5, pickable=True,
        ),
    ]
    view = pdk.ViewState(
        longitude=float(records["longitude"].median()),
        latitude=float(records["latitude"].median()), zoom=5.35,
    )
    deck = pdk.Deck(
        layers=layers, initial_view_state=view, map_style=None,
        tooltip={
            "html": (
                "<b>{site label}</b><br>{assemblage label}<br>"
                "{status}<br>Site ID: {site_id}"
            ),
            "style": {"backgroundColor": "#17324D", "color": "white"},
        },
    )
    destination = Path(output_directory) / f"interactive_monitoring_network_{year}.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(destination), notebook_display=False, open_browser=False)
    return deck, destination, records


def create_monitoring_animation(
    specialist_path: str | Path,
    spatial: SpatialEvidence,
    output_directory: str | Path,
) -> Path:
    """Create the annual map animation and return its path."""

    columns = [
        "data_origin", "assemblage", "site_id", "sample_year", "easting",
        "northing", "match_decision", "chemistry_included_in_release",
    ]
    records = pd.read_parquet(specialist_path, columns=columns)
    records = records.loc[records["data_origin"].eq("connected_river_pipeline")].copy()
    decisions = records["match_decision"].fillna("").astype(str)
    records["display status"] = np.select(
        [
            records["chemistry_included_in_release"].fillna(False),
            decisions.str.startswith("provisional"),
            decisions.str.startswith("no_"),
        ],
        ["Chemistry included", "Retained for review", "No defensible match"],
        default="Other excluded",
    )
    priority = {
        "Chemistry included": 0, "Retained for review": 1,
        "No defensible match": 2, "Other excluded": 3,
    }
    records["status priority"] = records["display status"].map(priority)
    records = records.sort_values("status priority").drop_duplicates(
        ["assemblage", "site_id", "sample_year"]
    )
    points = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(records["easting"], records["northing"]),
        crs="EPSG:27700",
    ).to_crs("EPSG:4326")
    points["longitude"] = points.geometry.x
    points["latitude"] = points.geometry.y
    years = sorted(points["sample_year"].dropna().astype(int).unique())
    colours = {
        "Chemistry included": "#138A72", "Retained for review": "#E9B949",
        "No defensible match": "#D95D39", "Other excluded": "#8D99AE",
    }
    min_x, min_y, max_x, max_y = spatial.bounds
    frames = []
    destination = Path(output_directory) / "monitoring_match_status_2018_2025.gif"
    static_destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as frame_directory:
        for frame_number, year in enumerate(years):
            figure, axis = plt.subplots(figsize=(8.2, 9.2), facecolor="white")
            spatial.areas.plot(ax=axis, color="#EDF1F2", edgecolor="white", linewidth=0.35)
            frame = points.loc[points["sample_year"].eq(year)]
            for status in priority:
                selected = frame.loc[frame["display status"].eq(status)]
                axis.scatter(
                    selected["longitude"], selected["latitude"],
                    s=10 if status == "Chemistry included" else 7,
                    color=colours[status], alpha=0.58, linewidths=0,
                    label=f"{status} ({len(selected):,})",
                )
            axis.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
            axis.set_axis_off()
            axis.set_title(f"Joined monitoring network in {year}", loc="left", fontsize=15, weight="bold")
            axis.text(0.01, 0.02, f"{len(frame):,} biological sites shown", transform=axis.transAxes, fontsize=10, color="#263238", weight="bold")
            axis.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8)
            figure.tight_layout(pad=0.6)
            frame_path = Path(frame_directory) / f"frame_{frame_number:02d}.png"
            figure.savefig(frame_path, dpi=115, facecolor="white")
            plt.close(figure)
            with PILImage.open(frame_path) as image:
                if frame_number == len(years) - 1:
                    image.convert("RGB").save(static_destination)
                frames.append(image.convert("RGB").copy())
    frames[0].save(
        destination, format="GIF", save_all=True, append_images=frames[1:],
        duration=900, loop=0, disposal=2,
    )
    return destination


# Part 2: monitoring and environmental relationships


# --- Biological and pressure-response figures ----------------------------
# These plots move from monitoring effort to exploratory associations. They are
# deliberately labelled as associations rather than causal responses.
def plot_biological_monitoring(
    specialist_path: str | Path,
    release_directory: str | Path,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Summarise survey effort and repeat-site richness through time."""

    records = pd.read_parquet(
        specialist_path,
        columns=["data_origin", "assemblage", "site_id", "sample_year", "taxon_richness"],
    )
    coverage = json.loads(
        (Path(release_directory) / "public_temporal_coverage_qa.json").read_text()
    )
    years = coverage["observed_automated_product_years"]
    records = records.loc[records["data_origin"].eq("connected_river_pipeline")].copy()
    records["sample_year"] = pd.to_numeric(records["sample_year"], errors="coerce")
    records["taxon_richness"] = pd.to_numeric(records["taxon_richness"], errors="coerce")
    records = records.dropna(subset=["assemblage", "site_id", "sample_year"])
    records["sample_year"] = records["sample_year"].astype(int)
    records = records.loc[records["sample_year"].between(min(years), max(years))]
    colours = {
        "macroinvertebrate": "#3A7D44", "freshwater_fish": "#E28E2C",
        "diatom": "#4C78A8", "macrophyte": "#72A98F",
        "transitional_coastal_fish": "#9C755F",
    }
    order = list(ASSEMBLAGE_LABELS)
    annual_counts = records.groupby(["sample_year", "assemblage"], observed=True).size().rename("survey rows").reset_index()
    site_year = records.dropna(subset=["taxon_richness"]).groupby(
        ["assemblage", "site_id", "sample_year"], as_index=False,
    )["taxon_richness"].median()
    site_year["observed years"] = site_year.groupby(["assemblage", "site_id"])["sample_year"].transform("nunique")
    repeated = site_year.loc[site_year["observed years"].ge(5)].copy()
    repeated["site mean richness"] = repeated.groupby(["assemblage", "site_id"])["taxon_richness"].transform("mean")
    repeated["change from site mean"] = repeated["taxon_richness"] - repeated["site mean richness"]
    annual_change = repeated.groupby(["assemblage", "sample_year"], observed=True).agg(
        sites=("site_id", "nunique"),
        mean_change=("change from site mean", "mean"),
        lower_quartile=("change from site mean", lambda values: values.quantile(0.25)),
        upper_quartile=("change from site mean", lambda values: values.quantile(0.75)),
    ).reset_index()
    slopes = []
    for (assemblage, site_id), frame in repeated.groupby(["assemblage", "site_id"], observed=True):
        slopes.append({
            "assemblage": assemblage, "site_id": site_id,
            "taxon-richness slope per year": np.polyfit(frame["sample_year"], frame["taxon_richness"], 1)[0],
        })
    slope_table = pd.DataFrame(slopes)
    trend_summary = slope_table.groupby("assemblage").agg(
        repeated_sites=("site_id", "size"),
        median_slope=("taxon-richness slope per year", "median"),
        lower_quartile=("taxon-richness slope per year", lambda values: values.quantile(0.25)),
        upper_quartile=("taxon-richness slope per year", lambda values: values.quantile(0.75)),
    ).reindex(order)
    trend_summary.index = [ASSEMBLAGE_LABELS[value] for value in trend_summary.index]

    figure = plt.figure(figsize=(16, 9.2))
    grid = figure.add_gridspec(2, 5, height_ratios=[1.1, 1], hspace=0.58, wspace=0.38)
    coverage_axis = figure.add_subplot(grid[0, :])
    for assemblage in order:
        frame = annual_counts.loc[annual_counts["assemblage"].eq(assemblage)]
        coverage_axis.plot(frame["sample_year"], frame["survey rows"], marker="o", linewidth=2.2, color=colours[assemblage], label=ASSEMBLAGE_LABELS[assemblage])
    if 2020 in years:
        coverage_axis.axvspan(2019.7, 2020.3, color="#D9D9D9", alpha=0.45, zorder=0)
        coverage_axis.text(
            2020, 0.96, "Reduced monitoring\nin 2020",
            transform=coverage_axis.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color="#526876",
        )
    coverage_axis.set(title="Monitoring effort changes substantially between years", xlabel="Sample year", ylabel="Survey rows", xticks=years)
    coverage_axis.grid(axis="y", alpha=0.18)
    coverage_axis.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    trend_ticks = years[::2]
    if years[-1] not in trend_ticks:
        trend_ticks = [*trend_ticks, years[-1]]
    for position, assemblage in enumerate(order):
        axis = figure.add_subplot(grid[1, position])
        frame = annual_change.loc[annual_change["assemblage"].eq(assemblage)]
        axis.fill_between(frame["sample_year"], frame["lower_quartile"], frame["upper_quartile"], color=colours[assemblage], alpha=0.14)
        axis.plot(frame["sample_year"], frame["mean_change"], marker="o", color=colours[assemblage], linewidth=2.1)
        axis.axhline(0, color="#6C757D", linewidth=0.9)
        count = int(trend_summary.loc[ASSEMBLAGE_LABELS[assemblage], "repeated_sites"])
        axis.set(title=f"{ASSEMBLAGE_LABELS[assemblage]}\n{count:,} repeat sites", xlabel="Year", xticks=trend_ticks)
        if position == 0:
            axis.set_ylabel("Change from site mean (taxa)")
        axis.grid(axis="y", alpha=0.16)
    figure.suptitle("Biological monitoring through time in the joined dataset", fontsize=16, weight="bold", y=0.98)
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.91, hspace=0.62, wspace=0.38)
    _save_figure(figure, output_directory, "joined_data_through_time.png")
    return figure, trend_summary


def plot_biology_pressure_associations(
    specialist_path: str | Path,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Summarise within-site biology-pressure correlations without causal claims."""

    pressures = {
        "relative_nutrient_pressure": "Nutrient pressure",
        "relative_organic_oxygen_stress": "Organic/oxygen stress",
        "relative_metals_pressure": "Metals pressure",
    }
    columns = [
        "data_origin", "assemblage", "site_id", "sample_year", "taxon_richness",
        "chemistry_included_in_release", *pressures,
    ]
    records = pd.read_parquet(specialist_path, columns=columns)
    records = records.loc[
        records["data_origin"].eq("connected_river_pipeline")
        & records["chemistry_included_in_release"].fillna(False)
    ].copy()
    site_year = records.groupby(
        ["assemblage", "site_id", "sample_year"], as_index=False,
    )[["taxon_richness", *pressures]].median()
    correlations = []
    for (assemblage, site_id), frame in site_year.groupby(["assemblage", "site_id"], observed=True):
        for column, label in pressures.items():
            paired = frame[["taxon_richness", column]].dropna()
            if len(paired) < 3 or paired["taxon_richness"].nunique() < 2 or paired[column].nunique() < 2:
                continue
            correlations.append({
                "assemblage": assemblage, "site_id": site_id, "pressure": label,
                "observed years": len(paired),
                "Spearman correlation": paired["taxon_richness"].corr(paired[column], method="spearman"),
            })
    site_correlations = pd.DataFrame(correlations)
    summary = site_correlations.groupby(["assemblage", "pressure"], observed=True).agg(
        sites=("site_id", "nunique"),
        median_correlation=("Spearman correlation", "median"),
        lower_quartile=("Spearman correlation", lambda values: values.quantile(0.25)),
        upper_quartile=("Spearman correlation", lambda values: values.quantile(0.75)),
    ).reset_index()
    order = list(ASSEMBLAGE_LABELS)
    pressure_order = list(pressures.values())
    matrix = summary.pivot(index="assemblage", columns="pressure", values="median_correlation").reindex(index=order, columns=pressure_order)
    support = summary.pivot(index="assemblage", columns="pressure", values="sites").reindex(index=order, columns=pressure_order).fillna(0)

    figure, (heat_axis, distribution_axis) = plt.subplots(1, 2, figsize=(14.5, 6.5), gridspec_kw={"width_ratios": [1.2, 0.8]})
    image = heat_axis.imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    heat_axis.set_xticks(range(len(pressure_order)), pressure_order, rotation=20, ha="right")
    heat_axis.set_yticks(range(len(order)), [ASSEMBLAGE_LABELS[value] for value in order])
    for row_index in range(len(order)):
        for column_index in range(len(pressure_order)):
            sites = int(support.iat[row_index, column_index])
            correlation = matrix.iat[row_index, column_index]
            if sites < 20:
                heat_axis.add_patch(Rectangle((column_index - 0.5, row_index - 0.5), 1, 1, facecolor="#EADDCB", edgecolor="white", hatch="///", alpha=0.86))
            label = "no estimate" if sites == 0 else f"{correlation:+.2f}\n{sites} sites"
            heat_axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8, color="#17324D" if sites < 20 else "black")
    heat_axis.set_title("A. Median within-site correlation", loc="left", fontsize=12, weight="bold")
    box_data = [
        site_correlations.loc[site_correlations["pressure"].eq(label), "Spearman correlation"].dropna().to_numpy()
        for label in pressure_order
    ]
    box_plot = distribution_axis.boxplot(
        box_data, orientation="horizontal", tick_labels=pressure_order,
        patch_artist=True, showfliers=False, widths=0.55,
        medianprops={"color": "#17324D", "linewidth": 2},
    )
    for box, colour in zip(box_plot["boxes"], ["#6BAED6", "#9ECAE1", "#F4A582"]):
        box.set_facecolor(colour); box.set_alpha(0.8)
    distribution_axis.axvline(0, color="#657786", linewidth=1)
    distribution_axis.set(xlim=(-1, 1), xlabel="Within-site Spearman correlation")
    distribution_axis.grid(axis="x", alpha=0.18)
    distribution_axis.set_title("B. Variation between individual sites", loc="left", fontsize=12, weight="bold")
    figure.colorbar(image, ax=heat_axis, fraction=0.045, pad=0.03, label="Median site correlation")
    figure.suptitle("Biology-pressure patterns at repeat sites", x=0.01, ha="left", fontsize=15, weight="bold")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    _save_figure(figure, output_directory, "within_site_pressure_comovement.png")
    return figure, summary, site_correlations


def _binned_response(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    within_site: bool,
    bins: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return site-balanced binned response estimates and bootstrap intervals."""

    data = frame[["site_id", x_column, y_column]].dropna().copy()
    if within_site:
        data = data.groupby("site_id").filter(
            lambda group: len(group) >= 3
            and group[x_column].nunique() > 1
            and group[y_column].nunique() > 1
        )
        data["x"] = data[x_column] - data.groupby("site_id")[x_column].transform("mean")
        data["y"] = data[y_column] - data.groupby("site_id")[y_column].transform("mean")
    else:
        data = data.groupby("site_id", as_index=False)[[x_column, y_column]].median()
        data = data.rename(columns={x_column: "x", y_column: "y"})
    if data.empty:
        return pd.DataFrame(columns=["x", "estimate", "lower", "upper", "sites", "rows"])
    lower_x, upper_x = data["x"].quantile([0.01, 0.99])
    data = data.loc[data["x"].between(lower_x, upper_x)].copy()
    data["bin"] = pd.qcut(data["x"], bins, duplicates="drop")
    if within_site:
        balanced = data.groupby(["site_id", "bin"], observed=True, as_index=False).agg(
            x=("x", "median"), y=("y", "mean"), rows=("y", "size"),
        )
    else:
        balanced = data.assign(rows=1)
    rng = np.random.default_rng(random_seed)
    rows = []
    for interval, group in balanced.groupby("bin", observed=True):
        values = group["y"].to_numpy(float)
        if len(values) > 1:
            draws = rng.choice(values, size=(600, len(values)), replace=True).mean(axis=1)
            lower, upper = np.quantile(draws, [0.025, 0.975])
        else:
            lower = upper = values[0]
        rows.append({
            "bin": str(interval), "x": float(group["x"].median()),
            "estimate": float(values.mean()), "lower": float(lower),
            "upper": float(upper), "sites": int(group["site_id"].nunique()),
            "rows": int(group["rows"].sum()),
        })
    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)


def plot_pressure_response_curves(
    specialist_path: str | Path,
    output_directory: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Contrast spatial pressure gradients with within-site responses."""

    specifications = [
        (
            "macroinvertebrate", "biology_metric_whpt_aspt", "WHPT ASPT",
            "relative_nutrient_pressure", "Nutrient pressure", "#3A7D44",
        ),
        (
            "macrophyte", "biology_metric_rmni", "RMNI",
            "relative_nutrient_pressure", "Nutrient pressure", "#72A98F",
        ),
        (
            "freshwater_fish", "biology_metric_fish_taxa_richness", "Fish taxa richness",
            "relative_nutrient_pressure", "Nutrient pressure", "#E28E2C",
        ),
    ]
    columns = [
        "data_origin", "assemblage", "site_id", "sample_year",
        "chemistry_included_in_release",
        *sorted({item for spec in specifications for item in (spec[1], spec[3])}),
    ]
    records = pd.read_parquet(specialist_path, columns=columns)
    records = records.loc[
        records["data_origin"].eq("connected_river_pipeline")
        & records["chemistry_included_in_release"].fillna(False)
    ].copy()

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[pd.DataFrame] = []
    for column_index, (assemblage, outcome, outcome_label, pressure, pressure_label, colour) in enumerate(specifications):
        frame = records.loc[records["assemblage"].eq(assemblage)].groupby(
            ["site_id", "sample_year"], as_index=False,
        )[[outcome, pressure]].median()
        between = _binned_response(
            frame, x_column=pressure, y_column=outcome,
            within_site=False, random_seed=120 + column_index,
        )
        within = _binned_response(
            frame, x_column=pressure, y_column=outcome,
            within_site=True, random_seed=220 + column_index,
        )
        for design, curve in (("between sites", between), ("within sites", within)):
            labelled = curve.copy()
            labelled["assemblage"] = assemblage
            labelled["outcome"] = outcome_label
            labelled["pressure"] = pressure_label
            labelled["design"] = design
            curve_rows.append(labelled)

        between_data = frame[["site_id", pressure, outcome]].dropna().groupby("site_id")[[pressure, outcome]].median()
        within_data = frame[["site_id", pressure, outcome]].dropna().copy()
        within_data = within_data.groupby("site_id").filter(
            lambda group: len(group) >= 3
            and group[pressure].nunique() > 1
            and group[outcome].nunique() > 1
        )
        within_data["x"] = within_data[pressure] - within_data.groupby("site_id")[pressure].transform("mean")
        within_data["y"] = within_data[outcome] - within_data.groupby("site_id")[outcome].transform("mean")
        summary_rows.append({
            "assemblage": ASSEMBLAGE_LABELS[assemblage],
            "outcome": outcome_label, "pressure": pressure_label,
            "sites in spatial curve": len(between_data),
            "spatial Spearman correlation": between_data[pressure].corr(between_data[outcome], method="spearman"),
            "sites in repeat-site curve": int(within_data["site_id"].nunique()),
            "within-site Spearman correlation": within_data["x"].corr(within_data["y"], method="spearman"),
        })

        for row_index, (axis, curve) in enumerate(zip(axes[:, column_index], (between, within))):
            axis.fill_between(curve["x"], curve["lower"], curve["upper"], color=colour, alpha=0.16)
            axis.plot(curve["x"], curve["estimate"], marker="o", color=colour, linewidth=2.2)
            axis.grid(alpha=0.16)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlabel(
                f"{pressure_label} (site median)" if row_index == 0
                else "Change in pressure from the site's mean"
            )
            if column_index == 0:
                axis.set_ylabel(
                    outcome_label if row_index == 0
                    else f"Change in {outcome_label} from the site's mean"
                )
            if row_index == 1:
                axis.axhline(0, color="#657786", linewidth=0.9)
                axis.axvline(0, color="#657786", linewidth=0.9)
        axes[0, column_index].set_title(
            f"{ASSEMBLAGE_LABELS[assemblage]} · {outcome_label}",
            loc="left", fontsize=11.5, weight="bold", pad=10,
        )

    axes[0, 0].text(-0.18, 1.13, "A  Differences between sites", transform=axes[0, 0].transAxes, fontsize=11, weight="bold")
    axes[1, 0].text(-0.18, 1.13, "B  Within-site change at repeat sites", transform=axes[1, 0].transAxes, fontsize=11, weight="bold")
    figure.suptitle(
        "Spatial patterns and within-site change",
        x=0.01, y=0.985, ha="left", fontsize=15.5, weight="bold",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.88, hspace=0.42, wspace=0.30)
    _save_figure(figure, output_directory, "site_balanced_pressure_response_curves.png")
    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    return figure, summary, curves


# Part 2: integrated evidence pathway


# --- Evidence-flow figure -------------------------------------------------
# The final river-style diagram shows where candidate evidence is retained,
# reviewed or lost as the matching checks are applied.
def prepare_evidence_flow(records: pd.DataFrame) -> pd.DataFrame:
    """Assign each survey to one mutually exclusive category at every stage."""
    required = {
        "assemblage", "candidate_count", "match_decision",
        "chemistry_included_in_release", "flow_evidence_available",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise KeyError("Evidence-flow figure is missing columns: " + ", ".join(missing))

    flow = records.copy()
    flow["Biological programme"] = (
        flow["assemblage"].astype("string").map(ASSEMBLAGE_LABELS).fillna("Other biology")
    )
    has_candidate = pd.to_numeric(
        flow["candidate_count"], errors="coerce",
    ).fillna(0).gt(0)
    flow["Network connection"] = np.where(
        has_candidate, "Candidate route found", "No candidate route",
    )

    decision = flow["match_decision"].fillna("no_candidate_match").astype("string")
    flow["Scientific decision"] = np.select(
        [
            decision.eq("accepted"),
            decision.str.contains("discharge", na=False),
            decision.str.contains("tributary", na=False),
            decision.str.contains("distance", na=False),
            decision.str.startswith("provisional", na=False),
            decision.eq("no_candidate_match"),
        ],
        [
            "Accepted route",
            "Review: discharge on route",
            "Review: tributary uncertainty",
            "Review: distance sensitive",
            "Review: other evidence",
            "No route to assess",
        ],
        default="No defensible route",
    )

    chemistry_included = flow["chemistry_included_in_release"].fillna(False).astype(bool)
    flow_available = flow["flow_evidence_available"].fillna(False).astype(bool)
    accepted = decision.eq("accepted")
    review = decision.str.startswith("provisional", na=False)
    flow["Joined data"] = np.select(
        [chemistry_included & flow_available, chemistry_included, accepted, review],
        [
            "Biology + chemistry + flow",
            "Biology + chemistry; no flow",
            "Accepted route; chemistry unavailable",
            "Biology + review record",
        ],
        default="Biology retained; no defensible match",
    )
    return flow


def _spread_label_centres(
    centres: Mapping[str, float], *, minimum_gap: float,
    lower: float = 0.055, upper: float = 0.925,
) -> dict[str, float]:
    """Separate small-node labels while preserving their vertical order."""
    ordered = sorted(centres, key=centres.get)
    positions = [float(centres[label]) for label in ordered]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_gap)
    if positions and positions[-1] > upper:
        shift = positions[-1] - upper
        positions = [value - shift for value in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - minimum_gap)
    if positions and positions[0] < lower:
        shift = lower - positions[0]
        positions = [value + shift for value in positions]
    return dict(zip(ordered, positions))


def plot_evidence_river(
    records: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot survey records as they move through the matching steps."""
    flow = prepare_evidence_flow(records)
    total = len(flow)
    if total == 0:
        raise ValueError("The evidence-flow figure needs at least one survey row")

    stage_columns = [
        "Biological programme", "Network connection",
        "Scientific decision", "Joined data",
    ]
    # Orders are bottom-to-top, keeping the included evidence stream across the top.
    stage_orders = [
        [
            "Other biology", "Transitional/coastal fish", "Macrophytes",
            "Diatoms", "Freshwater fish", "Macroinvertebrates",
        ],
        ["No candidate route", "Candidate route found"],
        [
            "No route to assess", "No defensible route", "Review: other evidence",
            "Review: distance sensitive", "Review: tributary uncertainty",
            "Review: discharge on route", "Accepted route",
        ],
        [
            "Biology retained; no defensible match", "Biology + review record",
            "Accepted route; chemistry unavailable", "Biology + chemistry; no flow",
            "Biology + chemistry + flow",
        ],
    ]
    stage_orders = [
        [label for label in order if flow[column].eq(label).any()]
        for column, order in zip(stage_columns, stage_orders)
    ]
    colours = {
        "Macroinvertebrates": "#466D8A",
        "Freshwater fish": "#5F8F9D",
        "Diatoms": "#78AAB4",
        "Macrophytes": "#7FA37A",
        "Transitional/coastal fish": "#8D99AE",
        "Other biology": "#A7B0B8",
        "Candidate route found": "#497B87",
        "No candidate route": "#A4ADB5",
        "Accepted route": "#087E8B",
        "Review: discharge on route": "#D99A32",
        "Review: tributary uncertainty": "#E9B949",
        "Review: distance sensitive": "#EFCB68",
        "Review: other evidence": "#D5B56B",
        "No defensible route": "#C75B4A",
        "No route to assess": "#89959F",
        "Biology + chemistry + flow": "#176B61",
        "Biology + chemistry; no flow": "#5BC0BE",
        "Accepted route; chemistry unavailable": "#C9A227",
        "Biology + review record": "#E9C46A",
        "Biology retained; no defensible match": "#7D8899",
    }

    stage_x = [0.03, 0.32, 0.64, 0.96]
    stage_gaps = [0.022, 0.055, 0.024, 0.026]
    node_width = 0.011
    height_scale = 0.70 / total
    node_positions: dict[tuple[int, str], tuple[float, float, float]] = {}
    stage_counts: list[dict[str, int]] = []
    for stage, (column, order) in enumerate(zip(stage_columns, stage_orders)):
        counts = flow[column].value_counts().to_dict()
        ordered_counts = {label: int(counts[label]) for label in order}
        stage_counts.append(ordered_counts)
        occupied = (
            sum(ordered_counts.values()) * height_scale
            + stage_gaps[stage] * max(len(ordered_counts) - 1, 0)
        )
        cursor = 0.49 - occupied / 2
        for label, count in ordered_counts.items():
            bottom = cursor
            top = bottom + count * height_scale
            node_positions[(stage, label)] = (stage_x[stage], bottom, top)
            cursor = top + stage_gaps[stage]

    edges: list[tuple[tuple[int, str], tuple[int, str], int, str]] = []
    for stage in range(len(stage_columns) - 1):
        left_column, right_column = stage_columns[stage:stage + 2]
        links = (
            flow.groupby([left_column, right_column], observed=True)
            .size().rename("count").reset_index()
        )
        left_rank = {label: index for index, label in enumerate(stage_orders[stage])}
        right_rank = {label: index for index, label in enumerate(stage_orders[stage + 1])}
        links["left_rank"] = links[left_column].map(left_rank)
        links["right_rank"] = links[right_column].map(right_rank)
        links = links.sort_values(["left_rank", "right_rank"])
        for _, row in links.iterrows():
            source = str(row[left_column])
            target = str(row[right_column])
            colour = colours[source] if stage == 0 else colours[target]
            edges.append(((stage, source), (stage + 1, target), int(row["count"]), colour))

    outgoing = {node: position[1] for node, position in node_positions.items()}
    incoming = {node: position[1] for node, position in node_positions.items()}

    def add_ribbon(axis, source, target, count, colour):
        source_x, _, _ = node_positions[source]
        target_x, _, _ = node_positions[target]
        source_bottom = outgoing[source]
        source_top = source_bottom + count * height_scale
        target_bottom = incoming[target]
        target_top = target_bottom + count * height_scale
        outgoing[source] = source_top
        incoming[target] = target_top
        curve = (target_x - source_x) * 0.48
        vertices = [
            (source_x + node_width, source_bottom),
            (source_x + curve, source_bottom),
            (target_x - curve, target_bottom),
            (target_x - node_width, target_bottom),
            (target_x - node_width, target_top),
            (target_x - curve, target_top),
            (source_x + curve, source_top),
            (source_x + node_width, source_top),
            (source_x + node_width, source_bottom),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        axis.add_patch(PathPatch(
            MplPath(vertices, codes), facecolor=colour,
            edgecolor="none", alpha=0.42, zorder=1,
        ))

    fig, axis = plt.subplots(figsize=(18, 9.2))
    for source, target, count, colour in edges:
        if count > 0:
            add_ribbon(axis, source, target, count, colour)

    label_gaps = [0.066, 0.080, 0.071, 0.071]
    for stage, counts in enumerate(stage_counts):
        centres = {
            label: sum(node_positions[(stage, label)][1:]) / 2
            for label in counts
        }
        label_centres = _spread_label_centres(
            centres, minimum_gap=label_gaps[stage],
        )
        for label, count in counts.items():
            x_value, bottom, top = node_positions[(stage, label)]
            axis.add_patch(Rectangle(
                (x_value - node_width, bottom), 2 * node_width, top - bottom,
                facecolor=colours[label], edgecolor="white",
                linewidth=0.8, zorder=3,
            ))
            if stage == 0:
                text_x, alignment = x_value - 0.018, "right"
                line_start, line_end = x_value - node_width, text_x + 0.004
            else:
                text_x, alignment = x_value + 0.019, "left"
                line_start, line_end = x_value + node_width, text_x - 0.004
            centre = centres[label]
            label_centre = label_centres[label]
            axis.plot(
                [line_start, line_end], [centre, label_centre],
                color="#9AA8AE", linewidth=0.65, zorder=3,
            )
            axis.text(
                text_x, label_centre,
                f"{label}\n{count:,}  ({100 * count / total:.1f}%)",
                ha=alignment, va="center", fontsize=8.4,
                color="#263238", linespacing=1.18, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
            )

    headings = [
        "BIOLOGICAL PROGRAMME", "NETWORK CONNECTION",
        "MATCH CHECK", "JOINED DATA",
    ]
    for x_value, heading in zip(stage_x, headings):
        axis.text(
            x_value, 0.995, heading, ha="center", va="bottom",
            fontsize=9, weight="bold", color="#526876",
        )

    chemistry_included = flow["Joined data"].isin([
        "Biology + chemistry + flow", "Biology + chemistry; no flow",
    ]).sum()
    with_flow = flow["Joined data"].eq("Biology + chemistry + flow").sum()
    candidate_rows = flow["Network connection"].eq("Candidate route found").sum()
    metric_text = (
        f"{candidate_rows / total:.1%} found a connected candidate"
        f"   |   {chemistry_included / total:.1%} included chemistry"
        f"   |   {with_flow / total:.1%} combined chemistry and flow"
    )
    axis.text(
        0.5, 1.045, metric_text, transform=axis.transAxes,
        ha="center", va="center", fontsize=10.5, color="#314A59",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F1F5F6", "edgecolor": "#D4E0E3"},
    )
    axis.set(xlim=(-0.13, 1.18), ylim=(0, 1.13))
    axis.axis("off")
    axis.set_title(
        f"Survey records through the matching pipeline ({total:,} surveys)",
        loc="left", fontsize=17, weight="bold", pad=72,
    )
    fig.text(
        0.01, 0.018,
        "Band widths are survey records. Review and unmatched rows remain in the saved matching tables.",
        fontsize=9, color="#526876",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.90))

    summary_rows = []
    for column, counts in zip(stage_columns, stage_counts):
        for label, count in reversed(list(counts.items())):
            summary_rows.append({
                "stage": column,
                "category": label,
                "survey_rows": count,
                "share_percent": 100 * count / total,
            })
    return fig, pd.DataFrame(summary_rows)
