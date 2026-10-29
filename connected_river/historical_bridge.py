"""Continue a published historical site group using the current data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import TwoSlopeNorm

DATASET_DOI = "https://doi.org/10.5285/23f194b0-1e21-4f14-be9c-6eabdcc1feb7"
DATASET_DOWNLOAD = "https://data-package.ceh.ac.uk/data/23f194b0-1e21-4f14-be9c-6eabdcc1feb7.zip"
TAXON_MEMBER = re.compile(r"cp_[a-z]{2}_minv_taxonabundance_.*\.csv$")


@dataclass(frozen=True)
class HistoricalContinuation:
    """Tables and figure produced by the historical continuation analysis."""

    archive_path: Path
    bridge: pd.DataFrame
    annual: pd.DataFrame
    catchment_annual: pd.DataFrame
    area_changes: pd.DataFrame
    paired_changes: pd.DataFrame
    headline: pd.DataFrame
    figure: plt.Figure
    catchment_figure: plt.Figure
    area_figure: plt.Figure
    summary_markdown: str


def acquire_historical_archive(
    configured_path: str | Path | None,
    cache_directory: str | Path,
) -> Path:
    """Use a supplied archive or download the cited open dataset once."""

    if configured_path:
        supplied = Path(configured_path).expanduser().resolve()
        if not supplied.is_file():
            raise FileNotFoundError(f"Historical archive not found: {supplied}")
        return supplied
    destination = Path(cache_directory) / "historical_macroinvertebrate_dataset_v2.zip"
    if destination.is_file() and destination.stat().st_size > 10_000_000:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    with requests.get(DATASET_DOWNLOAD, stream=True, timeout=180) as response:
        response.raise_for_status()
        with temporary.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
    temporary.replace(destination)
    return destination


def _clean_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


# Only the historical site and sample keys are needed for the bridge. Keeping
# this narrow avoids mixing the deposited measurements into the current pipeline.
def load_historical_sample_keys(archive_path: str | Path) -> pd.DataFrame:
    """Read the deposited macroinvertebrate sample identities."""

    parts: list[pd.DataFrame] = []
    with ZipFile(archive_path) as archive:
        members = sorted(name for name in archive.namelist() if TAXON_MEMBER.search(name))
        if not members:
            raise ValueError("The archive contains no expected taxon-abundance tables")
        for member in members:
            with archive.open(member) as source:
                frame = pd.read_csv(
                    source,
                    usecols=["BIO_SITE_ID", "BIO_SAMPLE_ID", "BIO_SAMPLE_DATE"],
                    dtype="string",
                ).drop_duplicates()
            parts.append(frame)
    samples = pd.concat(parts, ignore_index=True).drop_duplicates()
    samples = samples.rename(columns={
        "BIO_SITE_ID": "site_id", "BIO_SAMPLE_ID": "sample_id",
        "BIO_SAMPLE_DATE": "sample_date",
    })
    samples["site_id"] = _clean_id(samples["site_id"])
    samples["sample_id"] = _clean_id(samples["sample_id"])
    samples["sample_date"] = pd.to_datetime(samples["sample_date"], errors="coerce")
    return samples.dropna(subset=["site_id", "sample_id", "sample_date"])


def _bootstrap_median(
    values: pd.Series,
    *,
    random_seed: int,
    draws: int = 800,
) -> tuple[float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return np.nan, np.nan, np.nan
    estimate = float(np.median(numeric))
    if len(numeric) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(random_seed)
    simulated = np.median(
        rng.choice(numeric, size=(draws, len(numeric)), replace=True), axis=1,
    )
    lower, upper = np.quantile(simulated, [0.025, 0.975])
    return estimate, float(lower), float(upper)


# Compare periods within sites that occur on both sides of the split. This avoids
# a change in the monitored site mix being mistaken for environmental change.
def _paired_period_change(
    samples: pd.DataFrame,
    *,
    historical_period: tuple[int, int],
    recent_period: tuple[int, int],
) -> pd.DataFrame:
    frame = samples.loc[
        samples["sample_year"].between(*historical_period)
        | samples["sample_year"].between(*recent_period)
    ].copy()
    frame["period"] = np.where(
        frame["sample_year"].between(*historical_period), "historical", "recent",
    )
    frame["sample_month"] = frame["sample_date"].dt.month
    represented = frame[["site_id", "period", "sample_month"]].drop_duplicates()
    shared = represented.groupby(["site_id", "sample_month"])["period"].nunique()
    shared = shared.loc[shared.eq(2)].reset_index()[["site_id", "sample_month"]]
    frame = frame.merge(shared, on=["site_id", "sample_month"], how="inner")
    values = frame.groupby(["site_id", "period"])["whpt_aspt"].median().unstack()
    values = values.dropna(subset=["historical", "recent"]).copy()
    values["change"] = values["recent"] - values["historical"]
    return values.reset_index()


# Part 2: historical continuation figures


# Link the historical site group to current observations, then report national
# and catchment patterns separately so local differences remain visible.
def build_historical_continuation(
    release_directory: str | Path,
    archive_path: str | Path,
    output_directory: str | Path,
    *,
    historical_period: tuple[int, int] = (1990, 1999),
) -> HistoricalContinuation:
    """Follow the deposited cohort with one consistently processed outcome."""

    release_directory = Path(release_directory)
    archive_path = Path(archive_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    coverage = json.loads(
        (release_directory / "public_temporal_coverage_qa.json").read_text()
    )
    final_year = max(coverage["observed_automated_product_years"])
    recent_period = (2019, final_year)

    deposited = load_historical_sample_keys(archive_path)
    ecology_path = release_directory.parent / "data/ecology/ecology_samples.parquet"
    ecology = pd.read_parquet(
        ecology_path,
        columns=[
            "assemblage", "site_id", "sample_id", "sample_date", "sample_year",
            "biology_metric_whpt_aspt", "catchment", "easting", "northing",
        ],
    )
    ecology = ecology.loc[ecology["assemblage"].eq("macroinvertebrate")].copy()
    ecology["site_id"] = _clean_id(ecology["site_id"])
    ecology["sample_id"] = _clean_id(ecology["sample_id"])
    ecology["sample_date"] = pd.to_datetime(ecology["sample_date"], errors="coerce")
    ecology["sample_year"] = pd.to_numeric(ecology["sample_year"], errors="coerce")
    ecology["whpt_aspt"] = pd.to_numeric(
        ecology["biology_metric_whpt_aspt"], errors="coerce",
    )

    exact = deposited.merge(
        ecology[["site_id", "sample_id", "sample_date"]].drop_duplicates(),
        on=["site_id", "sample_id", "sample_date"], how="inner",
    )
    cohort_sites = set(deposited["site_id"].dropna())
    cohort = ecology.loc[
        ecology["site_id"].isin(cohort_sites)
        & ecology["sample_year"].between(deposited["sample_date"].dt.year.min(), final_year)
    ].dropna(subset=["sample_date", "sample_year", "whpt_aspt"]).copy()
    cohort["sample_year"] = cohort["sample_year"].astype(int)

    site_year = cohort.groupby(["site_id", "sample_year"], as_index=False).agg(
        whpt_aspt=("whpt_aspt", "median"), samples=("sample_id", "nunique"),
        catchment=("catchment", "first"), easting=("easting", "median"),
        northing=("northing", "median"),
    )
    site_year["site_median_whpt_aspt"] = site_year.groupby("site_id")[
        "whpt_aspt"
    ].transform("median")
    site_year["site_centered_change"] = (
        site_year["whpt_aspt"] - site_year["site_median_whpt_aspt"]
    )
    annual_rows = []
    for offset, (year, frame) in enumerate(site_year.groupby("sample_year", sort=True)):
        estimate, lower, upper = _bootstrap_median(
            frame["whpt_aspt"], random_seed=4500 + offset,
        )
        centred, centred_lower, centred_upper = _bootstrap_median(
            frame["site_centered_change"], random_seed=6500 + offset,
        )
        annual_rows.append({
            "sample_year": int(year), "sites": int(frame["site_id"].nunique()),
            "median_whpt_aspt": estimate, "lower_95": lower, "upper_95": upper,
            "median_site_centered_change": centred,
            "centered_lower_95": centred_lower,
            "centered_upper_95": centred_upper,
        })
    annual = pd.DataFrame(annual_rows)
    annual["five_year_median"] = annual["median_whpt_aspt"].rolling(
        5, center=True, min_periods=3,
    ).median()
    annual["five_year_site_centered_change"] = annual[
        "median_site_centered_change"
    ].rolling(5, center=True, min_periods=3).median()

    paired = _paired_period_change(
        cohort, historical_period=historical_period, recent_period=recent_period,
    )
    estimate, lower, upper = _bootstrap_median(
        paired["change"], random_seed=8127, draws=1600,
    )
    headline = pd.DataFrame([{
        "historical period": f"{historical_period[0]}–{historical_period[1]}",
        "recent period": f"{recent_period[0]}–{recent_period[1]}",
        "paired sites": len(paired),
        "historical median WHPT ASPT": float(paired["historical"].median()),
        "recent median WHPT ASPT": float(paired["recent"].median()),
        "median within-site change": estimate,
        "lower 95%": lower, "upper 95%": upper,
        "sites with higher recent value (%)": float(100 * paired["change"].gt(0).mean()),
    }])

    site_attributes = (
        site_year.sort_values("sample_year")
        .groupby("site_id", as_index=False)
        .agg(
            catchment=("catchment", "first"), easting=("easting", "median"),
            northing=("northing", "median"),
        )
    )
    paired_attributes = paired.merge(site_attributes, on="site_id", how="left")
    catchment_support = (
        paired_attributes.dropna(subset=["catchment"])
        .groupby("catchment")["site_id"].nunique()
        .sort_values(ascending=False).head(6)
    )
    catchment_rows: list[pd.DataFrame] = []
    full_years = pd.DataFrame({"sample_year": range(1985, final_year + 1)})
    for catchment_name, paired_sites in catchment_support.items():
        trajectory = (
            site_year.loc[site_year["catchment"].eq(catchment_name)]
            .groupby("sample_year", as_index=False)
            .agg(
                sites=("site_id", "nunique"),
                median_site_centered_change=("site_centered_change", "median"),
            )
        )
        trajectory = full_years.merge(trajectory, on="sample_year", how="left")
        trajectory.loc[
            trajectory["sites"].fillna(0).lt(5), "median_site_centered_change"
        ] = np.nan
        trajectory["five_year_site_centered_change"] = trajectory[
            "median_site_centered_change"
        ].rolling(5, center=True, min_periods=3).median()
        trajectory["catchment"] = catchment_name
        trajectory["paired_sites"] = int(paired_sites)
        catchment_rows.append(trajectory)
    catchment_annual = pd.concat(catchment_rows, ignore_index=True)

    area_path = release_directory / "hydrological_area_map_geometry.geojson"
    areas = gpd.read_file(area_path).to_crs("EPSG:4326")
    areas["map_area_id"] = areas["id"].astype(str)
    valid_sites = site_attributes.dropna(subset=["easting", "northing"]).copy()
    points = gpd.GeoDataFrame(
        valid_sites[["site_id"]],
        geometry=gpd.points_from_xy(valid_sites["easting"], valid_sites["northing"]),
        crs="EPSG:27700",
    ).to_crs("EPSG:4326")
    site_areas = (
        gpd.sjoin(
            points,
            areas[["map_area_id", "catchment_name", "geometry"]],
            how="left", predicate="within",
        )
        .sort_values("site_id").drop_duplicates("site_id")
        [["site_id", "map_area_id", "catchment_name"]]
    )
    paired_areas = paired.merge(site_areas, on="site_id", how="left")
    area_summary = paired_areas.dropna(subset=["map_area_id"]).groupby(
        ["map_area_id", "catchment_name"], as_index=False,
    ).agg(
        paired_sites=("site_id", "nunique"),
        median_change=("change", "median"),
        sites_higher_percent=("change", lambda values: 100 * values.gt(0).mean()),
    )
    area_changes = areas.merge(
        area_summary, on=["map_area_id", "catchment_name"], how="left",
    )
    area_changes["well_supported"] = area_changes["paired_sites"].ge(5)
    bridge = pd.DataFrame([{
        "deposited samples": len(deposited),
        "exact samples found in current EA archive": len(exact),
        "exact match (%)": float(100 * len(exact) / len(deposited)),
        "deposited sites": int(deposited["site_id"].nunique()),
        "sites observed after 2018": int(cohort.loc[cohort["sample_year"].ge(2019), "site_id"].nunique()),
        "outcome used": "WHPT ASPT from the current EA archive",
        "latest complete year": final_year,
    }])

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    bar_colours = np.where(annual["sites"].ge(50), "#5F8F9D", "#C8D2D7")
    axes[0].bar(
        annual["sample_year"], annual["sites"], color=bar_colours, width=0.9,
    )
    axes[0].text(
        1966, axes[0].get_ylim()[1] * 0.91, "Early years: fewer than 50 sites",
        ha="left", va="top", fontsize=8, color="#657786",
    )
    axes[0].set(title="A. Monitoring effort", xlabel="Year", ylabel="Sites sampled")

    axes[1].fill_between(
        annual["sample_year"], annual["lower_95"], annual["upper_95"],
        color="#9BC1BC", alpha=0.28, label="95% site-bootstrap interval",
    )
    axes[1].plot(annual["sample_year"], annual["median_whpt_aspt"], color="#6A9A8B", alpha=0.55, linewidth=1.2, label="Annual median")
    axes[1].plot(annual["sample_year"], annual["five_year_median"], color="#17324D", linewidth=2.6, label="Five-year median")
    axes[1].axvline(2018.5, color="#D95D39", linestyle="--", linewidth=1.2)
    axes[1].set(
        title="B. National WHPT ASPT",
        xlabel="Year", ylabel="Site-balanced annual median",
    )
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    axes[2].fill_between(
        annual["sample_year"], annual["centered_lower_95"],
        annual["centered_upper_95"], color="#E7B665", alpha=0.24,
        label="95% site-bootstrap interval",
    )
    axes[2].plot(
        annual["sample_year"], annual["median_site_centered_change"],
        color="#D28A22", alpha=0.55, linewidth=1.2, label="Annual median",
    )
    axes[2].plot(
        annual["sample_year"], annual["five_year_site_centered_change"],
        color="#9B5D00", linewidth=2.6, label="Five-year median",
    )
    axes[2].axhline(0, color="#657786", linewidth=1.0)
    axes[2].set(
        title="C. Change within the same sites", xlabel="Year",
        ylabel="WHPT ASPT relative to each site's median",
    )
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    for axis in axes:
        axis.axvline(2018.5, color="#D95D39", linestyle="--", linewidth=1.1)
        axis.axvspan(2019.6, 2020.4, color="#D9D9D9", alpha=0.35)
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Monitoring effort and WHPT ASPT through time",
        x=0.01, y=0.985, ha="left", fontsize=15.5, weight="bold",
    )
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.15, top=0.82, wspace=0.27)
    figure.savefig(output_directory / "historical_cohort_continuation.png", dpi=180, bbox_inches="tight")

    catchment_figure, catchment_axes = plt.subplots(
        2, 3, figsize=(15.5, 8.0), sharex=True, sharey=True,
    )
    catchment_colours = ["#2A6F97", "#358F80", "#8F6B3E", "#6C5B7B", "#C06C54", "#477998"]
    for axis, (catchment_name, paired_sites), colour in zip(
        catchment_axes.flat, catchment_support.items(), catchment_colours,
    ):
        frame = catchment_annual.loc[catchment_annual["catchment"].eq(catchment_name)]
        axis.plot(
            frame["sample_year"], frame["median_site_centered_change"],
            color=colour, linewidth=1.0, alpha=0.45,
        )
        axis.plot(
            frame["sample_year"], frame["five_year_site_centered_change"],
            color=colour, linewidth=2.5,
        )
        axis.axhline(0, color="#7B8790", linewidth=0.9)
        axis.axvspan(2019.6, 2020.4, color="#D9D9D9", alpha=0.35)
        axis.set_title(
            f"{str(catchment_name).title()}\n{int(paired_sites):,} paired sites",
            loc="left", fontsize=10.5,
        )
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in catchment_axes[:, 0]:
        axis.set_ylabel("Site-centred WHPT ASPT")
    for axis in catchment_axes[-1, :]:
        axis.set_xlabel("Year")
    catchment_figure.suptitle(
        "Site-centred WHPT ASPT in six well-monitored catchments",
        x=0.01, y=0.985, ha="left", fontsize=15.5, weight="bold",
    )
    catchment_figure.text(
        0.01, 0.935,
        "Six catchments with the most sites in the paired comparison; annual medians and five-year rolling medians, shown where at least five sites contribute.",
        ha="left", fontsize=9.5, color="#526876",
    )
    catchment_figure.subplots_adjust(
        left=0.07, right=0.985, bottom=0.08, top=0.87, hspace=0.34, wspace=0.20,
    )
    catchment_figure.savefig(
        output_directory / "historical_catchment_trajectories.png",
        dpi=180, bbox_inches="tight",
    )

    supported_areas = area_changes.loc[area_changes["well_supported"]].copy()
    maximum = max(1.05, float(supported_areas["median_change"].abs().max()))
    colour_norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    colour_map = plt.get_cmap("RdYlGn")
    area_figure, (map_axis, rank_axis) = plt.subplots(
        1, 2, figsize=(15.5, 9.2), gridspec_kw={"width_ratios": [0.9, 1.15]},
    )
    area_changes.plot(
        ax=map_axis, color="#E1E5E7", edgecolor="white", linewidth=0.35,
    )
    supported_areas.plot(
        ax=map_axis, column="median_change", cmap=colour_map, norm=colour_norm,
        edgecolor="white", linewidth=0.45,
    )
    min_x, min_y, max_x, max_y = supported_areas.total_bounds
    x_pad, y_pad = (max_x - min_x) * 0.05, (max_y - min_y) * 0.05
    map_axis.set(
        xlim=(min_x - x_pad, max_x + x_pad),
        ylim=(min_y - y_pad, max_y + y_pad),
    )
    colour_bar = area_figure.colorbar(
        plt.cm.ScalarMappable(norm=colour_norm, cmap=colour_map),
        ax=map_axis, orientation="horizontal", shrink=0.72, pad=0.025,
    )
    colour_bar.set_label("Median within-site WHPT ASPT change")
    map_axis.set_axis_off()
    map_axis.set_title("A. Median change by hydrometric area", loc="left")

    ranking = supported_areas.sort_values("median_change").reset_index(drop=True)
    positions = np.arange(len(ranking))
    rank_axis.barh(
        positions, ranking["median_change"],
        color=colour_map(colour_norm(ranking["median_change"])), height=0.72,
    )
    rank_axis.axvline(0, color="#657786", linewidth=1.0)
    rank_axis.set_yticks(
        positions,
        [
            f"{name} ({int(count):,})"
            for name, count in zip(ranking["catchment_name"], ranking["paired_sites"])
        ],
        fontsize=7.5,
    )
    rank_axis.set(
        xlabel="Median within-site WHPT ASPT change",
    )
    rank_axis.set_title("B. Areas with at least five paired sites", loc="left")
    rank_axis.grid(axis="x", alpha=0.16)
    rank_axis.spines[["top", "right"]].set_visible(False)
    area_figure.suptitle(
        "Paired-site WHPT ASPT change by hydrometric area",
        x=0.01, y=0.985, ha="left", fontsize=15.5, weight="bold",
    )
    area_figure.text(
        0.01, 0.945,
        f"Median within-site change from {historical_period[0]}–{historical_period[1]} to {recent_period[0]}–{recent_period[1]}; labels give paired-site counts and grey areas have fewer than five.",
        ha="left", fontsize=9.5, color="#526876",
    )
    area_figure.subplots_adjust(
        left=0.04, right=0.985, bottom=0.07, top=0.90, wspace=0.20,
    )
    area_figure.savefig(
        output_directory / "historical_change_by_hydrometric_area.png",
        dpi=180, bbox_inches="tight",
    )

    first_supported = annual.loc[
        annual["sample_year"].ge(1985) & annual["sites"].ge(50)
    ].iloc[0]
    last_annual = annual.iloc[-1]
    supported_count = len(supported_areas)
    positive_count = int(supported_areas["median_change"].gt(0).sum())
    summary_markdown = (
        "**Long-term pattern.** Coverage is sparse before the mid-1980s, so those early "
        "values are shown but not used for the period comparison. The five-year median "
        f"site-centred value changes from {first_supported['five_year_site_centered_change']:+.2f} "
        f"in {int(first_supported['sample_year'])} to "
        f"{last_annual['five_year_site_centered_change']:+.2f} in {int(last_annual['sample_year'])}. "
        f"At hydrometric-area level, {positive_count} of {supported_count} areas with at "
        "least five paired sites have a positive median change. These are descriptive "
        "patterns and do not identify the cause of change."
    )

    bridge.to_csv(output_directory / "historical_bridge_summary.csv", index=False)
    annual.to_csv(output_directory / "historical_cohort_annual_whpt_aspt.csv", index=False)
    catchment_annual.to_csv(
        output_directory / "historical_catchment_annual_whpt_aspt.csv", index=False,
    )
    area_summary.to_csv(
        output_directory / "historical_hydrometric_area_changes.csv", index=False,
    )
    paired.to_csv(output_directory / "historical_cohort_paired_changes.csv", index=False)
    headline.to_csv(output_directory / "historical_cohort_change_summary.csv", index=False)
    plt.close(figure)
    plt.close(catchment_figure)
    plt.close(area_figure)
    return HistoricalContinuation(
        archive_path=archive_path, bridge=bridge, annual=annual,
        catchment_annual=catchment_annual, area_changes=area_changes,
        paired_changes=paired, headline=headline, figure=figure,
        catchment_figure=catchment_figure, area_figure=area_figure,
        summary_markdown=summary_markdown,
    )
