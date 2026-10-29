import os
from pathlib import Path

import pandas as pd

from connected_river.download.core_chemistry import download_core_chemistry
from connected_river.download.priority_metals import (
    download_priority_metals,
    fetch_determinand_codelist,
)
from connected_river.download.target_determinands import (
    download_additional_target_determinands,
)
from connected_river.workflow import adapt_custom_network, prepare_workspace


def test_prepare_workspace_keeps_generated_data_separate(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CONNECTED_RIVER_WORKSPACE", str(repository))
    workspace = prepare_workspace(repository, tmp_path / "workspace")

    assert (workspace / "config/source_catalog.yaml").is_file()
    assert (workspace / "notebooks/The_Connected_River_Part_1_Building.ipynb").is_file()
    assert (workspace / "notebooks/The_Connected_River_Part_2_Exploring.ipynb").is_file()
    assert workspace != repository
    assert os.environ["CONNECTED_RIVER_WORKSPACE"] == str(workspace)


def test_custom_network_adapter_maps_the_minimum_contract(tmp_path):
    source = tmp_path / "network.csv"
    destination = tmp_path / "adapted.csv"
    pd.DataFrame(
        {
            "edge": [1],
            "upstream": [10],
            "downstream": [11],
            "metres": [125.0],
            "line": ["LINESTRING (0 0, 1 1)"],
        }
    ).to_csv(source, index=False)

    adapt_custom_network(
        source,
        destination,
        {
            "edge_id": "edge",
            "from_node": "upstream",
            "to_node": "downstream",
            "length_m": "metres",
            "geometry_wkt": "line",
        },
    )

    adapted = pd.read_csv(destination)
    assert adapted.loc[0, "FNODE_FULL"] == 10
    assert adapted.loc[0, "TNODE_FULL"] == 11
    assert adapted.loc[0, "LENGTH"] == 125.0


def test_determinand_codelist_cache_uses_the_declared_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNECTED_RIVER_WORKSPACE", str(tmp_path))
    cache = tmp_path / "data/raw/water_quality/determinand_codelist.csv"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        "notation,prefLabel,uri\n0079,Copper,https://example.test/0079\n",
        encoding="utf-8",
    )

    codelist = fetch_determinand_codelist(refresh=False)

    assert codelist.loc[0, "priority_metal"] == "copper"


def test_partitioned_download_directories_use_the_declared_workspace(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("CONNECTED_RIVER_WORKSPACE", str(tmp_path))
    cache = tmp_path / "data/raw/water_quality/determinand_codelist.csv"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        "notation,prefLabel,uri\n0079,Copper,https://example.test/0079\n",
        encoding="utf-8",
    )

    assert download_core_chemistry([]) == []
    assert download_priority_metals([]) == []
    assert download_additional_target_determinands([]) == []
    assert (tmp_path / "data/raw/water_quality/core_chemistry").is_dir()
    assert (tmp_path / "data/raw/water_quality/metals_priority_fast").is_dir()
    assert (tmp_path / "data/raw/water_quality/target_determinands").is_dir()
