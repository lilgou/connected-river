# Consented discharge download

import pandas as pd

from connected_river.download import discharge_permits, nrfa


class _DischargeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DischargeSession:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get(self, _url, *, params, timeout):
        assert timeout == 180
        if params.get("returnCountOnly") == "true":
            return _DischargeResponse({"count": 3})
        offset = int(params["resultOffset"])
        records = [
            (1, "A", -1.0, 51.0),
            (2, "B", -2.0, 52.0),
            (3, "C", -3.0, 53.0),
        ][offset:offset + 2]
        return _DischargeResponse({
            "features": [
                {
                    "attributes": {
                        "OBJECTID_1": object_id,
                        "PERMIT_NUMBER": permit,
                    },
                    "geometry": {"x": longitude, "y": latitude},
                }
                for object_id, permit, longitude, latitude in records
            ]
        })


def test_arcgis_discharge_download_is_complete_and_workspace_local(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("CONNECTED_RIVER_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(discharge_permits, "ARCGIS_PAGE_SIZE", 2)
    monkeypatch.setattr(
        discharge_permits, "session_with_retries", lambda **_kwargs: _DischargeSession(),
    )
    monkeypatch.setattr(discharge_permits.time, "sleep", lambda _seconds: None)

    result = discharge_permits.download_consented_discharges()

    output = result["csv"]
    table = pd.read_csv(output)
    assert output.is_relative_to(tmp_path)
    assert table["OBJECTID_1"].tolist() == [1, 2, 3]
    assert table["longitude"].tolist() == [-1.0, -2.0, -3.0]
    assert (tmp_path / "outputs/logs/download_manifest.json").is_file()


# NRFA downloads


class _NrfaResponse:
    text = (
        "id,name,catchment-area,easting,northing,latitude,longitude,river,location,"
        "gdf-mean-flow,gdf-percent-complete\n"
        "43009,Test gauge,123.4,345000,234000,52.0,-1.0,Test River,Test Place,"
        "4.2,99.8\n"
    )

    def raise_for_status(self):
        return None


class _NrfaSession:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get(self, url, *, params, timeout):
        assert url == nrfa.NRFA_STATION_INFO_URL
        assert params["station"] == "*"
        assert params["format"] == "csv"
        assert "gdf-statistics" in params["fields"]
        assert timeout == 180
        return _NrfaResponse()


def test_nrfa_station_catalogue_is_downloaded_to_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNECTED_RIVER_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(nrfa, "session_with_retries", lambda: _NrfaSession())

    output = nrfa.download_nrfa_station_catalogue()

    assert output == tmp_path / "data/raw/nrfa/nrfa_station_catalogue.parquet"
    catalogue = pd.read_parquet(output)
    assert catalogue.loc[0, "id"] == 43009
    assert catalogue.loc[0, "gdf-mean-flow"] == 4.2
    assert nrfa.is_valid_station_catalogue(output)
    assert (tmp_path / "outputs/logs/download_manifest.json").is_file()


def test_nrfa_flow_directories_use_the_declared_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNECTED_RIVER_WORKSPACE", str(tmp_path))

    assert nrfa.download_nrfa_flows([]) == []
    assert (tmp_path / "data/raw/nrfa/flows").is_dir()
