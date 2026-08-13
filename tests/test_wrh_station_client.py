from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.wrh_station_client import WRHStationClient, WRHStationError


def _response(*, age_minutes: int = 5) -> dict:
    timestamp = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    return {
        "SUMMARY": {"RESPONSE_CODE": 1, "RESPONSE_MESSAGE": "OK"},
        "STATION": [{
            "STID": "E7138",
            "OBSERVATIONS": {
                "date_time": [timestamp],
                "air_temp_set_1": [64.0],
                "relative_humidity_set_1": [93.0],
                "wind_speed_set_1": [1.0],
            },
        }],
    }


class TestWRHStationClient:
    @patch("src.wrh_station_client.requests.get")
    def test_reads_viewer_token_and_latest_observation(self, mock_get):
        token_response = MagicMock(
            ok=True,
            text="const apiToken = '0123456789abcdef0123456789abcdef';",
        )
        data_response = MagicMock(ok=True)
        data_response.json.return_value = _response()
        mock_get.side_effect = [token_response, data_response]

        result = WRHStationClient().get_station_observation("E7138", 15)

        assert result["temperature_f"] == 64.0
        assert result["humidity"] == 93.0
        assert result["wind_speed_mph"] == 1.0
        assert mock_get.call_args_list[1].kwargs["headers"]["Origin"] == (
            "https://www.weather.gov"
        )

    @patch("src.wrh_station_client.requests.get")
    def test_rejects_stale_observation(self, mock_get):
        token_response = MagicMock(
            ok=True,
            text="const apiToken = '0123456789abcdef0123456789abcdef';",
        )
        data_response = MagicMock(ok=True)
        data_response.json.return_value = _response(age_minutes=16)
        mock_get.side_effect = [token_response, data_response]

        with pytest.raises(WRHStationError, match="stale"):
            WRHStationClient().get_station_observation("E7138", 15)

    @patch("src.wrh_station_client.requests.get")
    def test_rejects_missing_public_token(self, mock_get):
        mock_get.return_value = MagicMock(ok=True, text="no token here")

        with pytest.raises(WRHStationError, match="token was not found"):
            WRHStationClient().get_station_observation("E7138")
