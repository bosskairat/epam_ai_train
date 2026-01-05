from .client import MCPClient


class WeatherMCP:
    def __init__(self):
        self.client = MCPClient("https://api.open-meteo.com/v1")

    def get_weather(self, latitude, longitude, days=1):
        result = self.client.call(
            "forecast",
            {
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max"
                ],
                "forecast_days": days,
                "timezone": "auto"
            }
        )

        if not result["ok"]:
            return {
                "ok": False,
                "message": "Weather service is temporarily unavailable"
            }

        return {
            "ok": True,
            "data": result["data"]
        }

