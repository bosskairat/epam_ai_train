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


class GeoMCP:
    def __init__(self):
        self.client = MCPClient("https://geocoding-api.open-meteo.com/v1")

    def get_city_coords(self, city_name: str):
        """
        Get latitude and longitude of a city using Open-Meteo Geocoding API.
        """
        result = self.client.call(
            "search",
            {
                "name": city_name,
                "count": 1,
                "language": "en",
                "format": "json"
            }
        )

        if not result["ok"]:
            return {
                "ok": False,
                "message": "Geocoding service is temporarily unavailable"
            }

        data = result.get("data", {})
        results = data.get("results", [])

        if not results:
            return {
                "ok": False,
                "message": f"City '{city_name}' not found"
            }

        city = results[0]

        return {
            "ok": True,
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "name": city.get("name"),
            "country": city.get("country")
        }

