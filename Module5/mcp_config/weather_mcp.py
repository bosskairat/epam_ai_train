from .client import MCPClient

class WeatherMCP:
    def __init__(self):
        self.client = MCPClient("https://api.open-meteo.com/v1")

    def get_weather(self, latitude: float, longitude: float, days: int = 1):
        """
        Fetch weather forecast.

        :param latitude: float
        :param longitude: float
        :param days: number of forecast days to return (default 1)
        :return: JSON from Open-Meteo
        """
        # Open-Meteo supports daily fields
        daily_fields = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "weathercode",
            "precipitation_probability_max"
        ]

        return self.client.call(
            "forecast",
            {
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
                "daily": ",".join(daily_fields),
                "timezone": "auto",
                "forecast_days": days
            }
        )
