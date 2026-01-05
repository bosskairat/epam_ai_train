import requests

class MCPClient:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call(self, endpoint: str, params: dict):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
