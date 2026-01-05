import requests

class MCPClient:
    def __init__(self, base_url, timeout=5):
        self.base_url = base_url
        self.timeout = timeout

    def call(self, endpoint: str, params: dict):
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return {
                "ok": True,
                "data": response.json()
            }

        except requests.exceptions.Timeout:
            return {
                "ok": False,
                "error": "Service timeout"
            }

        except requests.exceptions.ConnectionError:
            return {
                "ok": False,
                "error": "Service unavailable"
            }

        except requests.exceptions.HTTPError as e:
            return {
                "ok": False,
                "error": f"HTTP error: {e.response.status_code}"
            }

        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }
