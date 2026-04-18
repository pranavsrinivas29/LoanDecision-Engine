import requests


def post_request(base_url: str, endpoint: str, payload: dict, timeout: int = 60):
    url = f"{base_url.rstrip('/')}{endpoint}"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_request(base_url: str, endpoint: str, timeout: int = 30):
    url = f"{base_url.rstrip('/')}{endpoint}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()