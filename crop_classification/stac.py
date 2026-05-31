import requests
from pystac_client import Client

try:
    from config import USERNAME, PASSWORD, ROI_BBOX
except ImportError:
    raise ImportError(
        "config.py not found. Copy crop_classification/config_template.py to config.py and fill in Copernicus credentials."
    )

_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"


def get_token():
    """Authenticate with Copernicus Data Space and return an access token."""
    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
    }
    response = requests.post(_TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json().get("access_token")


def search_sentinel2(
    start_date, end_date, max_cloud_cover=20, collection="sentinel-2-l2a"
):
    """Search Copernicus STAC for Sentinel-2 L2A scenes over the configured ROI."""
    client = Client.open(_STAC_URL)
    search = client.search(
        collections=[collection],
        bbox=ROI_BBOX,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    return list(search.get_items())


def list_scene_assets(items):
    """Return a summary of asset keys and counts for a list of STAC items."""
    return {item.id: list(item.assets.keys()) for item in items}
