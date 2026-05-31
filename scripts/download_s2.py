"""Example script to search Sentinel-2 scenes via Copernicus STAC and optionally download asset links."""

import argparse
import os
import requests
from crop_classification.stac import get_token, search_sentinel2


def download_asset(url, local_path, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)


def main():
    parser = argparse.ArgumentParser(
        description="Search Sentinel-2 scenes and save STAC asset URLs."
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-cloud", type=int, default=20)
    parser.add_argument("--output-dir", default="downloads")
    parser.add_argument("--download-assets", action="store_true")
    args = parser.parse_args()

    items = search_sentinel2(args.start_date, args.end_date, args.max_cloud)
    print(f"Found {len(items)} scenes")

    for item in items:
        print(f"Scene: {item.id}")
        for key, asset in item.assets.items():
            print(f"  - {key}: {asset.href}")

    if args.download_assets:
        token = get_token()
        for item in items:
            item_dir = os.path.join(args.output_dir, item.id)
            os.makedirs(item_dir, exist_ok=True)
            for key, asset in item.assets.items():
                local_file = os.path.join(item_dir, f"{key}.tif")
                print(f"Downloading {asset.href} to {local_file}")
                download_asset(asset.href, local_file, token=token)


if __name__ == "__main__":
    main()
