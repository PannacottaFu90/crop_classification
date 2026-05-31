"""Example configuration file for the crop classification project.

Copy this file to the repository root as `config.py` and fill in the credentials
and connection strings before running the pipeline.
"""

import os

# Azure Storage settings
STORAGE_ACCOUNT = "s2wallonia2023"
CONTAINER_NAME = "sentinel2-data"
STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY", "")
CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};"
    f"AccountKey={STORAGE_KEY};EndpointSuffix=core.windows.net"
)

# Copernicus Data Space credentials
USERNAME = ""
PASSWORD = ""
MY_PREFIX = "raw/"

# Study area and Sentinel-2 settings
ROI_BBOX = [4.55, 50.25, 5.15, 50.55]
BANDS = [
    "B02_10m",
    "B03_10m",
    "B04_10m",
    "B05_20m",
    "B06_20m",
    "B07_20m",
    "B08_10m",
    "B8A_20m",
    "B11_20m",
    "B12_20m",
    "SCL_20m",
]

# Output defaults
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
DATA_DIR = "data"
