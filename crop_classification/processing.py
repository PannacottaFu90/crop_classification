import os
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

from .azure_blob import download_blob_to_bytes, get_container_client
from .config_template import ROI_BBOX, BANDS

NO_DATA = -9999.0
VALID_SCL = [4, 5, 6, 7, 11]


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def list_raw_blobs(prefix="raw/"):
    container_client = get_container_client()
    return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]


def group_scenes_by_month(blob_names):
    """Group raw blob paths by month key and band name."""
    monthly = defaultdict(list)
    for blob_name in blob_names:
        filename = os.path.basename(blob_name)
        if filename.startswith("."):
            continue
        if len(filename) < 8 or not filename[:6].isdigit():
            continue
        month_key = f"{filename[0:4]}-{filename[4:6]}"
        monthly[month_key].append(blob_name)
    return monthly


def get_cloud_mask(scl_array, invalid_codes=None):
    """Return a boolean mask where invalid pixels are True."""
    if invalid_codes is None:
        invalid_codes = [0, 1, 3, 8, 9, 10, 11]
    return np.isin(scl_array.astype(int), invalid_codes)


def _open_blob_raster(blob_name):
    data = download_blob_to_bytes(blob_name)
    mem = MemoryFile(data.read())
    return mem.open()


def _compute_out_shape(src, bbox, target_resolution):
    window = from_bounds(*bbox, transform=src.transform)
    source_resolution = abs(src.transform.a)
    scale = source_resolution / target_resolution
    height = int(np.ceil(window.height / scale))
    width = int(np.ceil(window.width / scale))
    return window, (height, width)


def read_band_window(blob_name, bbox=None, target_resolution=20):
    """Read a raster band from Azure and resample it to the target resolution."""
    if bbox is None:
        bbox = ROI_BBOX
    with _open_blob_raster(blob_name) as src:
        window, out_shape = _compute_out_shape(src, bbox, target_resolution)
        data = src.read(
            1, window=window, out_shape=out_shape, resampling=Resampling.bilinear
        )
        profile = src.profile.copy()
        profile.update(
            height=out_shape[0],
            width=out_shape[1],
            transform=rasterio.windows.transform(window, src.transform),
            dtype=data.dtype,
        )
    return data.astype(float), profile


def create_monthly_band_composite(
    month_key, band_name, monthly_index, output_dir="composites", target_resolution=20
):
    """Create a monthly median composite for a single Sentinel-2 band."""
    file_list = [
        name
        for name in monthly_index.get(month_key, [])
        if band_name in os.path.basename(name)
    ]
    if not file_list:
        return None

    arrays = []
    profile = None

    for file_name in sorted(file_list):
        try:
            band_array, band_profile = read_band_window(
                file_name, target_resolution=target_resolution
            )
            scl_blob = file_name.replace(band_name, "SCL_20m")
            scl_array, _ = read_band_window(
                scl_blob, target_resolution=target_resolution
            )
            if profile is None:
                profile = band_profile
                profile.update(dtype=rasterio.float32, count=1, nodata=NO_DATA)

            invalid_mask = get_cloud_mask(scl_array)
            band_array[invalid_mask] = np.nan
            arrays.append(band_array)
        except Exception:
            continue

    if not arrays:
        return None

    composite = np.nanmedian(np.stack(arrays, axis=0), axis=0)
    composite[np.isnan(composite)] = NO_DATA

    ensure_directory(output_dir)
    output_path = os.path.join(output_dir, f"{month_key}_{band_name}.tif")
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(composite.astype(rasterio.float32), 1)
    return output_path


def generate_monthly_composites(
    month_key, output_dir="composites", target_resolution=20
):
    """Generate monthly composites for all Sentinel-2 bands for a given month."""
    blob_names = list_raw_blobs()
    monthly_index = group_scenes_by_month(blob_names)
    files = []
    for band_name in BANDS:
        result = create_monthly_band_composite(
            month_key, band_name, monthly_index, output_dir, target_resolution
        )
        if result:
            files.append(result)
    return files


def load_composite_stack(composite_dir, feature_names=None):
    """Load a stack of composite bands from a local folder."""
    file_names = sorted([f for f in os.listdir(composite_dir) if f.endswith(".tif")])
    if feature_names is not None:
        file_names = [f for f in file_names if os.path.splitext(f)[0] in feature_names]

    arrays = []
    names = []
    profile = None
    for file_name in file_names:
        path = os.path.join(composite_dir, file_name)
        with rasterio.open(path) as src:
            arrays.append(src.read(1).astype(float))
            names.append(os.path.splitext(file_name)[0])
            if profile is None:
                profile = src.profile

    if not arrays:
        raise FileNotFoundError(f"No composite TIFF files found in {composite_dir}")

    stack = np.stack(arrays, axis=0)
    return stack, names, profile


def load_raster_stack(raster_path):
    """Load a multi-band raster from a local file path."""
    with rasterio.open(raster_path) as src:
        stack = src.read().astype(float)
        profile = src.profile
    return stack, profile


def extract_training_samples(feature_stack, label_mask):
    """Extract training samples from a feature stack using a mask raster."""
    if feature_stack.ndim != 3:
        raise ValueError("feature_stack must have shape (bands, rows, cols)")
    if feature_stack.shape[1:] != label_mask.shape:
        raise ValueError("feature_stack and label_mask must share spatial dimensions")

    indices = np.where(label_mask > 0)
    X = feature_stack[:, indices[0], indices[1]].T
    y = label_mask[indices].astype(int)
    return X, y


def compute_spectral_indices(feature_stack, feature_names):
    """Compute common vegetation indices based on a loaded feature stack."""
    band_map = {name: i for i, name in enumerate(feature_names)}
    index_arrays = []
    index_names = []

    def safe_index(a, b):
        return (a - b) / np.where((a + b) == 0, np.nan, (a + b))

    if "B08_10m" in band_map and "B04_10m" in band_map:
        index_arrays.append(
            safe_index(
                feature_stack[band_map["B08_10m"]], feature_stack[band_map["B04_10m"]]
            )
        )
        index_names.append("NDVI")
    if "B03_10m" in band_map and "B08_10m" in band_map:
        index_arrays.append(
            safe_index(
                feature_stack[band_map["B03_10m"]], feature_stack[band_map["B08_10m"]]
            )
        )
        index_names.append("NDWI")
    if "B08_10m" in band_map and "B05_20m" in band_map:
        index_arrays.append(
            safe_index(
                feature_stack[band_map["B08_10m"]], feature_stack[band_map["B05_20m"]]
            )
        )
        index_names.append("NDRE")

    if index_arrays:
        stack = np.stack(index_arrays, axis=0)
        return stack, index_names
    return None, []
