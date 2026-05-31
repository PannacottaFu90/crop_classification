import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import generic_filter
import rasterio


def majority_filter(class_map, window_size=3):
    """Apply a majority-mode filter to a classification map."""
    if class_map.ndim != 2:
        raise ValueError("class_map must be a 2D array")

    def _mode(arr):
        values = arr.astype(int)
        counts = np.bincount(values)
        return np.argmax(counts)

    return generic_filter(
        class_map, function=_mode, size=window_size, mode="nearest"
    ).astype(class_map.dtype)


def build_colormap(unique_codes, palette="tab20"):
    """Build a ListedColormap and legend entries for unique class codes."""
    cmap = plt.get_cmap(palette, len(unique_codes))
    colors = [mcolors.to_hex(cmap(i)) for i in range(len(unique_codes))]
    return colors


def plot_classification_map(
    class_map, code_to_name=None, output_path=None, title="Classification map"
):
    """Save or display a classification map with an optional legend."""
    mask = np.zeros_like(class_map, dtype=float)
    mask[class_map > 0] = class_map[class_map > 0]
    unique_codes = np.unique(class_map[class_map > 0])
    colors = build_colormap(unique_codes)
    cmap = mcolors.ListedColormap(colors)

    display_map = np.zeros_like(class_map, dtype=float)
    for i, code in enumerate(unique_codes):
        display_map[class_map == code] = i + 1
    display_map[class_map == 0] = np.nan

    plt.figure(figsize=(12, 9))
    plt.imshow(display_map, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.title(title)

    if code_to_name is not None:
        legend_patches = [
            mpatches.Patch(color=colors[i], label=code_to_name.get(code, str(code)))
            for i, code in enumerate(unique_codes)
        ]
        plt.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=9,
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_raster(array, profile, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array.astype(profile.get("dtype", array.dtype)), 1)
