"""Put per tile predictions back together into one map and save it as a GeoTIFF."""

import numpy as np
import rasterio


def stitch_tiles(predictions, offsets, tile_size, padded_shape):
    """Place every tile prediction back at its original pixel location.

    Because the tiles produced by split_into_tiles do not overlap, each prediction can be
    written directly into its slot without any blending between neighbouring tiles.
    """
    mosaic = np.zeros(padded_shape, dtype=np.uint8)

    for prediction, (row, col) in zip(predictions, offsets):
        mosaic[row:row + tile_size, col:col + tile_size] = prediction

    return mosaic


def save_prediction_as_geotiff(prediction, profile, output_path):
    """Save a single band class map as a GeoTIFF, reusing the CRS and transform of the source image."""
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8")

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(prediction.astype("uint8"), 1)
