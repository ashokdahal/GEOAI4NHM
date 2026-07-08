"""Split a large raster into fixed size tiles that a model can process one at a time.

Segmentation models are trained on small patches, for example the 512 by 512 scenes used
in Exercise 2, so applying one to a much larger area of interest requires splitting that
area into patches, running the model on each one, and putting the results back together.
This module handles the splitting side of that process. The mosaic module handles putting
the pieces back together.
"""

import numpy as np
import rasterio


def read_raster_as_array(image_path):
    """Read every band of a GeoTIFF into memory.

    Returns a (bands, height, width) float32 array together with the rasterio profile,
    which records the coordinate reference system and geographic transform needed to
    write a correctly georeferenced GeoTIFF later.
    """
    with rasterio.open(image_path) as src:
        array = src.read().astype(np.float32)
        profile = src.profile
    return array, profile


def pad_to_multiple(array, tile_size):
    """Pad an array so its height and width are each an exact multiple of tile_size.

    Padding with zeros at the bottom and right edges avoids partial tiles, which keeps the
    tiling and stitching logic simple. The original height and width are returned as well,
    so the padding can be removed again after inference.
    """
    _, height, width = array.shape
    padded_height = int(np.ceil(height / tile_size)) * tile_size
    padded_width = int(np.ceil(width / tile_size)) * tile_size

    padded = np.zeros((array.shape[0], padded_height, padded_width), dtype=array.dtype)
    padded[:, :height, :width] = array
    return padded, (height, width)


def split_into_tiles(array, tile_size):
    """Split a padded (bands, height, width) array into a grid of non overlapping tiles.

    Returns a list of tiles and a matching list of (row, col) pixel offsets, which record
    where each tile belongs in the full image so predictions can be placed back correctly.
    """
    _, height, width = array.shape
    tiles, offsets = [], []

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            tile = array[:, row:row + tile_size, col:col + tile_size]
            tiles.append(tile)
            offsets.append((row, col))

    return tiles, offsets
