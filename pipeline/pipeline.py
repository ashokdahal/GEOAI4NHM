"""Orchestrate the full pipeline: download imagery, tile it, run inference, save a map.

This module simply calls the other modules in order. Each step can also be run on its own,
which is what the accompanying notebook does first, before showing this single entry point.
"""

import os

import numpy as np

from .download import download_composite
from .tiling import read_raster_as_array, pad_to_multiple, split_into_tiles
from .model import load_model
from .inference import run_inference_on_tiles
from .mosaic import stitch_tiles, save_prediction_as_geotiff


def run_pipeline(extent, temporal_extent, output_dir, checkpoint_path=None, tile_size=256, device="cpu"):
    """Run the full inference pipeline for one area and period of interest.

    Parameters
    ----------
    extent: dict with keys west, south, east, north, in degrees.
    temporal_extent: list of two ISO dates defining the compositing period.
    output_dir: folder used for downloaded and output files.
    checkpoint_path: optional path to trained U-Net weights, as saved in Exercise 2.
    tile_size: size in pixels of the square tiles the model is applied to.
    device: "cpu" or "cuda".

    Returns
    -------
    Path to the saved GeoTIFF prediction map.
    """
    os.makedirs(output_dir, exist_ok=True)

    scene_path = download_composite(extent, temporal_extent, output_dir=os.path.join(output_dir, "scene"))
    scene_array, profile = read_raster_as_array(scene_path)
    scene_array = np.clip(scene_array / 10000.0, 0.0, 1.0)

    padded_array, original_shape = pad_to_multiple(scene_array, tile_size)
    tiles, offsets = split_into_tiles(padded_array, tile_size)

    model = load_model(checkpoint_path=checkpoint_path, device=device)
    predictions = run_inference_on_tiles(model, tiles, device)

    padded_mosaic = stitch_tiles(predictions, offsets, tile_size, padded_array.shape[1:])
    mosaic = padded_mosaic[:original_shape[0], :original_shape[1]]

    output_path = os.path.join(output_dir, "burn_scar_map.tif")
    save_prediction_as_geotiff(mosaic, profile, output_path)

    return output_path
