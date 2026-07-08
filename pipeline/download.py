"""Download a cloud free Sentinel 2 composite for a given area and period using openEO.

The bands requested here match the six bands used to train the U-Net in Exercise 2
(Blue, Green, Red, Near Infrared, Shortwave Infrared 1, Shortwave Infrared 2), so a model
trained there can be applied directly to the file produced by this module.
"""

import glob

import openeo

BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def mask_clouds(scl_cube, data_cube):
    """Remove cloud, cloud shadow and cirrus pixels from a data cube using the SCL band."""
    cloud_classes = (scl_cube == 3) | (scl_cube == 8) | (scl_cube == 9) | (scl_cube == 10)
    return data_cube.mask(cloud_classes)


def download_composite(extent, temporal_extent, output_dir, backend_url="openeo.dataspace.copernicus.eu"):
    """Download a cloud free Sentinel 2 Level 2A composite and save it as a GeoTIFF.

    Parameters
    ----------
    extent: dict with keys west, south, east, north, in degrees.
    temporal_extent: list of two ISO dates, for example ["2023-06-01", "2023-08-31"].
    output_dir: folder where the downloaded GeoTIFF is placed.
    backend_url: openEO backend to connect to.

    Returns
    -------
    Path to the downloaded GeoTIFF file.
    """
    connection = openeo.connect(backend_url).authenticate_oidc()

    reflectance = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=extent,
        temporal_extent=temporal_extent,
        bands=BANDS,
        max_cloud_cover=80,
    )
    scl = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=extent,
        temporal_extent=temporal_extent,
        bands=["SCL"],
        max_cloud_cover=80,
    )

    composite = mask_clouds(scl.band("SCL"), reflectance).reduce_dimension(dimension="t", reducer="median")

    job_title = "Inference input composite"
    composite.save_result(format="GTiff").create_job(title=job_title).start_and_wait()

    job = [j for j in connection.list_jobs() if j["title"] == job_title][0]
    connection.job(job["id"]).get_results().download_files(output_dir)

    return glob.glob(f"{output_dir}/*.tif")[0]
