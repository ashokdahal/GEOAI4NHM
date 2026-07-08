# GEOAI4NHM

Code and instructional material for the Geoversity course **GeoAI for Natural Hazard Mapping (GEOAI4NHM)**. The course introduces the application of deep learning and geospatial artificial intelligence methods to the mapping and assessment of natural hazards, including wildfires, landslides, and floods, using open satellite data and modern segmentation architectures.

## Overview

The material is organised as a sequence of six hands-on exercises delivered as Jupyter notebooks. Each exercise builds cumulatively on the previous ones, progressing from data acquisition and preprocessing to model construction, transfer learning with foundation models, systematic evaluation, and deployment of a complete inference pipeline. Every notebook is self-contained and can be executed directly in Google Colab without any local installation, using the button provided in the table below.

## Exercises

| # | Exercise | Description | Open in Colab |
|---|----------|--------------|----------------|
| 1 | [Image Preprocessing with openEO](Excercises/01_Openeo.ipynb) | Introduces openEO for requesting and processing Sentinel-2 imagery via Python, comparing processing levels, applying band math, and constructing a cloud-free Best Available Pixel composite. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokdahal/GEOAI4NHM/blob/main/Excercises/01_Openeo.ipynb) |
| 2 | [A U-Net for Wildfire Burn Scar Mapping](Excercises/02_unet_wildfire_mapping.ipynb) | Builds a U-Net convolutional neural network from scratch and trains it on the HLS Burn Scars dataset to segment wildfire burn scars from satellite imagery. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokdahal/GEOAI4NHM/blob/main/Excercises/02_unet_wildfire_mapping.ipynb) |
| 3 | [A Vision Transformer for Landslide Mapping](Excercises/03_segformer_landslide_mapping.ipynb) | Fine-tunes SegFormer, a Vision Transformer segmentation architecture with a pre-trained Mix Transformer backbone, on the Landslide4Sense dataset. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokdahal/GEOAI4NHM/blob/main/Excercises/03_segformer_landslide_mapping.ipynb) |
| 4 | [Fine-Tuning TerraMind for Flood Mapping](Excercises/04_terramind_flood_mapping.ipynb) | Fine-tunes TerraMind, a geospatial foundation model pre-trained jointly on Sentinel-1 radar, Sentinel-2 optical, and elevation data, for flood extent mapping. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokdahal/GEOAI4NHM/blob/main/Excercises/04_terramind_flood_mapping.ipynb) |
| 5 | [Learning Curves, Hyperparameter Tuning and Evaluation Metrics](Excercises/05_evaluation_metrics.ipynb) | Presents model-agnostic tools for diagnosing training behaviour and comparing segmentation models, including learning curves, hyperparameter tuning, and standard evaluation metrics. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokdahal/GEOAI4NHM/blob/main/Excercises/05_evaluation_metrics.ipynb) |
| 6 | [An AI Inference Pipeline for Wildfire Mapping](Excercises/06_inference_pipeline.ipynb) | Assembles a complete inference pipeline that applies a trained model to an arbitrary geographic extent and produces a finished map as a georeferenced GeoTIFF. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokdahal/GEOAI4NHM/blob/main/Excercises/06_inference_pipeline.ipynb) |

## Getting Started

Each exercise can be run in one of two ways:

1. **Google Colab (recommended)** – Click the corresponding "Open in Colab" badge above. This provisions a free GPU-enabled runtime and requires no local setup. Any additional dependencies are installed within the notebook itself.
2. **Local execution** – Clone this repository and open the notebooks in Jupyter or JupyterLab. A CUDA-capable GPU is recommended for Exercises 2 through 6, as these involve training or fine-tuning deep learning models.

```bash
git clone https://github.com/ashokdahal/GEOAI4NHM.git
cd GEOAI4NHM/Excercises
```

## Repository Structure

```
GEOAI4NHM/
├── Excercises/     Jupyter notebooks for Exercises 1-6
├── pipeline/        Supporting Python modules for data download, tiling,
│                     mosaicking, model definition, and inference used by
│                     the exercises (notably Exercise 6)
└── README.md
```

## Prerequisites

Participants are expected to have a working knowledge of Python and basic familiarity with array-based scientific computing (e.g. NumPy). No prior experience with deep learning or remote sensing is assumed; the necessary concepts are introduced progressively across the exercises.
