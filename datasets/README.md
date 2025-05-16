# Dataset Creation Guide

This repository contains multiple datasets. **Each dataset lives in its own folder** under the `datasets/` directory and provides a `build.py` script that generates all required files from the raw sources.

---

## Prerequisites

- A *virtual environment* (recommended) created for the repo
- Any additional dataset‑specific dependencies listed in that dataset’s `requirements.txt`

```bash
# Example: create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate 
```


## Quick Start

```bash
# 1. Move into the dataset folder you want to build
cd datasets/<dataset_name>

# 2. Run the build script
python build.py 
```

## Dataset-specific exceptions

### acs

Requires setting the `key` variable in `line 23` in order to download the original data.

### education
Requires manually downloading the raw file from `https://nces.ed.gov/ipeds/SummaryTables/DownloadExcel/3601_2023_1_1_5ed94380-cc4e-494e-aae3-3293fcdde3c4` and naming it: `../data/education.xlsx`.


### nhanes
This dataset relies on an R preprocessing pipeline, expressed in a `nhanes.R` script rather than the usual `build.py`. We ship a ready‑to‑use `Docker`.

```bash
# Build the image (from the repository root)
docker build -t nhanes-r-env -f datasets/nhanes/Dockerfile datasets/nhanes

# Run the build script inside the container
docker run --rm -v $(pwd)/datasets/nhanes:/workspace nhanes-r-env Rscript nhanes.R
```
