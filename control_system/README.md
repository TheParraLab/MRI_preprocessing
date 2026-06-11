# MRI Preprocessing Container

This directory contains the Docker image and compose files for the MRI preprocessing pipeline.

## Directory Structure

- `dockerfile` — Builds the preprocessing container image with:
  - Python 3 + pydicom / numpy / pandas / nibabel / scipy / yappi
  - dcm2niix for DICOM-to-NIfTI conversion
  - niftyreg for image registration
- `docker-compose.yml` — Linux compose file (uses `${NIFTI_DIRECTORY_PATH}` env var)
- `docker-compose-wsl.yml` — WSL compose file
- `startup.sh` — Container entrypoint (runs `tail` to keep container alive; preprocessing is done via `docker exec`)
- `README.md` — This file

## Usage

### Build the image

```bash
cd control_system
docker build -t mri_preprocessing .
```

### Run via docker-compose

#### Linux

```bash
export PROJECT_DIRECTORY_PATH=/path/to/project
export DATA_DIRECTORY_PATH=/path/to/raw/data
export NIFTI_DIRECTORY_PATH=/path/to/nifti/output

docker compose up --build
```

#### WSL

```bash
export PROJECT_DIRECTORY_PATH=/path/to/project
export DATA_DIRECTORY_PATH=/path/to/raw/data

docker compose -f docker-compose-wsl.yml up --build
```

### Access the container

```bash
docker exec -it control bash
cd /FL_system/code/preprocessing/
```

### Run preprocessing

```bash
python 01_scanDicom.py --scan_dir /FL_system/data/raw --save_dir /FL_system/data
```

Or run the full pipeline:

```bash
bash /FL_system/code/preprocessing/00_preprocess.sh
```

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `PROJECT_DIRECTORY_PATH` | Path to the project root on the host (mounted as `/FL_system`) | Required |
| `DATA_DIRECTORY_PATH` | Path to raw DICOM data on the host (mounted as `/FL_system/data/raw`) | Required |
| `NIFTI_DIRECTORY_PATH` | Path to NIfTI output on the host (mounted as `/FL_system/data/nifti`) | Only in `docker-compose.yml` |
