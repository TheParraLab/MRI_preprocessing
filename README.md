# MRI Preprocessing Pipeline

A modular pipeline for automated MRI DICOM preprocessing. Converts raw DICOM MRI data into model-ready inputs through a series of numbered processing steps.

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Container](#starting-the-container)
  - [Direct Container Access](#direct-container-access)
  - [Running Preprocessing Steps](#running-preprocessing-steps)
- [Preprocessing Workflow](#preprocessing-workflow)
- [Testing](#testing)
- [TODO / Roadmap](#todo--roadmap)

## Key Features

- **Automated DICOM Scanning**: Recursively scans directories for MRI DICOM files and extracts metadata.
- **Intelligent Parsing**: Identifies scan types, filters artifacts, and orders sequences by acquisition time.
- **NIfTI Conversion**: Converts DICOM series to NIfTI format using dcm2niix.
- **Spatial Alignment**: Coregisters scans to a reference volume.
- **Modular Design**: Each pipeline step is an independent script that can be run manually or in sequence.
- **Containerized**: Docker image with all dependencies pre-installed (Python, pydicom, nibabel, niftyreg, dcm2niix).

## Project Structure

```
MRI_preprocessing/
├── code/
│   └── preprocessing/       # Core Python preprocessing scripts
│       ├── 01_scanDicom.py  # Scan DICOM files and extract metadata
│       ├── 02_parseDicom.py # Filter and order scans
│       ├── 03_saveNifti.py  # Convert DICOM to NIfTI
│       ├── 04_saveRAS.py    # Reorient to RAS
│       ├── 05_alignScans.py # Coregister scans
│       ├── 06_genInputs.py  # Generate model inputs
│       ├── DICOM.py         # DICOM handling utilities
│       ├── toolbox.py       # Shared helper functions
│       └── 00_preprocess.sh # Run full pipeline
├── control_system/          # Docker image and compose files
│   ├── dockerfile           # Container image definition
│   ├── docker-compose.yml   # Linux compose file
│   ├── docker-compose-wsl.yml  # WSL compose file
│   ├── startup.sh           # Container entrypoint
│   └── README.md            # Container documentation
├── code/test/               # Unit and integration tests
├── docs/                    # Code reviews and improvement recommendations
├── start_control.sh         # Container startup script
├── install.py               # Docker + NVIDIA toolkit installer (Linux)
├── mount_kirbyPro.sh        # Machine-specific mount script
├── requirements.txt         # Python runtime dependencies
└── requirements-dev.txt     # Development/testing dependencies
```

## Installation

### Prerequisites

- Linux or WSL2
- Python 3.10+
- NVIDIA GPU (for preprocessing acceleration)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TheParraLab/MRI_preprocessing
   cd MRI_preprocessing
   ```

2. **Install Docker and NVIDIA Container Toolkit:**
   ```bash
   sudo python3 install.py
   ```
   *This installs Docker, configures GPU access, and verifies the setup.*

## Usage

### Starting the Container

1. Copy `.env.example` to `.env` and fill in all required paths:

```bash
cp .env.example .env
# Edit .env with your deployment paths
```

2. Run the startup script:

```bash
bash start_control.sh
```

The script auto-detects Docker, Singularity/Apptainer, or Conda and starts accordingly. Each run creates a timestamped deployment log in `deployments/`. With Docker, each invocation gets a unique container name (`control-<timestamp>`), so **multiple containers can run concurrently**.

### Direct Container Access

Find your container name with `docker ps | grep mri_control-`, then attach:

```bash
docker exec -it <container_name> bash
```

Navigate to `/FL_system/code/preprocessing/` to run preprocessing scripts.

### Running Preprocessing Steps

Each step can be run manually:

```bash
# Step 1: Scan DICOM files
python 01_scanDicom.py --scan_dir /FL_system/data/raw --save_dir /FL_system/data

# Step 2: Parse and filter
python 02_parseDicom.py --save_dir /FL_system/data

# Full pipeline:
bash /FL_system/code/preprocessing/00_preprocess.sh
```

## Preprocessing Workflow

The pipeline consists of numbered scripts that should generally be run in order:

1. **01_scanDicom.py** — Scans raw DICOM data, extracts metadata, produces `Data_table.csv`
2. **02_parseDicom.py** — Filters scans (removes T2, DWI, computed images), orders by trigger time, produces `Data_table_timing.csv`
3. **03_saveNifti.py** — Converts selected DICOM series to NIfTI format using dcm2niix; after conversion it runs a post-conversion audit comparing the nifti directory against `Data_table_timing.csv` (missing / extra / duplicate-Major / ghost sessions) and writes `nifti_audit.json` to the deployment log dir (pure audit, never aborts)
4. **04_saveRAS.py** — Reorients NIfTI files to RAS orientation
5. **05_alignScans.py** — Coregisters all scans to a reference volume
6. **06_genInputs.py** — Generates numpy inputs for model training

Intermediate outputs:
- `/FL_system/data/Data_table.csv` — DICOM metadata table (step 01 output)
- `/FL_system/data/Data_table_timing.csv` — Filtered and ordered table (step 02 output)
- `/FL_system/data/nifti/` — NIfTI files (step 03 output)
- `/FL_system/data/RAS/` — RAS-oriented NIfTI files (step 04 output)
- `/FL_system/data/coreg/` — Coregistered scans (step 05 output)
- `/FL_system/data/inputs/` — Final model inputs (step 06 output)

## Testing

```bash
# Run all tests
pytest -v

# Run unit tests only (fastest)
pytest code/test/test_scanDicom_unit.py -v

# Run comprehensive tests
pytest code/test/test_scanDicom_full.py -v

# Run deterministic known-result tests
pytest code/test/test_synthetic_known_result.py -v
```

Test coverage for `01_scanDicom.py` is comprehensive (89 tests). See `code/test/TESTS.md` for the full test suite documentation.

## TODO / Roadmap

- **`02_parseDicom.py --multi` hangs after logging futures completed** — When `--multi` is passed, the script stops outputting logs after printing a handful of "Future N completed successfully" lines and remains stuck for hours. Works fine in serial mode (without `--multi`). Root cause suspected to be logger contention (`FileHandlerWithLock` / `_init_child_logger` / `QueueListener`) under `ProcessPoolExecutor`. Multiprocessing has been disabled until this is resolved.

### HIGH PRIORITY

- ~~**Self-contained Docker image with code baked in**~~ (done) — Code is `COPY`'d into the image at build time and CUDA-enabled NiftyReg is compiled at build time (`CHECK_GPU=OFF` skips the configure-time GPU probe so the build works on GPU-less machines/CI; builds for sm_60–sm_86 + PTX). Containers start immediately — no first-run compile — which also makes Singularity/Apptainer pulls work on read-only container filesystems.
- ~~**Per-deployment script logging**~~ (done) — `toolbox.get_log_dir()` now resolves the log directory from the `LOG_DIR` environment variable, which `start_control.sh` sets to `/deployment/logs` for Docker and Singularity runs (bound to `deployments/<deployment-id>/logs/` on the host) and to a host-local `deployments/<deployment-id>/logs/` for bare Conda runs. Manual/local runs fall back to `<repo_root>/logs`. All six pipeline steps use this helper, so every deployment produces its own isolated log set under `deployments/<id>/`.
- ~~**Post-conversion NIfTI audit (step 03)**~~ (done) — 03_saveNifti.py now audits the nifti directory against Data_table_timing.csv (missing/extra/duplicate-Major/ghost), writes deployments/<id>/logs/nifti_audit.json, never aborts.

### MEDIUM PRIORITY

- **Update HPC Singularity path** — The current workflow expects users to manually build Singularity images from `.def` files (which is broken on modern clusters). Move toward pushing the Docker image to an internal registry and allowing Apptainer/Singularity to pull directly from there.
