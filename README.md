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
├── test/                    # Unit and integration tests
├── docs/                    # Code reviews and improvement recommendations
├── start_control.sh         # Container startup script
├── access_preprocessing.sh  # Direct CLI access to container
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

```bash
bash start_control.sh
```

You will be prompted for:
1. The path to your raw DICOM data directory
2. The path for NIfTI output

The container mounts your host directories into `/FL_system/data/raw/` and `/FL_system/data/nifti/` inside the container.

### Direct Container Access

While the container is running:

```bash
bash access_preprocessing.sh
```

This opens an interactive shell inside the container. Navigate to `/FL_system/code/preprocessing/` to run preprocessing scripts.

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
3. **03_saveNifti.py** — Converts selected DICOM series to NIfTI format using dcm2niix
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
pytest test/ -v

# Run unit tests only (fastest)
pytest test/test_scanDicom_unit.py -v

# Run comprehensive tests
pytest test/test_scanDicom_full.py -v

# Run deterministic known-result tests
pytest test/test_synthetic_known_result.py -v
```

Test coverage for `01_scanDicom.py` is comprehensive (89 tests). See `test/TESTS.md` for the full test suite documentation.
