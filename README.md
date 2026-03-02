# MRI Preprocessing Pipeline

A generalized implementation of MRI preprocessing for various ML/AI tasks within the Parra Lab. This project is designed to automate the ingestion, analysis, and processing of raw DICOM MRI data into model-ready inputs.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the System](#starting-the-system)
  - [Web Control Interface](#web-control-interface)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
- [Preprocessing Workflow](#preprocessing-workflow)
- [Testing](#testing)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

The MRI Preprocessing Pipeline is a modular system built to handle large datasets of MRI scans. It runs within a Docker container to ensure a consistent environment and supports both an interactive web-based control system and a scriptable command-line interface.

The core functionality resides in `code/preprocessing/`, where a series of Python scripts handle everything from DICOM extraction to NIfTI conversion and spatial alignment.

## Key Features

- **Automated Scanning**: Recursively scans directories for MRI DICOM files.
- **Metadata Extraction**: Extracts and standardizes DICOM header information into CSV tables.
- **Intelligent Parsing**: Identifies scan types (T1, T2, etc.) and orders sequences based on acquisition times.
- **Modular Design**: Each step of the pipeline is a standalone script, allowing for flexible execution and debugging.
- **Containerized Environment**: Fully Dockerized setup for easy deployment on Linux and WSL systems.
- **Web Interface**: (In Development) A Flask-based dashboard to monitor and control the processing status.

## Project Structure

```
MRI_preprocessing/
├── code/
│   └── preprocessing/       # Core python scripts for data processing
│       ├── 01_scanDicom.py  # Scans and extracts DICOM metadata
│       ├── 02_parseDicom.py # Filters and orders scans
│       ├── ...              # Subsequent processing steps
│       ├── DICOM.py         # DICOM handling utilities
│       └── toolbox.py       # General helper functions
├── control_system/          # Docker and Web App configuration
│   ├── app/                 # Flask web application
│   └── docker*              # Docker Compose files
├── data/                    # Data storage (mounted volumes)
├── test/                    # Unit and integration tests
├── start_control.sh         # Main entry point script
└── install.py               # Dependency installation script
```

## Installation

### Prerequisites
- Linux or Windows Subsystem for Linux (WSL2)
- Python 3.x
- Docker & Docker Compose (installed automatically via `install.py` if not present)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TheParraLab/MRI_preprocessing
   cd MRI_preprocessing
   ```

2. **Install dependencies and setup Docker:**
   ```bash
   python3 install.py
   ```
   *Note: This script attempts to install Docker and configure GPU access. If you prefer, you can install Docker manually.*

## Usage

### Starting the System

The primary way to interact with the pipeline is through the `start_control.sh` script.

```bash
bash start_control.sh
```

You will be prompted to:
1.  Enable the webserver component (y/n).
2.  Provide the path to your raw DICOM data on the host machine.

The system maps your local data directory to `/FL_system/data/raw/` inside the Docker container.

### Web Control Interface
If enabled, the web interface is accessible at `http://localhost:5000`. It provides a dashboard to view the status of the preprocessing steps.
*(Note: The web interface is currently under active development).*

### Command Line Interface (CLI)
For batch processing or direct control, you can access the container's shell:

**Option 1: Convenience Script**
```bash
bash access_preprocessing.sh
```

**Option 2: Direct Docker Exec**
```bash
docker exec -it control bash
cd /FL_system/code/preprocessing/
```

## Preprocessing Workflow

The pipeline consists of numbered scripts in `code/preprocessing/` that should generally be run in order:

1.  **01_scanDicom.py**: Scans raw data and builds a `Data_table.csv` of all found DICOM files.
    *   *Documentation*: See `code/preprocessing/01_scanDicom.py` for detailed usage and arguments.
2.  **02_parseDicom.py**: Filters relevant scans (e.g., T1) and orders them by time.
3.  **03_saveNifti.py**: Converts selected DICOM series to NIfTI format.
4.  **04_saveRAS.py**: Reorients NIfTI files to RAS orientation.
5.  **05_alignScans.py**: Aligns scans to a reference volume.
6.  **06_genInputs.py**: Generates final model inputs.

To run a specific step manually inside the container:
```bash
python 01_scanDicom.py --scan_dir /FL_system/data/raw --save_dir /FL_system/data
```

## Testing

Unit and integration tests are located in the `test/` directory.

To run tests (ensure you have `pytest` installed):
```bash
pytest test/
```

## Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewFeature`).
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.

Please ensure all new code is well-documented and passes existing tests.

## Acknowledgements
- [Parra Lab](https://www.ccny.cuny.edu/bme/people/lucas-parra)
- Contributors: [Add names here]

---
*For questions or support, please contact nleotta000@citymail.cuny.edu*
