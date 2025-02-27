# MRI Preprocessing

> This is a generalized implementation of MRI preprocesing for various ML/AI tasks within the Parra Lab

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

---

## Features

- Displays an interactive webserver for easy control of the preprocessing process
- Takes in raw DICOM directory, analyzes its contents, and produces model inputs with little to no manual intervention
- DICOM headers will be scaned and parsed during processing, a full list of necessary DICOM attributes will be provided in this README at a future date.
---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TheParraLab/MRI_preprocessing
   cd MRI_preprocessing
   ```

2. Install dependencies:
   ```bash
   python3 install.py
   ```
   Note: This installation script works to install docker and configure it to have access to the GPU for ML applications.  For preprocessing alone, GPU access is not required, but docker can be installed manually.

3. Start the application:
   ```bash
   bash start_control.sh
   ```

---

## Usage

When started with 'start_control.sh', you will be asked to provide the path to the raw data on your local system. This supplied directory will be placed at /FL_system/data/raw/ within the docker container.
After this is provided, the system will start up and be accessible on port 5000 (Note: The current webserver is under development and not finalized, it should not be exposed outside the network)

### Preprocessing

The code to perform the preprocessing is provided in /code/preprocessing.  The 00_preprocess.sh script will run all preprocessing steps in series, placing fully processed data into /data/inputs.
When required, individual preprocessing scripts can be run by accessing the CLI of the container, and running `python3 0X_script.py` from within the /FL_system/code/preprocessing/ directory.  Python scripts will be modified to take in parameters as command line arguments in the near future.
---

## Acknowledgements
TODO: Populate Acknowledgements
- [Tool/Library 1](https://example.com)
- [Tool/Library 2](https://example.com)

---

*Feel free to reach out if you have any questions or suggestions!*
nleotta000@citymail.cuny.edu
