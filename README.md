# MRI Preprocessing

> This is a generalized implementation of MRI preprocesing for various ML/AI tasks within the Parra Lab

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

---

## Features

- Displays an interactive webserver for easy control and monitoring of the preprocessing process
- Takes in raw DICOM directory, analyzes its contents, and produces model inputs with little to no manual intervention
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

3. Start the application:
   ```bash
   bash start_control.sh
   ```

---

## Usage

When started with 'start_control.sh', you will be asked to provide the path to the raw data on your local system.
After this is provided, the system will start up and be accessible on port 5000 (TODO: Finalize port selection)

---

## Acknowledgements
TODO: Populate Acknowledgements
- [Tool/Library 1](https://example.com)
- [Tool/Library 2](https://example.com)

---

*Feel free to reach out if you have any questions!*
