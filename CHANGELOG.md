# Changelog

All notable changes to MRI_preprocessing are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

The pipeline turns breast MRI DICOM acquisitions into a coregistered, harmonized NIfTI dataset in six steps (`01_scanDicom` → `02_parseDicom` → `03_saveNifti` → `04_saveRAS` → `05_alignScans` → `06_genInputs`), deployable via Docker, conda (`environment.yml`), or a native HPC path.

## [1.0.0] - 2026-08-27

### Added
- Full 6-step pipeline with per-step logging, `--start_step` / `--stop_step` / `--steps` orchestration in `00_preprocess.sh`, and GPU-enabled NiftyReg coregistration (prebuilt CUDA `reg_f3d` baked into the image).
- Pinned, reproducible environments: `requirements.txt` / `requirements-dev.txt` (Docker & CI) and `environment.yml` (conda / bare-HPC), validated against the 1.0 container image.
- Fat-saturation detection gate (`detect_fs`) with a 21-test suite covering the protocol's ambiguous non-FS phrasings.
- Post-conversion NIfTI audit in step 03: after dcm2niix, `03_saveNifti.py` cross-checks the NIfTI tree against the timing table and writes a `nifti_audit.json` report (missing / unexpected / duplicate sessions, plus "ghost" sessions — table rows with no directory on disk). Covers the 6-test `test_saveNifti_audit.py` suite.
- Deterministic synthetic known-result tests for 01/02, plus toolbox and hybrid-path suites — 140+ tests total, run across Python 3.10–3.12 in CI.
- Per-deployment traceability: a `deployment_finalize` step (EXIT/INT/TERM trap in `start_control.sh`) records the resolved image id / `.sif` size into each deployment's `manifest.json`.
- Version string: `__version__ = "1.0.0"` in `code/__init__.py`, printed as a banner by `run_pipeline_conda.sh` and `00_preprocess.sh`, and stamped as `org.opencontainers.image.version`, `org.label-schema.version` and `org.label-schema.version-final` `1.0` OCI labels in the container image.

### Fixed
- Step-03 NIfTI audit now reports ghost sessions (table rows with no on-disk directory) instead of silently iterating the table only.

### Changed
- `02_parseDicom.py --multi`: flag is accepted but DEPRECATED and ignored; multiprocessing is disabled (see Known Issues).

### Known Issues (deferrals, tracked for 1.1)
- `02_parseDicom.py --multi` hangs after logging futures completed under `ProcessPoolExecutor`; suspected logger contention (`FileHandlerWithLock` / `QueueListener`). Regression vs pre-rewrite behavior. Target: 1.1.
- HPC Singularity/Apptainer: current documented path assumes manual `.def` builds (broken on modern clusters); move to pulling the pushed Docker image via Apptainer. Target: 1.1.

[1.0.0]: https://github.com/parra-lab/MRI_preprocessing/compare/v0.0.0...v1.0.0  (placeholder link — adjust if the remote differs)
