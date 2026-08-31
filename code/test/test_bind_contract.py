"""
Bind-list regression test (change 6) for the HPC path contract.

The pipeline's in-container writable targets are scattered across the code
(toolbox guard defaults, 03_saveNifti SAVE_DIR, DICOM tmp_save, the logger
LOG_DIR default).  start_control.sh and docker-compose.yml each bind a fixed
set of host dirs onto in-container paths.  If the two ever drift apart, or if
a bind is removed, a read-only SIF run dies with a confusing OSError instead
of running.

This test does not run a container (CI has none).  Instead it statically
asserts the *contract* both launch files must satisfy:

  1. Every launch file binds a writable base at /FL_system/data, which covers
     all top-level writes (Data_table*.csv, step-03 progress pickles) and the
     /FL_system/data/tmp scratch dir the DICOM split step and logger use.
  2. Every launch file binds /deployment (logs).
  3. The five per-purpose output dirs (nifti/RAS/coreg/inputs) plus raw are
     bound in BOTH files, and the sets match exactly (no silent drift).

Run with: pytest code/test/test_bind_contract.py -v
"""

import re
from pathlib import Path

import pytest

proj_root = Path(__file__).resolve().parents[2]
START_CONTROL = proj_root / "start_control.sh"
COMPOSE = proj_root / "control_system" / "docker-compose.yml"

# In-container output subdirs (relative to /FL_system/data) the pipeline
# writes to.  raw is read-mostly but must still be a bind in both launchers.
OUTPUT_SUBDIRS = ["raw", "nifti", "RAS", "coreg", "inputs"]

# Container-side path extracted from a "HOST:/CONTAINER" bind token.
_BIND_RE = re.compile(r":\s*(/[A-Za-z0-9_./-]+)")


def _container_bind_targets(path: Path) -> list:
    """Return the sorted, de-duplicated container-side bind paths in a file."""
    text = path.read_text()
    targets = set()
    for m in _BIND_RE.finditer(text):
        p = m.group(1).rstrip("/")
        if p:
            targets.add(p)
    return sorted(targets)


def _norm(p: str) -> str:
    return p.rstrip("/")


@pytest.fixture(scope="module")
def sc_targets():
    assert START_CONTROL.exists(), f"missing {START_CONTROL}"
    return _container_bind_targets(START_CONTROL)


@pytest.fixture(scope="module")
def compose_targets():
    assert COMPOSE.exists(), f"missing {COMPOSE}"
    return _container_bind_targets(COMPOSE)


def test_base_bind_present_in_start_control(sc_targets):
    assert _norm("/FL_system/data") in sc_targets, (
        "start_control.sh must bind a writable base for /FL_system/data "
        "(covers Data_table*.csv, step-03 pickles, /FL_system/data/tmp)."
    )


def test_base_bind_present_in_compose(compose_targets):
    assert _norm("/FL_system/data") in compose_targets, (
        "docker-compose.yml must bind a writable base for /FL_system/data."
    )


def test_deployment_logs_bind_in_start_control(sc_targets):
    assert _norm("/deployment") in sc_targets, (
        "start_control.sh must bind a writable dir for /deployment (logs)."
    )


def test_deployment_logs_bind_in_compose(compose_targets):
    assert _norm("/deployment") in compose_targets, (
        "docker-compose.yml must bind a writable dir for /deployment (logs)."
    )


@pytest.mark.parametrize("sub", OUTPUT_SUBDIRS)
def test_output_dir_bound_in_start_control(sc_targets, sub):
    assert _norm(f"/FL_system/data/{sub}") in sc_targets, (
        f"start_control.sh must bind /FL_system/data/{sub}."
    )


@pytest.mark.parametrize("sub", OUTPUT_SUBDIRS)
def test_output_dir_bound_in_compose(compose_targets, sub):
    assert _norm(f"/FL_system/data/{sub}") in compose_targets, (
        f"docker-compose.yml must bind /FL_system/data/{sub}."
    )


def test_two_launch_files_agree_on_data_paths(sc_targets, compose_targets):
    """The two launchers must not drift apart on the in-container data paths.

    Compare only the /FL_system/data subtree so unrelated env-specific binds
    (e.g. GPU device nodes) do not cause a spurious failure."""

    def data_subtree(targets):
        return {
            _norm(p) for p in targets
            if _norm(p) == "/FL_system/data" or _norm(p).startswith("/FL_system/data/")
        }

    sc = data_subtree(sc_targets)
    c = data_subtree(compose_targets)
    missing_in_sc = c - sc
    missing_in_c = sc - c
    assert not missing_in_sc and not missing_in_c, (
        "start_control.sh and docker-compose.yml disagree on in-container data "
        f"binds. Only in compose: {sorted(missing_in_sc)}; "
        f"only in start_control.sh: {sorted(missing_in_c)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
