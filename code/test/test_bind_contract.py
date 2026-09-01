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


# ---------------------------------------------------------------------------
# Env-flag regression guard (root cause of the 'could not open image' failure)
# ---------------------------------------------------------------------------
#
# Classic Singularity 3.x / Apptainer define two DIFFERENT flags:
#   -e, --env            boolean: pass all host environment variables
#   --env KEY=VALUE      value-taking: set one specific variable
#
# A value-taking call written as `-e KEY=VALUE` therefore causes the runtime
# to parse KEY=VALUE as the image path, producing the user-visible error:
#   "could not open image ...DATA_DIRECTORY_PATH=/hpc/..."
#
# Pin the launcher to the safe form --env KEY=VALUE and forbid the broken
# short-form `-e KEY=VALUE` pattern so a future refactoring cannot silently
# regress this.
_SHORT_E_ENV_RE = re.compile(r"(^|\s)-e\s+[A-Za-z_][A-Za-z0-9_]*=")
_LONG_E_ENV_RE = re.compile(r"--env\s+[A-Za-z_][A-Za-z0-9_]*=")


def test_start_control_uses_long_env_flag_not_short_e():
    """start_control.sh must use --env KEY=VALUE, not -e KEY=VALUE."""
    assert START_CONTROL.exists(), f"missing {START_CONTROL}"
    text = START_CONTROL.read_text()
    # Exclude comment lines — the comment block above the EnvFlags array
    # explains the rule and legitimately mentions the `-e KEY=VALUE` pattern.
    code_only = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    bad_in_code = re.findall(_SHORT_E_ENV_RE, code_only)
    assert not bad_in_code, (
        "start_control.sh uses the short -e flag which is boolean on "
        "classic Singularity 3.x / Apptainer. Use --env KEY=VALUE instead; "
        "otherwise the runtime parses the KEY=VALUE token as the image path "
        "and reports 'could not open image'."
    )


def test_start_control_has_env_flags_for_pipeline_vars():
    """The pipeline variables must still be passed into the container via --env.

    The pipeline inside the SIF currently relies on LOG_DIR (and, in the
    future, on explicit *_DIRECTORY_PATH overrides for debugging); if these
    are ever removed entirely, the SIF runs with host env unset and the
    logger falls back silently, masking configuration errors.
    """
    assert START_CONTROL.exists(), f"missing {START_CONTROL}"
    text = START_CONTROL.read_text()
    code_only = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )
    required = [
        re.compile(r"--env\s+LOG_DIR="),
    ]
    for pat in required:
        assert pat.search(code_only), (
            f"start_control.sh must pass '{pat.pattern}' into the "
            f"container; the container logger (toolbox.get_log_dir) expects "
            f"a LOG_DIR env var."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
