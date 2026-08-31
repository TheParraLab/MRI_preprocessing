"""
Unit tests for toolbox.ensure_dir_writable() — the read-only SIF guard used
by step 02 (DICOM.split) and step 03 (saveNifti) to create dirs under a
user-controlled path that may land on the squashfs build layer.

Coverage goals
--------------
1. Writable target path → created (and usable).
2. Already-existing target → no-op.
3. Non-creatable target (chmod-555 read-only ancestor) → raises
   RuntimeError with the launcher hint, NOT a raw OSError.
4. The auto-derived bind hint (_derive_bind_hint) produces the canonical
   $PWD/mri_data_base/<sub>:/FL_system/data/<sub> form for paths under
   the writable base, and no bind line for paths outside it.

Run with: pytest code/test/test_rofs_guard.py -v
"""

import os
import sys
import shutil
import tempfile
import unittest

from pathlib import Path

proj_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(proj_root / "code" / "preprocessing"))
from toolbox import ensure_dir_writable, _derive_bind_hint
from DICOM import ensure_dir_writable as dicom_ensure_dir_writable

# DICOM must re-export the same helper so the call site in DICOM.split
# and the (future) call site in 03_saveNifti both resolve to one impl.
assert ensure_dir_writable is dicom_ensure_dir_writable, \
    "DICOM must re-export the same helper from toolbox"


@unittest.skipIf(sys.platform == "win32", "chmod-based read-only simulation is POSIX-only")
class TestEnsureDirWritable(unittest.TestCase):

    def test_creates_writable_dir(self):
        root = tempfile.mkdtemp(prefix="rwguard_w_")
        try:
            target = os.path.join(root, "a", "b", "c")
            ensure_dir_writable(target, context="test")
            self.assertTrue(os.path.isdir(target))
            probe = os.path.join(target, "probe.txt")
            with open(probe, "w") as f:
                f.write("ok")
            self.assertTrue(os.path.exists(probe))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_idempotent_on_existing_dir(self):
        root = tempfile.mkdtemp(prefix="rwguard_e_")
        target = os.path.join(root, "already_there")
        os.makedirs(target)
        try:
            ensure_dir_writable(target, context="test")
            self.assertTrue(os.path.isdir(target))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_read_only_ancestor_raises_runtime_error_not_oserror(self):
        root = tempfile.mkdtemp(prefix="rwguard_ro_")
        os.chmod(root, 0o555)
        try:
            target = os.path.join(root, "impossible", "dir")
            with self.assertRaises(RuntimeError) as cm:
                ensure_dir_writable(target, context="scan results cache")
            msg = str(cm.exception)
            self.assertIn(target, msg, "message must name the failed path")
            self.assertIn("read-only", msg.lower(), "message must say read-only")
            self.assertIn("start_control.sh", msg, "message must point at the launcher")
            # the original OSError must be preserved via __cause__
            self.assertIsInstance(cm.exception.__cause__, OSError)
            # RuntimeError is NOT a subclass of OSError, so this also asserts
            # we did NOT re-raise / leak a bare OSError
            self.assertFalse(issubclass(type(cm.exception), OSError))
        finally:
            os.chmod(root, 0o755)
            shutil.rmtree(root, ignore_errors=True)

    def test_existing_dir_under_read_only_ancestor_is_noop(self):
        """If the target already exists, no makedirs is attempted, so the
        guard is a no-op even though the ancestor is read-only."""
        root = tempfile.mkdtemp(prefix="rwguard_pre_")
        target = os.path.join(root, "pre_existing")
        os.makedirs(target)
        os.chmod(root, 0o555)
        try:
            ensure_dir_writable(target, context="test")  # must not raise
            self.assertTrue(os.path.isdir(target))
        finally:
            os.chmod(root, 0o755)
            shutil.rmtree(root, ignore_errors=True)


class TestDeriveBindHint(unittest.TestCase):
    """The bind hint is the part actually useful to the HPC user — make sure
    it maps an in-container path to the canonical host-side base path."""

    def test_data_base_itself(self):
        hint = _derive_bind_hint('/FL_system/data')
        self.assertIn('--bind "$PWD/mri_data_base:/FL_system/data"', hint)

    def test_nifti_subdir(self):
        hint = _derive_bind_hint('/FL_system/data/nifti/')
        self.assertIn('--bind "$PWD/mri_data_base/nifti:/FL_system/data/nifti"', hint)

    def test_tmp_directory_scan(self):
        hint = _derive_bind_hint('/FL_system/data/tmp/Directory_Scan/')
        self.assertIn('--bind "$PWD/mri_data_base/tmp/Directory_Scan:/FL_system/data/tmp/Directory_Scan"', hint)

    def test_path_outside_base_has_no_bind_line(self):
        self.assertEqual(_derive_bind_hint('/some/other/path'), '')

    def test_empty_path(self):
        self.assertEqual(_derive_bind_hint(''), '')


if __name__ == "__main__":
    unittest.main()
