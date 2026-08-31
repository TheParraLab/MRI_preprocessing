"""
Unit tests for ensure_dir_writable() — the read-only SIF guard used by
DICOM.split (step 02) to create tmp_save/directory_scan/.

Coverage goals
--------------
1. A writable target path is created (and is a usable directory).
2. An already-existing target path is a no-op (idempotent).
3. A non-creatable target (read-only ancestor, chmod-555 simulation of
   the Apptainer/Singularity squashfs default) raises RuntimeError
   with the actionable bind hint, NOT a raw OSError.
4. The hint mentions the canonical writable base path and the
   start_control.sh launcher, so a user following the message can
   fix the problem without reading the code.

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
from DICOM import ensure_dir_writable


@unittest.skipIf(sys.platform == "win32", "chmod-based read-only simulation is POSIX-only")
class TestEnsureDirWritable(unittest.TestCase):

    def test_creates_writable_dir(self):
        root = tempfile.mkdtemp(prefix="rwguard_w_")
        try:
            target = os.path.join(root, "a", "b", "c")
            ensure_dir_writable(target, context="test")
            self.assertTrue(os.path.isdir(target))
            # and it is in fact writable end-to-end
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
            # must not raise; must not delete either
            ensure_dir_writable(target, context="test")
            self.assertTrue(os.path.isdir(target))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_read_only_ancestor_raises_runtime_error_with_hint(self):
        root = tempfile.mkdtemp(prefix="rwguard_ro_")
        os.chmod(root, 0o555)  # read-only: subdir creation must fail
        try:
            target = os.path.join(root, "impossible", "dir")
            with self.assertRaises(RuntimeError) as cm:
                ensure_dir_writable(target, context="scan results cache")
            msg = str(cm.exception)
            self.assertIn(target, msg, "message must name the failed path")
            self.assertIn("read-only", msg.lower(), "message must say read-only")
            self.assertIn("--bind", msg, "message must give the bind command")
            self.assertIn("mri_data_base/tmp", msg, "message must give the canonical base path")
            self.assertIn("start_control.sh", msg, "message must point at the launcher")
            self.assertIsInstance(cm.exception, RuntimeError)
            self.assertNotIsInstance(cm.exception, OSError)
            self.assertIsInstance(cm.exception.__cause__, OSError)
        finally:
            os.chmod(root, 0o755)
            shutil.rmtree(root, ignore_errors=True)

    def test_existing_dir_under_read_only_ancestor_is_ok(self):
        """If the target dir already exists, no makedirs is attempted, so the
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


if __name__ == "__main__":
    unittest.main()
