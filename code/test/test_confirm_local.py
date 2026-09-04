"""End-to-end tests for the confirm_local CLI against a source-truth manifest."""
import json
import os
import sys

_TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "tools", "data_checksum_analysis")
sys.path.insert(0, _TOOLS_DIR)

import checksum_core
import confirm_local


def _write(path, content: bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def _write_session(root, session, files):
    session_dir = os.path.join(str(root), session)
    for name, content in files.items():
        _write(os.path.join(session_dir, name), content)


def _read_lines(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _build_manifest_and_local(tmp_path):
    source_root = tmp_path / "source"
    src = str(source_root)
    _write_session(src, "P1", {"01.nii": b"p1a", "02.nii": b"p1b"})
    _write_session(src, "P2", {"01.nii": b"p2"})

    manifest_out = tmp_path / "manifest.json"
    header, results = checksum_core.scan_tree(src, algo="sha256")
    with open(str(manifest_out), "w") as f:
        json.dump({"header": header, "results": results}, f)

    local_root = tmp_path / "local"
    _write_session(str(local_root), "P1", {"01.nii": b"p1a", "02.nii": b"p1b"})
    _write_session(str(local_root), "P2", {"01.nii": b"DIFFERENT"})
    _write_session(str(local_root), "EXTRA", {"xx": b"ex"})
    return str(manifest_out), str(local_root)


def test_confirm_local_basic(tmp_path, capsys):
    manifest, local = _build_manifest_and_local(tmp_path)
    outdir = str(tmp_path / "out")
    rc = confirm_local.main([manifest, local, "-o", outdir])
    assert rc == 0

    assert _read_lines(os.path.join(outdir, "confirmed.txt")) == ["P1"]
    assert _read_lines(os.path.join(outdir, "stale.txt")) == ["P2"]
    assert _read_lines(os.path.join(outdir, "absent.txt")) == []

    status = json.load(open(os.path.join(outdir, "manifest_status.json")))
    st = status["status"]
    assert st["confirmed"] == ["P1"]
    assert st["stale"] == ["P2"]
    assert st["absent"] == []
    assert "details" not in st  # details off by default


def test_confirm_local_list_all(tmp_path):
    manifest, local = _build_manifest_and_local(tmp_path)
    outdir = str(tmp_path / "out")
    rc = confirm_local.main([manifest, local, "--list-all", "-o", outdir])
    assert rc == 0
    assert _read_lines(os.path.join(outdir, "unlisted_present.txt")) == ["EXTRA"]


def test_confirm_local_default_outdir_is_manifest_dir(tmp_path):
    manifest, local = _build_manifest_and_local(tmp_path)
    rc = confirm_local.main([manifest, local])
    assert rc == 0
    assert os.path.isfile(os.path.join(os.path.dirname(manifest), "confirmed.txt"))
    assert os.path.isfile(os.path.join(os.path.dirname(manifest), "manifest_status.json"))


def test_confirm_local_emit_details(tmp_path):
    manifest, local = _build_manifest_and_local(tmp_path)
    outdir = str(tmp_path / "out")
    confirm_local.main([manifest, local, "--emit-details", "-o", outdir])
    status = json.load(open(os.path.join(outdir, "manifest_status.json")))
    st = status["status"]
    assert "details" in st
    assert st["details"]["P1"]["status"] == "confirmed"
    assert st["details"]["P2"]["status"] == "stale"
    for d in st["details"]["P2"]["files"]:
        assert d["match"] is False


def test_confirm_local_verify_pass(tmp_path, capsys):
    manifest, local = _build_manifest_and_local(tmp_path)
    outdir = str(tmp_path / "out")
    sessions_file = os.path.join(outdir, "confirmed.txt")
    os.makedirs(outdir, exist_ok=True)
    confirm_local.main([manifest, local, "-o", outdir])
    assert _read_lines(sessions_file) == ["P1"]

    rc = confirm_local.main([manifest, local, "-o", outdir, "--verify", sessions_file])
    assert rc == 0
    report = json.load(open(os.path.join(outdir, "verify_report.json")))
    assert report["P1"]["match"] is True

    out = capsys.readouterr().out
    assert "P1" in out and "PASS" in out


def test_confirm_local_verify_fail(tmp_path, capsys):
    manifest, local = _build_manifest_and_local(tmp_path)
    outdir = str(tmp_path / "verify_out")
    os.makedirs(outdir, exist_ok=True)
    sessions_file = os.path.join(outdir, "p2.txt")
    with open(sessions_file, "w") as f:
        f.write("P2\n")
    rc = confirm_local.main([manifest, local, "-o", outdir, "--verify", sessions_file])
    assert rc == 3
    report = json.load(open(os.path.join(outdir, "verify_report.json")))
    assert report["P2"]["match"] is False
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_confirm_local_missing_manifest(tmp_path):
    rc = confirm_local.main([str(tmp_path / "nope.json"), str(tmp_path)])
    assert rc == 2

def test_confirm_local_missing_local_dir(tmp_path):
    manifest, _ = _build_manifest_and_local(tmp_path)
    rc = confirm_local.main([manifest, str(tmp_path / "no_such_dir")])
    assert rc == 2

def test_confirm_local_verify_missing_sessions_file(tmp_path):
    manifest, local = _build_manifest_and_local(tmp_path)
    rc = confirm_local.main([manifest, local, "--verify", str(tmp_path / "no_such.txt")])
    assert rc == 2
