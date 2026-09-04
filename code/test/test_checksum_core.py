"""Unit tests for checksum_core (pure stdlib)."""
import hashlib
import json
import os
import sys

import pytest

_TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "tools", "data_checksum_analysis")
sys.path.insert(0, _TOOLS_DIR)
import checksum_core


def _write(path, content: bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def _write_session(results_root, session, files):
    """files: dict {file_name: content_bytes}"""
    session_dir = os.path.join(str(results_root), session)
    for name, content in files.items():
        _write(os.path.join(session_dir, name), content)


def _results_dict(root, sessions_files):
    """Build a results mapping in-memory by scanning files (no hashing)."""
    results = {}
    for session, files in sessions_files.items():
        result_files = []
        for name, content in files.items():
            result_files.append({
                "file_name": name,
                "digest": hashlib.sha256(content).hexdigest(),
                "md5": hashlib.sha256(content).hexdigest(),
                "algo": "sha256",
                "size_bytes": len(content),
            })
        results[session] = {"files": result_files}
    return results


def test_alloc_scan_name_returns_base_first_when_free(tmp_path):
    out = checksum_core.alloc_scan_name(str(tmp_path), base="scan_results", ext="json")
    assert os.path.basename(out) == "scan_results.json"
    assert out == os.path.join(str(tmp_path), "scan_results.json")
    assert not os.path.exists(out)

def test_alloc_scan_name_loops_until_free(tmp_path):
    checksum_core.alloc_scan_name(str(tmp_path), base="scan_results", ext="json")
    _write(os.path.join(str(tmp_path), "scan_results.json"), b"")
    _write(os.path.join(str(tmp_path), "scan_results_1.json"), b"")
    out = checksum_core.alloc_scan_name(str(tmp_path), base="scan_results", ext="json")
    assert os.path.basename(out) == "scan_results_2.json"

def test_hash_file_sha256_matches_hashlib(tmp_path):
    payload = b"hello world; checksum core; 1234567890"
    p = os.path.join(str(tmp_path), "data.bin")
    _write(p, payload)
    assert checksum_core.hash_file(p, algo="sha256") == hashlib.sha256(payload).hexdigest()

def test_hash_file_chunking_does_not_change_result(tmp_path):
    payload = os.urandom(5 * (1 << 20))
    p = os.path.join(str(tmp_path), "big.bin")
    _write(p, payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert checksum_core.hash_file(p, algo="sha256", chunk=1 << 20) == expected
    assert checksum_core.hash_file(p, algo="sha256", chunk=4096) == expected
    assert checksum_core.hash_file(p, algo="sha256", chunk=7) == expected

def test_hash_file_md5(tmp_path):
    p = os.path.join(str(tmp_path), "m.bin")
    _write(p, b"abc")
    assert checksum_core.hash_file(p, algo="md5") == hashlib.md5(b"abc").hexdigest()

def test_hash_file_rejects_bad_algo(tmp_path):
    p = os.path.join(str(tmp_path), "x.bin")
    _write(p, b"")
    with pytest.raises(ValueError):
        checksum_core.hash_file(p, algo="nope")

def test_iter_session_files_skips_top_level_and_sorts(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "top-level.txt"), b"top")
    _write(os.path.join(root, "S1", "b.txt"), b"b")
    _write(os.path.join(root, "S1", "a.txt"), b"a")
    items = list(checksum_core.iter_session_files(root))
    names = [(s, fn) for s, _, fn in items]
    assert ("S1", "a.txt") in names
    assert ("S1", "b.txt") in names
    assert not any(s == os.path.basename(root) for s, _ in names)

def test_scan_tree_shape_and_backcompat(tmp_path):
    root = str(tmp_path)
    _write_session(root, "A", {"01.nii.gz": b"a1", "02.nii.gz": b"a2"})
    _write_session(root, "B", {"zz.txt": b"b"})
    header, results = checksum_core.scan_tree(root, algo="sha256")
    assert set(results.keys()) == {"A", "B"}
    assert header["algo"] == "sha256"
    assert header["n_sessions"] == 2
    assert header["n_files"] == 3
    assert header["skipped"] == []
    for session, data in results.items():
        for entry in data["files"]:
            assert set(entry.keys()) >= {"file_name", "algo", "digest", "size_bytes", "md5"}
            assert entry["md5"] == entry["digest"]
            assert entry["algo"] == "sha256"
            assert entry["size_bytes"] >= 0

def test_scan_tree_skip_excludes_session(tmp_path):
    root = str(tmp_path)
    _write_session(root, "KEEP", {"f1": b"1"})
    _write_session(root, "SKIP", {"f1": b"2"})
    _, results = checksum_core.scan_tree(root, algo="sha256", skip={"SKIP"})
    assert "KEEP" in results
    assert "SKIP" not in results

def test_load_scan_header_results_shape(tmp_path):
    p = os.path.join(str(tmp_path), "scan.json")
    payload = {
        "header": {"scan_dir": "x"},
        "results": {"S": {"files": [{"file_name": "a", "digest": "abc", "md5": "abc"}]}},
    }
    with open(p, "w") as f:
        json.dump(payload, f)
    header, results = checksum_core.load_scan(p)
    assert header == {"scan_dir": "x"}
    entry = results["S"]["files"][0]
    assert set(entry.keys()) >= {"file_name", "digest", "md5", "algo", "size_bytes"}

def test_load_scan_raw_results_shape(tmp_path):
    p = os.path.join(str(tmp_path), "scan.json")
    with open(p, "w") as f:
        json.dump({"results": {"S": {"files": [{"file_name": "a", "md5": "abc"}]}}}, f)
    header, results = checksum_core.load_scan(p)
    assert header == {}
    entry = results["S"]["files"][0]
    assert entry["digest"] == "abc" and entry["md5"] == "abc"
    assert entry["algo"] is None and entry["size_bytes"] is None

def test_load_scan_bare_results_mapping(tmp_path):
    p = os.path.join(str(tmp_path), "scan.json")
    with open(p, "w") as f:
        json.dump({"S": {"files": [{"file_name": "a", "digest": "xyz"}]}}, f)
    header, results = checksum_core.load_scan(p)
    assert header == {}
    assert results["S"]["files"][0]["file_name"] == "a"
    assert results["S"]["files"][0]["md5"] == "xyz"
    assert results["S"]["files"][0]["digest"] == "xyz"

def test_build_index_and_summarize_stats(tmp_path):
    primary = {
        "A": {"files": [
            {"file_name": "a", "digest": "same", "md5": "same", "algo": "sha256"},
            {"file_name": "b", "digest": "p_only", "md5": "p_only", "algo": "sha256"},
        ]},
    }
    secondary = {
        "A": {"files": [
            {"file_name": "a", "digest": "same", "md5": "same", "algo": "sha256"},
            {"file_name": "c", "digest": "s_only", "md5": "s_only", "algo": "sha256"},
        ]},
    }
    idx_p = checksum_core.build_index(primary)
    idx_s = checksum_core.build_index(secondary)
    assert set(idx_p) == {"A/a", "A/b"}
    assert set(idx_s) == {"A/a", "A/c"}
    stats = checksum_core.summarize_stats(idx_p, idx_s)
    assert stats["identical"] == 1
    assert stats["modified"] == 0
    assert stats["primary_only"] == 1
    assert stats["secondary_only"] == 1

def test_summarize_stats_flags_algorithm_mismatch():
    primary = {"A": {"files": [{"file_name": "a", "digest": "abc", "md5": "abc", "algo": "md5"}]}}
    secondary = {"A": {"files": [{"file_name": "a", "digest": "abc", "md5": "abc", "algo": "sha256"}]}}
    stats = checksum_core.summarize_stats(checksum_core.build_index(primary), checksum_core.build_index(secondary))
    assert stats["modified"] == 1
    assert stats["algorithm_mismatch"] is True

def test_classify_sessions_confirmed_stale_missing():
    primary = {
        "A": {"files": [
            {"file_name": "x", "digest": "aa", "md5": "aa", "algo": "sha256"},
        ]},
        "B": {"files": [
            {"file_name": "y", "digest": "bb", "md5": "bb", "algo": "sha256"},
        ]},
    }
    secondary = {
        "A": {"files": [
            {"file_name": "x", "digest": "aa", "md5": "aa", "algo": "sha256"},
        ]},
        "B": {"files": [
            {"file_name": "y", "digest": "DIFFERENT", "md5": "DIFFERENT", "algo": "sha256"},
        ]},
        "C": {"files": [
            {"file_name": "z", "digest": "cc", "md5": "cc", "algo": "sha256"},
        ]},
    }
    cls = checksum_core.classify_sessions(primary, secondary)
    assert "A" in cls["confirmed"]
    assert "B" in cls["stale"]
    assert cls["missing_from_primary"] == ["C"]
    assert [f["file_name"] for f in cls["confirmed_detail"]["A"]] == ["x"]
    assert cls["primary_sessions"] == ["A", "B"]
    assert cls["secondary_sessions"] == ["A", "B", "C"]

def test_classify_sessions_missing_file_in_secondary_is_stale():
    primary = {"A": {"files": [
        {"file_name": "x", "digest": "aa", "md5": "aa", "algo": "sha256"},
        {"file_name": "y", "digest": "bb", "md5": "bb", "algo": "sha256"},
    ]}}
    secondary = {"A": {"files": [
        {"file_name": "x", "digest": "aa", "md5": "aa", "algo": "sha256"},
    ]}}
    cls = checksum_core.classify_sessions(primary, secondary)
    assert "A" in cls["stale"]
    assert "A" not in cls["confirmed"]

def test_bounded_status_confirmed_stale_absent_unlisted(tmp_path):
    root = str(tmp_path)
    _write_session(root, "S1", {"f1": b"s1f1", "f2": b"s1f2"})
    _write_session(root, "S2", {"g1": b"s2g1"})
    _write_session(root, "EXTRA", {"ex": b"ex"})

    header, manifest_results = checksum_core.scan_tree(root, algo="sha256", skip={"EXTRA"})

    local_root = str(tmp_path / "local")
    _write_session(local_root, "S1", {"f1": b"s1f1", "f2": b"s1f2"})
    _write_session(local_root, "S2", {"g1": b"DIFFERENT"})
    _write_session(local_root, "EXTRA", {"ex": b"ex"})

    status = checksum_core.bounded_status(manifest_results, local_root)
    assert status["confirmed"] == ["S1"]
    assert status["stale"] == ["S2"]
    assert status["absent"] == []
    assert status["unlisted_present"] == ["EXTRA"]
    assert status["details"]["S1"]["status"] == "confirmed"
    assert status["details"]["S2"]["status"] == "stale"
    assert status["details"]["S2"]["files"][0]["match"] is False
    assert status["details"]["S2"]["files"][0]["local_digest"] != status["details"]["S2"]["files"][0]["manifest_digest"]

def test_bounded_status_absent_session(tmp_path):
    root = str(tmp_path)
    _write_session(root, "K", {"f": b"k"})
    _, manifest_results = checksum_core.scan_tree(root, algo="sha256")

    local_root = str(tmp_path / "local")
    os.makedirs(local_root, exist_ok=True)

    status = checksum_core.bounded_status(manifest_results, local_root)
    assert status["absent"] == ["K"]
    assert status["details"]["K"]["status"] == "absent"
    assert all(f["local_digest"] is None for f in status["details"]["K"]["files"])

def test_bounded_status_deterministic_across_workers(tmp_path):
    root = str(tmp_path)
    _write_session(root, "M", {"a": b"A", "b": b"B", "c": b"C", "d": b"D"})
    _, manifest = checksum_core.scan_tree(root, algo="sha256")

    local = str(tmp_path / "local")
    _write_session(local, "M", {"a": b"A", "b": b"B", "c": b"C", "d": b"D"})

    serial = checksum_core.bounded_status(manifest, local, n_workers=1)
    parallel_a = checksum_core.bounded_status(manifest, local, n_workers=4)
    parallel_b = checksum_core.bounded_status(manifest, local, n_workers=8)
    default = checksum_core.bounded_status(manifest, local)

    assert serial["details"] == parallel_a["details"]
    assert parallel_a["details"] == parallel_b["details"]
    assert parallel_a["details"] == default["details"]
    # files stay in manifest order
    names = [f["file_name"] for f in parallel_a["details"]["M"]["files"]]
    assert names == ["a", "b", "c", "d"]
    assert parallel_a["confirmed"] == ["M"]


def test_hash_session_files_missing_file_reports_none(tmp_path):
    # manifest lists 3 files, only 2 exist locally
    files = [
        {"file_name": "a", "digest": "aa" * 32, "algo": "sha256"},
        {"file_name": "b", "digest": "bb" * 32, "algo": "sha256"},
        {"file_name": "gone", "digest": "cc" * 32, "algo": "sha256"},
    ]
    out = checksum_core._hash_session_files(files, str(tmp_path), n_workers=8)
    assert [f["file_name"] for f in out] == ["a", "b", "gone"]
    assert all(f["local_digest"] is None for f in out)
    assert all(f["match"] is False for f in out)


def test_write_session_list(tmp_path):
    p = str(tmp_path / "out" / "sessions.txt")
    count = checksum_core.write_session_list(p, ["b", "a"])
    assert count == 2
    assert open(p).read() == "b\na\n"
