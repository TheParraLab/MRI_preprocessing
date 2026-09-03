"""
Shared checksum primitives for the data_checksum_analysis toolkit.

Stdlib-only, importable with no side effects. Provides hashing, tree
scanning, manifest loading, diff classification, and bounded local status
checking so the individual CLI scripts stop each carrying a divergent copy
of the same logic.

File-entry shape (additive; keeps old 'file_name' + 'md5' keys for
backward compatibility):
    {
        "file_name": str,
        "algo": "sha256" | "md5",
        "digest": str,
        "size_bytes": int,
        "md5": str,          # == digest, legacy key
    }
"""
import os
import json
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

DEFAULT_ALGO = "sha256"
DEFAULT_CHUNK = int(1 << 20)
ERROR_DIGEST = "ERROR_READING_FILE"


def _file_algo(fi, default=DEFAULT_ALGO):
    """Return the hash algorithm recorded for a single file entry, or the default."""
    return fi.get("algo") or default


def hash_file(path, algo=DEFAULT_ALGO, chunk=DEFAULT_CHUNK):
    if algo not in ("md5", "sha256"):
        raise ValueError(f"Unsupported hash algorithm: {algo!r} (use 'md5' or 'sha256').")
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_session_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue
        dirnames.sort()
        session_name = os.path.basename(dirpath)
        for file_name in sorted(filenames):
            file_path = os.path.join(dirpath, file_name)
            if os.path.isfile(file_path):
                yield session_name, file_path, file_name


def _file_entry(file_name, file_path, algo):
    entry = {
        "file_name": file_name,
        "algo": algo,
    }
    try:
        digest = hash_file(file_path, algo=algo)
        entry["digest"] = digest
        entry["size_bytes"] = os.path.getsize(file_path)
    except (OSError, PermissionError):
        entry["digest"] = ERROR_DIGEST
        entry["size_bytes"] = -1
    entry["md5"] = entry["digest"]
    return entry


def scan_tree(root, algo=DEFAULT_ALGO, skip=None, n_workers=None):
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Not a directory: {root}")
    skip_set = set(skip or [])
    start_time = datetime.now(timezone.utc)

    entries = []
    for session_name, file_path, file_name in iter_session_files(root):
        if session_name in skip_set:
            continue
        entries.append((session_name, file_path, file_name))
    entries.sort(key=lambda e: (e[0], e[2]))

    results = {}

    def process(item):
        session_name, file_path, file_name = item
        return session_name, _file_entry(file_name, file_path, algo)

    workers = n_workers if (n_workers and n_workers > 1) else None
    if workers is not None and len(entries) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(entries))) as ex:
            for session_name, entry in ex.map(process, entries):
                results.setdefault(session_name, {"files": []})["files"].append(entry)
    else:
        for session_name, entry in map(process, entries):
            results.setdefault(session_name, {"files": []})["files"].append(entry)

    for data in results.values():
        data["files"].sort(key=lambda f: f["file_name"])

    stop_time = datetime.now(timezone.utc)
    total_files = sum(len(v["files"]) for v in results.values())
    header = {
        "scan_dir": os.path.abspath(root),
        "start_time": start_time.isoformat(),
        "stop_time": stop_time.isoformat(),
        "skipped": sorted(skip_set),
        "algo": algo,
        "n_sessions": len(results),
        "n_files": total_files,
    }
    return header, results


def _normalize_file(fi):
    file_name = fi.get("file_name")
    digest = fi.get("digest")
    if digest is None:
        digest = fi.get("md5")
    md5 = fi.get("md5")
    if md5 is None:
        md5 = fi.get("digest")
    return {
        "file_name": file_name,
        "digest": digest,
        "md5": md5,
        "algo": fi.get("algo"),
        "size_bytes": fi.get("size_bytes"),
    }


def load_scan(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and ("results" in data or "header" in data):
        header = data.get("header", {}) or {}
        raw_results = data.get("results", {}) or {}
    else:
        header = {}
        raw_results = data
    results = {}
    for session, data_ in raw_results.items():
        files = data_.get("files", []) if isinstance(data_, dict) else []
        results[session] = {"files": [_normalize_file(fi) for fi in files]}
    return header, results


def build_index(results):
    index = {}
    for session, data in results.items():
        for fi in data.get("files", []):
            index[os.path.join(session, fi["file_name"])] = fi
    return index


def _digest_match(a, b):
    pa, sa = a.get("algo"), b.get("algo")
    return a.get("digest") == b.get("digest") and not (pa and sa and pa != sa)


def summarize_stats(primary_index, secondary_index):
    stats = {"identical": 0, "modified": 0, "primary_only": 0, "secondary_only": 0}
    stats["algorithm_mismatch"] = False
    for key in set(primary_index) | set(secondary_index):
        p = primary_index.get(key)
        s = secondary_index.get(key)
        if p is not None and s is not None:
            if _digest_match(p, s):
                stats["identical"] += 1
            else:
                stats["modified"] += 1
                if p.get("algo") and s.get("algo") and p["algo"] != s["algo"]:
                    stats["algorithm_mismatch"] = True
        elif p is not None:
            stats["primary_only"] += 1
        else:
            stats["secondary_only"] += 1
    return stats


def classify_sessions(primary, secondary):
    """Classify sessions given a source truth (primary) and a destination (secondary).

    Returns confirmed (safe to delete at source), stale (destination copies
    needing replacement), and missing_from_primary (destination-only sessions).
    """
    primary_sessions = sorted(primary.keys())
    primary_set = set(primary_sessions)
    secondary_index = build_index(secondary)

    confirmed = []
    stale = []
    confirmed_detail = {}

    for session in primary_sessions:
        files = primary[session].get("files", [])
        detail = []
        all_match = bool(files)
        for fi in files:
            s = secondary_index.get(os.path.join(session, fi["file_name"]))
            match = s is not None and _digest_match(fi, s)
            if not match:
                all_match = False
            detail.append(fi)
        if all_match:
            confirmed.append(session)
            confirmed_detail[session] = detail
        else:
            stale.append(session)

    return {
        "confirmed": sorted(confirmed),
        "stale": sorted(stale),
        "confirmed_detail": confirmed_detail,
        "primary_sessions": primary_sessions,
        "secondary_sessions": sorted(secondary.keys()),
        "missing_from_primary": sorted(set(secondary.keys()) - primary_set),
    }


def bounded_status(manifest_results, local_root):
    """Compare a manifest (source truth) against a local directory, hashing only
    the sessions named in the manifest.

    Returns confirmed / stale / absent plus unlisted_present (sessions present
    on disk but absent from the manifest) and per-file details.
    """
    confirmed, stale, absent = [], [], []
    details = {}

    for session in manifest_results:
        files = manifest_results[session].get("files", [])
        sdir = os.path.join(local_root, session)
        if not os.path.isdir(sdir):
            absent.append(session)
            details[session] = {
                "status": "absent",
                "files": [
                    {"file_name": fi["file_name"], "manifest_digest": fi.get("digest"),
                     "local_digest": None, "match": False}
                    for fi in files
                ],
            }
            continue

        file_details = []
        all_match = bool(files)
        for fi in files:
            algo = fi.get("algo") or DEFAULT_ALGO
            md = fi.get("digest")
            local_path = os.path.join(sdir, fi["file_name"])
            if os.path.isfile(local_path):
                try:
                    ld = hash_file(local_path, algo=algo)
                except (OSError, PermissionError):
                    ld = ERROR_DIGEST
                match = md == ld
            else:
                ld = None
                match = False
            if not match:
                all_match = False
            file_details.append({
                "file_name": fi["file_name"],
                "manifest_digest": md,
                "local_digest": ld,
                "match": match,
            })

        status = "confirmed" if all_match else "stale"
        if all_match:
            confirmed.append(session)
        else:
            stale.append(session)
        details[session] = {"status": status, "files": file_details}

    unlisted_present = []
    if os.path.isdir(local_root):
        manifest_set = set(manifest_results.keys())
        for name in sorted(os.listdir(local_root)):
            if os.path.isdir(os.path.join(local_root, name)) and name not in manifest_set:
                unlisted_present.append(name)

    return {
        "confirmed": sorted(confirmed),
        "stale": sorted(stale),
        "absent": sorted(absent),
        "unlisted_present": unlisted_present,
        "details": details,
    }


def alloc_scan_name(dir_path, base="scan_results", ext="json"):
    os.makedirs(dir_path, exist_ok=True)
    candidate = os.path.join(dir_path, f"{base}.{ext}")
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate = os.path.join(dir_path, f"{base}_{n}.{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def write_session_list(path, sessions):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        for s in sessions:
            f.write(f"{s}\n")
    return len(sessions)
