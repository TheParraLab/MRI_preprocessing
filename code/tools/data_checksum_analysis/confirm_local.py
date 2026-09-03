"""
Bounded local status check against the source-of-truth manifest.

Given a manifest JSON (the authoritative source scan) and a local
directory on the destination machine, hash ONLY the sessions listed in the
manifest and report which sessions are confirmed, stale, or absent locally.

Usage:
  python confirm_local.py <manifest.json> <local_dir>
                          [-o OUTDIR] [--list-all] [--emit-details]
  python confirm_local.py <manifest.json> <local_dir>
                          --verify <sessions.txt>
"""
import json
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checksum_core


def _default_outdir(manifest_path):
    return os.path.dirname(os.path.abspath(manifest_path)) or os.getcwd()


def _verify(manifest_results, local_root, session_list_path, outdir):
    if not os.path.isfile(session_list_path):
        print(f"Error: {session_list_path} not found.", file=sys.stderr)
        return 2
    with open(session_list_path, "r", encoding="utf-8") as f:
        sessions = [line.strip() for line in f if line.strip()]
    if not sessions:
        print(f"No sessions listed in {session_list_path}.", file=sys.stderr)
        return 0

    report = {}
    any_fail = False
    width = max([len(s) for s in sessions] + [0])
    for session in sessions:
        entry = manifest_results.get(session)
        if entry is None:
            report[session] = {"match": False, "files": [
                {"file_name": None, "manifest_digest": None, "local_digest": None, "match": False, "note": "not in manifest"}
            ]}
            any_fail = True
            continue
        session_dir = os.path.join(local_root, session)
        if not os.path.isdir(session_dir):
            report[session] = {"match": False, "files": [
                {"file_name": f.get("file_name"), "manifest_digest": f.get("digest"),
                 "local_digest": None, "match": False, "note": "session dir missing"}
                for f in entry.get("files", [])
            ]}
            any_fail = True
            continue

        files_detail = []
        all_match = True
        for mf in entry.get("files", []):
            fname = mf.get("file_name")
            if fname is None:
                continue
            manifest_digest = mf.get("digest")
            local_path = os.path.join(session_dir, fname)
            if not os.path.isfile(local_path):
                files_detail.append({"file_name": fname, "manifest_digest": manifest_digest,
                                     "local_digest": None, "match": False})
                all_match = False
                continue
            try:
                local_digest = checksum_core.hash_file(local_path, algo=checksum_core._file_algo(mf))
            except (OSError, PermissionError):
                local_digest = None
            match = local_digest is not None and local_digest == manifest_digest
            if not match:
                all_match = False
            files_detail.append({"file_name": fname, "manifest_digest": manifest_digest,
                                 "local_digest": local_digest, "match": match})
        report[session] = {"match": all_match, "files": files_detail}
        if not all_match:
            any_fail = True

    print(f"{'SESSION':<{width}}  RESULT")
    print("-" * (width + 10))
    for session in sessions:
        outcome = "PASS" if report[session]["match"] else "FAIL"
        print(f"{session:<{width}}  {outcome}")

    os.makedirs(outdir, exist_ok=True)
    report_path = os.path.join(outdir, "verify_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Verify report -> {report_path}")
    return 3 if any_fail else 0


def _status_manifest(manifest_path, local_dir, outdir, list_all, emit_details):
    if not os.path.isfile(manifest_path):
        print(f"Error: manifest {manifest_path} not found.", file=sys.stderr)
        return 2
    if not os.path.isdir(local_dir):
        print(f"Error: local directory {local_dir} not found.", file=sys.stderr)
        return 2

    manifest_header, manifest_results = checksum_core.load_scan(manifest_path)
    try:
        status = checksum_core.bounded_status(manifest_results, local_dir)
    except Exception as e:
        print(f"Error running bounded status: {e}", file=sys.stderr)
        return 1

    os.makedirs(outdir, exist_ok=True)
    checksum_core.write_session_list(os.path.join(outdir, "confirmed.txt"), status["confirmed"])
    checksum_core.write_session_list(os.path.join(outdir, "stale.txt"), status["stale"])
    checksum_core.write_session_list(os.path.join(outdir, "absent.txt"), status["absent"])
    if list_all:
        checksum_core.write_session_list(os.path.join(outdir, "unlisted_present.txt"),
                                         status["unlisted_present"])

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    status_doc = {
        "header": {
            "manifest": manifest_header,
            "manifest_path": os.path.abspath(manifest_path),
            "local_dir": os.path.abspath(local_dir),
            "started": now,
            "finished": now,
            "list_all": bool(list_all),
            "emit_details": bool(emit_details),
        },
        "status": {k: v for k, v in status.items() if not (k == "details" and not emit_details)},
    }
    status_path = os.path.join(outdir, "manifest_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_doc, f, indent=2, default=str)

    print("SUMMARY")
    print("-" * 20)
    print(f"Confirmed : {len(status['confirmed'])}")
    print(f"Stale     : {len(status['stale'])}")
    print(f"Absent    : {len(status['absent'])}")
    if list_all:
        print(f"Unlisted  : {len(status['unlisted_present'])}")
    print(f"Outputs   : {outdir}")
    return 0


def main(argv=None):
    parser = ArgumentParser(
        description="Bounded local status check against the source-of-truth manifest.")
    parser.add_argument("manifest", help="Manifest JSON (source-of-truth scan with 'results').")
    parser.add_argument("local_dir", help="Local directory on this machine to check against the manifest.")
    parser.add_argument("-o", "--outdir", help="Output directory. Default: directory containing the manifest.")
    parser.add_argument("--list-all", action="store_true",
                        help="Also emit unlisted_present.txt (local sessions not in the manifest).")
    parser.add_argument("--emit-details", action="store_true",
                        help="Include per-file detail in manifest_status.json.")
    parser.add_argument("--verify", help="Verify mode: re-hash the sessions listed in this text "
                                         "file and emit a PASS/FAIL report (verify_report.json).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    outdir = args.outdir or _default_outdir(args.manifest)

    if args.verify:
        if not os.path.isfile(args.verify):
            print(f"Error: {args.verify} not found.", file=sys.stderr)
            return 2
        if not os.path.isfile(args.manifest):
            print(f"Error: manifest {args.manifest} not found.", file=sys.stderr)
            return 2
        if not os.path.isdir(args.local_dir):
            print(f"Error: local directory {args.local_dir} not found.", file=sys.stderr)
            return 2
        try:
            _, manifest_results = checksum_core.load_scan(args.manifest)
        except Exception as e:
            print(f"Error loading manifest: {e}", file=sys.stderr)
            return 2
        return _verify(manifest_results, args.local_dir, args.verify, outdir)

    if not os.path.isfile(args.manifest):
        print(f"Error: manifest {args.manifest} not found.", file=sys.stderr)
        return 2
    if not os.path.isdir(args.local_dir):
        print(f"Error: local directory {args.local_dir} not found.", file=sys.stderr)
        return 2
    return _status_manifest(args.manifest, args.local_dir, outdir,
                            args.list_all, args.emit_details)


if __name__ == "__main__":
    sys.exit(main())
