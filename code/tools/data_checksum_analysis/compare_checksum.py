"""
Compare two scan result JSON files and classify sessions into three categories:
  - ready_for_deletion: every file in the session matches between source and destination.
  - need_transfer: at least one file differs or is missing in the destination.
  - missing_from_primary: sessions that exist in the destination but not the source.

The primary scan is the source (truth) and the secondary scan is the destination.

Non-interactive (preferred):
  python compare_checksum.py <primary.json> <secondary.json> [-o OUTDIR]

Interactive (legacy): pass no positional args to list scan_results/ and pick two.
Writes a comparison report JSON plus the three session text lists to OUTDIR
(default comparison_findings/).

Exit codes: 0 clean, 1 differences detected, 2 error.
"""
import os
import sys
import json
from datetime import datetime, timezone

import checksum_core as core


def _interactive_pick():
    print('Available scans for comparison:')
    print('Primary selection will be the source scan, and secondary should be the destination scan to compare against.')
    scans = [f for f in os.listdir(os.path.join(os.getcwd(), 'scan_results'))
             if f.lower().endswith('.json')]
    for i, f in enumerate(scans):
        print(f'{i}: {f}')
    if len(scans) < 2:
        raise ValueError('Need at least two scans in scan_results/ to compare interactively.')
    scan1_index = int(input('Select the primary scan to compare: '))
    scan2_index = int(input('Select the secondary scan to compare: '))
    return scan1_index, scan2_index


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Compare two scan result JSON files.")
    p.add_argument("scans", nargs="*",
                   help="Primary (source) and secondary (destination) scan JSONs. "
                        "Omit both to enter interactive mode.")
    p.add_argument("-o", "--outdir", default=None,
                   help="Directory for the report and session lists (default comparison_findings/).")
    args = p.parse_args(argv)

    if len(args.scans) == 2:
        primary_path, secondary_path = args.scans
    elif len(args.scans) == 0:
        try:
            i1, i2 = _interactive_pick()
        except (EOFError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        scans_path = os.path.join(os.getcwd(), 'scan_results')
        scans = sorted(f for f in os.listdir(scans_path) if f.lower().endswith('.json'))
        primary_path = os.path.join(scans_path, scans[i1])
        secondary_path = os.path.join(scans_path, scans[i2])
    else:
        print("Error: provide exactly two scan JSON paths (or omit both for interactive mode).", file=sys.stderr)
        return 2

    for path in (primary_path, secondary_path):
        if not os.path.isfile(path):
            print(f"Error: {path} not found.", file=sys.stderr)
            return 2

    try:
        primary_header, primary = core.load_scan(primary_path)
        print(f'Loaded primary scan: {primary_path} with {len(primary)} sessions')
        secondary_header, secondary = core.load_scan(secondary_path)
        print(f'Loaded secondary scan: {secondary_path} with {len(secondary)} sessions')
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error loading scan: {e}", file=sys.stderr)
        return 2

    start_time = datetime.now(timezone.utc)
    classification = core.classify_sessions(primary, secondary)
    stop_time = datetime.now(timezone.utc)

    confirmed = classification["confirmed"]
    stale = classification["stale"]
    confirmed_detail = classification["confirmed_detail"]
    missing = classification["missing_from_primary"]

    report = {
        "ready_for_deletion": [
            {"path": os.path.join(session, fi["file_name"]), "md5": fi.get("digest")}
            for session in confirmed for fi in confirmed_detail.get(session, [])
        ],
        "need_transfer": [
            {"session": session, "file_count": len(primary[session].get("files", []))}
            for session in stale
        ],
        "missing_from_primary": [
            {"session": session, "file_count": len(secondary[session].get("files", []))}
            for session in missing
        ],
    }

    header = {
        "primary": primary_header,
        "secondary": secondary_header,
        "analysis": {
            "start_time": start_time.isoformat(),
            "stop_time": stop_time.isoformat(),
        },
    }
    output = {"header": header, "report": report}

    outdir = args.outdir if args.outdir else "comparison_findings"
    os.makedirs(outdir, exist_ok=True)

    output_file = "comparison_report.json"
    output_path = os.path.join(outdir, output_file)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4, default=str)
    print(f'Comparison report saved to: {output_path}')

    core.write_session_list(os.path.join(outdir, "sessions_ready_for_deletion.txt"), confirmed)
    core.write_session_list(os.path.join(outdir, "sessions_need_transfer.txt"), stale)
    core.write_session_list(os.path.join(outdir, "sessions_missing_from_primary.txt"), missing)

    print('-=' * 20)
    print('SUMMARY')
    print('-=' * 20)
    print(f'Need Transfer: {len(stale)}')
    print(f'Deletion Ready: {len(confirmed)}')
    print(f'Missing from Primary: {len(missing)}')

    if stale or missing:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
