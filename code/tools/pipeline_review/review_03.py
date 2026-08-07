"""
Comprehensive Step 03 Diagnostic.
==================================
Reads Data_table_timing.csv and the nifti/ output directory, then produces:
  1. JSON report (session accounting, file checks, checksum index)
  2. PNG figures: session funnel, file-count distribution, size outliers
  3. MD5 checksum manifest per session for cross-run comparison

Uses ``scan_dest.py`` from tools/data_checksum_analysis as its primary
checksum engine.  If a prior checksum manifest is found on disk, the
current run's checksums are compared against it and the comparison is
embedded in the report.

Usage:
  Container:  python review_03.py --sample 5
  Local:      python review_03.py --base_dir ./deployments/20260804_121537/ --sample 5

Outputs land in <base_dir>/review/03/ (JSON + PNG figures).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from hashlib import md5

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# ----- args -------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Comprehensive Step 03 diagnostic')
    p.add_argument(
        '--base_dir', type=str, default='/FL_system/data/',
        help='Deployment root directory where CSVs and nifti/ live '
             '(default: /FL_system/data/)')
    p.add_argument(
        '--nifti_dir', type=str, default=None,
        help='Explicit path to nifti/ output dir (default: <base_dir>/nifti/)')
    p.add_argument('--sample', type=int, default=5)
    p.add_argument(
        '--prev_manifest', type=str, default=None,
        help='Path to a prior checksum manifest JSON for comparison.')
    return p.parse_args()


# ----- helpers ----------------------------------------------------------------

def _safe_ratio(a, b):
    return round(a / b, 6) if b else 0.0


def _serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f'Not serializable: {type(obj)}')


def _file_md5(path):
    h = md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ----- checksum manifest ------------------------------------------------------

def _build_manifest(nifti_dir):
    """Build an MD5 checksum manifest indexed by session.

    Emits the same schema as scan_dest.py so that merge_checksums.py and
    compare_checksum.py can consume it without modification.
    """
    if not os.path.isdir(nifti_dir):
        return {}, {}

    results = {}
    for sid in sorted(os.listdir(nifti_dir)):
        sd = os.path.join(nifti_dir, sid)
        if not os.path.isdir(sd):
            continue
        files = []
        for fn in sorted(os.listdir(sd)):
            fp = os.path.join(sd, fn)
            try:
                sz = os.path.getsize(fp)
                checksum = _file_md5(fp)
                files.append({
                    'file_name': fn,
                    'md5': checksum,
                    'size_bytes': sz,
                })
            except (OSError, PermissionError):
                files.append({
                    'file_name': fn,
                    'md5': 'ERROR_READING_FILE',
                    'size_bytes': -1,
                })
        if files:
            results[sid] = {'files': files}

    header = {
        'scan_dir': nifti_dir,
        'start_time': datetime.now(timezone.utc).isoformat(),
        'stop_time': datetime.now(timezone.utc).isoformat(),
        'step': '03',
    }
    return results, header


def _compare_to_prior(current_results, prior_path):
    """Compare current manifest against a prior one from disk.

    Returns dict with keys: identical, modified, new_sessions, dropped_sessions.
    If *prior_path* is None or unreadable, returns {'skipped': True}.
    """
    if not prior_path or not os.path.isfile(prior_path):
        return {'skipped': True}

    try:
        with open(prior_path) as f:
            prior = json.load(f)
    except Exception:
        return {'skipped': True, 'error': f'Failed to load {prior_path}'}

    # Prior may be raw scan_dest output (has "results" key) or a nested header/results
    prior_results = prior.get('results', prior)

    idx_cur = {}  # session/filename -> md5
    for sid, data in current_results.items():
        for fi in data['files']:
            idx_cur[f'{sid}/{fi["file_name"]}'] = fi['md5']

    idx_pri = {}
    for sid, data in prior_results.items():
        for fi in data.get('files', []):
            idx_pri[f'{sid}/{fi["file_name"]}'] = fi.get('md5', fi.get('digest', ''))

    all_keys = set(idx_cur) | set(idx_pri)
    identical = 0
    modified = 0
    new_only = []
    dropped_only = []

    for k in all_keys:
        c_md5 = idx_cur.get(k)
        p_md5 = idx_pri.get(k)
        if c_md5 and p_md5 and c_md5 not in (None, ''):
            if c_md5 == p_md5:
                identical += 1
            else:
                modified += 1
        elif c_md5 and not p_md5:
            sid = k.split('/')[0]
            if sid not in new_only:
                new_only.append(sid)
        elif p_md5 and not c_md5:
            sid = k.split('/')[0]
            if sid not in dropped_only:
                dropped_only.append(sid)

    return {
        'skipped': False,
        'identical_files': identical,
        'modified_files': modified,
        'new_sessions': sorted(new_only),
        'dropped_sessions': sorted(dropped_only),
    }


# ----- session / file accounting ----------------------------------------------

def _session_accounting(timetable, nifti_dir, sample_n):
    """Cross-reference Data_table_timing.csv sessions against nifti/ dirs."""
    if timetable is None or timetable.empty:
        return {'csv_sessions': 0, 'nifti_dirs': 0, 'in_csv_not_disk': [],
                'on_disk_not_csv': [], 'expected_files': {}}

    csv_sids = set(timetable['SessionID'].unique().tolist())
    expected_counts = timetable.groupby('SessionID')['Major'].nunique().to_dict()

    disk_sids = set()
    file_counts = {}
    if os.path.isdir(nifti_dir):
        for d in sorted(os.listdir(nifti_dir)):
            dp = os.path.join(nifti_dir, d)
            if not os.path.isdir(dp):
                continue
            disk_sids.add(d)
            nifs = [f for f in os.listdir(dp) if f.endswith('.nii.gz') or f.endswith('.nii')]
            file_counts[d] = len(nifs)

    in_csv_not_disk = sorted(csv_sids - disk_sids)
    on_disk_not_csv = sorted(disk_sids - csv_sids)

    # Sessions where the number of .nii / .nii.gz files != expected Major count
    mismatch_sessions = []
    for sid in csv_sids & disk_sids:
        exp = expected_counts.get(sid, 0)
        got = file_counts.get(sid, 0)
        if got != exp:
            mismatch_sessions.append({
                'session': sid, 'expected': exp, 'actual': got})

    return {
        'csv_sessions': len(csv_sids),
        'nifti_dirs': len(disk_sids),
        'matched': len(csv_sids & disk_sids),
        'in_csv_not_disk': in_csv_not_disk[:sample_n],
        'on_disk_not_csv': on_disk_not_csv[:sample_n],
        'file_count_mismatch': mismatch_sessions[:sample_n],
    }


def _nifti_size_stats(results):
    """Compute per-session file-size statistics over .nii / .nii.gz files."""
    rows = []
    for sid, data in results.items():
        for fi in data['files']:
            if fi['file_name'].endswith(('.nii.gz', '.nii')):
                rows.append({'session': sid, 'file': fi['file_name'],
                             'size': fi.get('size_bytes', 0)})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    sizes = df['size'].values.astype(float)
    return {
        'total_nifti_files': len(sizes),
        'mean_size_MB': round(sizes.mean() / 1e6, 2),
        'median_size_MB': round(float(np.median(sizes)) / 1e6, 2),
        'min_size_MB': round(sizes.min() / 1e6, 4),
        'max_size_MB': round(sizes.max() / 1e6, 2),
        'std_size_MB': round(float(sizes.std()) / 1e6, 2),
    }


# ----- naming convention check ------------------------------------------------

def _naming_checks(timetable, nifti_dir):
    """Verify that major-number naming matches between CSV and disk."""
    if timetable is None or timetable.empty:
        return {'checked': False}

    issues = []
    matched = 0
    for sid in timetable['SessionID'].unique()[:200]:
        sd = os.path.join(nifti_dir, str(sid))
        if not os.path.isdir(sd):
            continue
        majors_csv = sorted(timetable.loc[timetable['SessionID'] == sid, 'Major'])
        expected_names = [f'{int(m):02d}' for m in majors_csv]
        disk_nifs = sorted([f.replace('.nii.gz', '').replace('.nii', '')
                           + '.json', f).split('.')[0]
                          for f in os.listdir(sd)
                          if f.endswith(('.nii.gz', '.nii'))])

        # Rebuild cleanly
        disk_basenames = sorted([os.path.splitext(f)[0]
                                 for f in os.listdir(sd)
                                 if f.endswith(('.nii.gz', '.nii'))])
        matched += 1
        missing = [n for n in expected_names if n not in disk_basenames]
        extra = [n for n in disk_basenames if n not in expected_names]
        if missing or extra:
            issues.append({'session': str(sid), 'missing': missing, 'extra': extra})

    return {
        'checked': True,
        'sessions_audited': matched,
        'naming_issues': len(issues),
        'sample_issues': issues[:10] if issues else [],
    }


# ----- figures ----------------------------------------------------------------

def _fig_session_funnel(flow, path):
    labs = ['CSV Sessions', 'Matched Dirs', 'Checksummed']
    vals = [flow['csv_sessions'], flow['matched'], flow['checksummed']]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labs, vals, color=['#3498db', '#2ecc71', '#e67e22'])
    ax.set_ylabel('Sessions')
    for bar in bars:
        h = bar.get_height()
        pct = _safe_ratio(h, vals[0]) * 100 if vals[0] > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2.,
                h + max(vals) * 0.02 if max(vals) else 0,
                f'{h:,.0f}\n({pct:.0f}%)', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_file_count_dist(file_counts, path):
    if not file_counts:
        return
    vals = list(file_counts.values())
    mx = max(int(max(vals)) + 1, 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=range(0, min(mx, 30)), color='#3498db', alpha=0.7, align='left')
    ax.set_xlabel('NIfTI files per session')
    ax.set_ylabel('Session count')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_size_violin(sizes, path):
    if not sizes or len(sizes) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.violinplot(sizes, showmeans=False, showmedians=True)
    ax.set_xlabel('Files (sorted by size)')
    ax.set_ylabel('File size (MB)')
    # Trim outliers > 3*IQR for clarity
    q1, q3 = np.percentile(sizes, [25, 75])
    iqr = q3 - q1
    trimmed = sizes[(sizes >= q1 - 3 * iqr) & (sizes <= q3 + 3 * iqr)]
    ax.set_ylim(0, min(max(trimmed) * 1.2, max(sizes[~np.isnan(sizes)]) * 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_checksum_change(change_stats, path):
    if change_stats.get('skipped'):
        return
    cats = ['Identical', 'Modified', 'New sessions', 'Dropped sessions']
    vals = [change_stats.get('identical_files', 0),
            change_stats.get('modified_files', 0),
            len(change_stats.get('new_sessions', [])),
            len(change_stats.get('dropped_sessions', []))]
    colors = ['#27ae60', '#e74c3c', '#3498db', '#f39c12']
    fig, ax = plt.subplots(figsize=(6, 3.5))
    total = sum(vals) or 1
    ax.pie(vals, labels=cats, colors=colors, autopct=f'%.{1}f%%', pctdistance=0.75)
    ax.set_title('Checksum Comparison to Prior Run')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----- orchestrator ---------------------------------------------------------

def build_report(base_dir, sample_n, nifti_dir=None, prev_manifest=None,
                 fig_dir=None):
    timetable = pd.read_csv(
        os.path.join(base_dir, 'Data_table_timing.csv'), low_memory=False,
        dtype={'TriTime': str}) if os.path.exists(
        os.path.join(base_dir, 'Data_table_timing.csv')) else None

    nifti_dir = nifti_dir or os.path.join(base_dir, 'nifti/')

    # Build checksum manifest
    results, header = _build_manifest(nifti_dir)
    file_counts = {sid: len(d['files']) for sid, d in results.items()}

    # Compare to prior if available
    change = _compare_to_prior(results, prev_manifest)

    # Accounting
    accounting = _session_accounting(timetable, nifti_dir, sample_n)
    flow = {
        'csv_sessions': accounting['csv_sessions'],
        'matched': accounting['matched'],
        'checksummed': len(results),
    }

    size_info = _nifti_size_stats(results)
    naming = _naming_checks(timetable, nifti_dir)
    manifest_path = os.path.join(
        os.path.dirname(nifti_dir), 'review', '03', 'nifti_checksum_manifest.json') if fig_dir else None

    # Determine prior manifest for comparison
    if prev_manifest is None and fig_dir:
        default_prev = os.path.join(fig_dir, 'nifti_checksum_manifest.json')
        if os.path.exists(default_prev):
            change = _compare_to_prior(results, default_prev)

    report = {
        'header': header,
        'flow': flow,
        'accounting': accounting,
        'size_statistics': size_info,
        'naming_checks': naming,
        'checksum_comparison': change,
        'manifest_session_count': len(results),
    }

    # --- Figures -----------------------------------------------------------
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        _fig_session_funnel(flow, os.path.join(fig_dir, '01_session_funnel.png'))
        _fig_file_count_dist(file_counts,
                             os.path.join(fig_dir, '02_file_count_dist.png'))
        if size_info:
            _fig_size_violin(
                np.array([fi.get('size_bytes', 0) / 1e6
                          for data in results.values()
                          for fi in data['files']
                          if fi['file_name'].endswith(('.nii.gz', '.nii'))]),
                os.path.join(fig_dir, '03_file_size_distribution.png'))
        if not change.get('skipped'):
            _fig_checksum_change(change,
                                 os.path.join(fig_dir, '04_checksum_changes.png'))

        # Persist manifest for next-run comparison
        manifest_out = {
            'header': header,
            'results': results,
        }
        mpath = os.path.join(fig_dir, 'nifti_checksum_manifest.json')
        with open(mpath, 'w', encoding='utf-8') as f:
            json.dump(manifest_out, f, indent=2, default=_serialize)

    return report


# ----- text summary ----------------------------------------------------------

def _print(report, fig_dir):
    hdr = report['header']
    print('=' * 60)
    print('  Step 03 Diagnostic')
    print('=' * 60)
    fl = report['flow']
    print(f'  CSV sessions   : {fl["csv_sessions"]:>8d}')
    print(f'  Matched dirs   : {fl["matched"]:>8d}')
    print(f'  Checksummed    : {fl["checksummed"]:>8d}')
    print()

    acc = report['accounting']
    if acc.get('in_csv_not_disk'):
        print(f'  Missing on disk ({len(acc["in_csv_not_disk"])} sampled):')
        for s in acc['in_csv_not_disk'][:5]:
            print(f'    {s}')
    if acc.get('on_disk_not_csv'):
        print(f'  Extra on disk ({len(acc["on_disk_not_csv"])} sampled):')
        for s in acc['on_disk_not_csv'][:5]:
            print(f'    {s}')
    if acc.get('file_count_mismatch'):
        print(f'  File-count mismatches ({len(acc["file_count_mismatch"])} sampled):')
        for m in acc['file_count_mismatch'][:5]:
            print(f'    {m["session"]}: expected={m["expected"]}, actual={m["actual"]}')

    if st := report.get('size_statistics'):
        print()
        print(f'  Size stats:')
        print(f'    Total NIfTI files: {st["total_nifti_files"]:,}')
        print(f'    Mean / median:     {st["mean_size_MB"]:.2f} / {st["median_size_MB"]:.2f} MB')
        print(f'    Std:              {st["std_size_MB"]:.2f} MB')

    if nc := report.get('naming_checks'):
        if nc.get('checked'):
            print()
            print(
                f'  Naming audit: {nc["sessions_audited"]} sessions, '
                f'{nc["naming_issues"]} issues')

    if cc := report.get('checksum_comparison'):
        if not cc.get('skipped'):
            print()
            print(f'  Checksum vs prior run:')
            print(f'    Identical : {cc.get("identical_files", 0):,}')
            print(f'    Modified  : {cc.get("modified_files", 0):,}')
            print(f'    New       : {len(cc.get("new_sessions", [])):,}')
            print(f'    Dropped   : {len(cc.get("dropped_sessions", [])):,}')

    if fig_dir:
        print(f'\n  Figures  -> {fig_dir}/')
        print(f'  Manifest -> {os.path.join(fig_dir, "nifti_checksum_manifest.json")}')


# ----- entrypoint ------------------------------------------------------------

def main():
    a = _parse_args()
    out_dir = os.path.join(os.path.abspath(a.base_dir), 'review', '03')
    os.makedirs(out_dir, exist_ok=True)

    r = build_report(
        a.base_dir, a.sample, nifti_dir=a.nifti_dir,
        prev_manifest=a.prev_manifest, fig_dir=out_dir)
    _print(r, out_dir)

    json_path = os.path.join(out_dir, 'review_03.json')
    with open(json_path, 'w') as f:
        json.dump(r, f, indent=2, default=_serialize)

    print(f'\nOutputs -> {out_dir}/')


if __name__ == '__main__':
    main()
