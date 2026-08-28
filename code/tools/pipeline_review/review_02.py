"""
Comprehensive Step 02 Diagnostic.
=================================
Reads all intermediate tables from the remote site and produces:
  1. JSON report + text summary (filtering -> splitting -> ordering)
  2. PNG figures: funnel, distributions, removal breakdown, TriTime sanity

Usage:
  Container:  python review_02.py --sample 5
  Local:      python review_02.py --base_dir ./deployments/20260804_121537/ --sample 5

Outputs land in <base_dir>/review/02/ (JSON + PNG figures).
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# ----- args -------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Comprehensive Step 02 diagnostic')
    p.add_argument(
        '--base_dir', type=str, default='/FL_system/data/',
        help=('Deployment root directory where CSVs live '
               '(default: /FL_system/data/)'))
    p.add_argument('--removal_dir', type=str, default='/deployment/logs/removal_log',
                    help=('Directory containing removal CSVs (default: /deployment/logs/removal_log)'))
    p.add_argument('--sample', type=int, default=5)
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
    raise TypeError(f'Not serializable: {type(obj)}')


def _load(path):
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False, dtype={'TriTime': str})


# ----- removal helpers --------------------------------------------------------

def _removals(rem_dir):
    r = {}
    for fn in sorted(os.listdir(rem_dir)):
        if not (fn.startswith('Removed_') and fn.endswith('.csv')):
            continue
        lbl = fn.replace('Removed_', '').replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(rem_dir, fn), low_memory=False,
                             dtype={'SessionID': str})
            r[lbl] = {'scans': len(df), 'sessions': int(df['SessionID'].nunique())}
        except Exception as e:
            r[lbl] = {'error': str(e)}
    return r


# ----- accounting -------------------------------------------------------------

def _account(raw_n, final_n, removals):
    removed = sum(v.get('scans', 0) for v in removals.values() if isinstance(v, dict))
    total = removed + final_n
    diff = raw_n - total
    return {
        'raw_scans': raw_n, 'removed_scans': removed, 'final_scans': final_n,
        'accounted': total, 'unaccounted': diff,
        'pct_unaccounted': _safe_ratio(diff, raw_n),
        'balanced': abs(_safe_ratio(diff, raw_n)) < 0.01,
    }


# ----- analysis ---------------------------------------------------------------

def _flow_info(df):
    if df is None or df.empty:
        return {'scans': 0, 'sessions': 0}
    return {'scans': len(df), 'sessions': int(df['SessionID'].nunique())}


def _sps(df):
    if df is not None and not df.empty:
        return df.groupby('SessionID').size()


# ----- figures ----------------------------------------------------------------

def _fig_funnel(flow, path):
    labs = ['Raw', 'Filter', 'Split', 'Timing']
    keys = ['raw', 'filtered', 'split', 'timing']
    vals = [flow[k]['sessions'] for k in keys
            ]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labs, vals, color=['#7f8c8d', '#2ecc71', '#3498db', '#e74c3c'])
    ax.set_ylabel('Sessions')
    for bar in bars:
        h = bar.get_height()
        pct = _safe_ratio(h, vals[0]) * 100 if vals[0] > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2.,
                h + max(vals)*0.02,
                f'{h:,.0f}\n({pct:.0f}%)', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_scans_dist(sps_dict, path):
    names = [('Raw', '#95a5a6'), ('Filter', '#27ae60'),
             ('Split', '#2980b9'), ('Timing', '#c0392b')]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, c in names:
        s = sps_dict.get(name)
        if s is not None:
            mx = int(s.max()) + 1
            ax.hist(s, bins=range(1, min(mx, 30)), alpha=0.45,
                   label=name, color=c)
    ax.set_xlabel('Scans per session')
    ax.set_ylabel('Session count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_removals(removals, path):
    cats, vals = [], []
    for cat in sorted(removals.keys(), key=lambda x: removals[x].get('sessions', 0)):
        info = removals[cat]
        if isinstance(info, dict) and 'sessions' in info:
            cats.append(cat)
            vals.append(info['sessions'])

    fig, ax = plt.subplots(figsize=(10, max(4, len(cats)*0.35)))
    ax.barh(range(len(cats)), vals, color='#e74c3c')
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel('Unique sessions removed')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_post_dist(arr, path):
    if not arr:
        return
    a = np.array(arr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(a, bins=range(1, int(min(a.max()+2, 25))), color='#8e44ad', alpha=0.7)
    ax.set_xlabel('Post-scan count')
    ax.set_ylabel('Sessions')
    ax.axvline(int(np.median(a)), color='white', linestyle='--')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_tritime(total, n_u, n_z, n_o, path):
    n_n = total - n_u - n_z - n_o if total >= n_u + n_z + n_o else 0

    labels = ['All zero', 'All Unknown', 'Same\nnon-zero', 'Multi-valued']
    counts = [n_z, n_u, n_o, n_n]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    for i, (lbl, cnt) in enumerate(zip(labels, counts)):
        c = ['#e74c3c', '#95a5a6', '#f39c12', '#27ae60'][i]
        bottom_ = sum(counts[:i])
        ax.bar(i, cnt, color=c, bottom=bottom_)
        ax.text(i, bottom_ + cnt/2., f'{cnt:,}', va='center')

    ax.set_ylabel('Sessions')
    ax.set_title('Post-scan TriTime states')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----- analysis helpers -------------------------------------------------------

def _degeneracy(post_df):
    """TriTime degeneracy among post-scans per session."""
    deg = post_df.groupby('SessionID')['TriTime'].agg(['nunique', 'first'])
    deg.columns = ['nunique', 'sample_value']
    degen = deg[deg['nunique'] == 1]
    val = degen['sample_value'].str.lower().str.strip()
    return int(len(degen)), int((val == 'unknown').sum()), int((val == '0').sum()), \
           int((~val.isin(['unknown', '0'])).sum())


def _major_sanity(post_df):
    """Check that sorted Majors match linspace(1..N) per session."""
    post = post_df.sort_values(['SessionID', 'Major']).reset_index(drop=True)
    g = post.groupby('SessionID')
    n = g.transform('count')['Major'].values
    c = g.cumcount().values
    expect = 1.0 + c * (n - 1.) / np.where(n > 1, n - 1., 1.)
    bad = post.loc[np.abs(post['Major'].values - expect) > .5]
    return list(np.unique(bad['SessionID'].values))


def _acqtime_order(post_df):
    """Sessions where AcqTime is not ascending by Major."""
    post = post_df.sort_values(['SessionID', 'Major'])
    a, s = post['AcqTime'].values, np.asarray(post['SessionID'])
    bnd = np.zeros(len(a), dtype=bool); bnd[1:] = s[1:] != s[:-1]
    d = np.zeros(len(a)); d[1:] = a[1:] - a[:-1]
    oos = s[~bnd & (d < -.5)]
    return list(np.unique(oos))


def _post_dist(g):
    """Post-scan count histogram data."""
    ct = g.size()
    b = {n: int((ct == n).sum()) for n in [1,2,3,4,5]}
    b['6_plus'] = int((ct >= 6).sum())
    t = sum(b.values())
    return {f'{n}_post': {'count': v, 'fraction': _safe_ratio(v,t)}
            for n,v in b.items()}, ct.values.tolist()


def _details(df, sids, n):
    cols = ['SessionID','Series_desc','TriTime','AcqTime','Pre_scan','Post_scan','Major']
    r, seen = {}, set()
    for sid in sids[:n]:
        sk = str(sid)
        if sk in seen: continue; seen.add(sk)
        sub = df[df['SessionID']==sid][cols]
        if len(sub): r[sk] = sub.to_dict('records')
    return r


# ----- orchestrator ---------------------------------------------------------

def build_report(base, sample_n, fig_dir=None, removal_dir=None):
    raw  = _load(os.path.join(base, 'Data_table.csv'))
    filt = _load(os.path.join(base, 'Data_table_filtered.csv'))
    spl  = _load(os.path.join(base, 'Data_table_split.csv'))
    tim  = _load(os.path.join(base, 'Data_table_timing.csv'))
    rdir = removal_dir if removal_dir else os.path.join(base, 'removal_log')
    rem = _removals(rdir) if os.path.isdir(rdir) else {}

    # Synthesize SessionID for raw table (step-01 uses ID + DATE columns)
    if raw is not None and 'SessionID' not in raw.columns:
        raw['SessionID'] = raw['ID'].astype(str) + '_' + raw['DATE'].astype(str)

    flow = {'raw': _flow_info(raw), 'filtered': _flow_info(filt),
            'split': _flow_info(spl), 'timing': _flow_info(tim)}
    sps = {'Raw': _sps(raw), 'Filter': _sps(filt),
           'Split': _sps(spl), 'Timing': _sps(tim)}

    # Filtering analysis
    filt_stats = {
        'sessions_in': flow['raw']['sessions'],
        'sessions_out': flow['filtered']['sessions'],
        'lost': flow['raw']['sessions'] - flow['filtered']['sessions'],
        'fraction_lost': _safe_ratio(
            flow['raw']['sessions']-flow['filtered']['sessions'],
            flow['raw']['sessions']),
        'removal_breakdown': rem,
    }
    exp_cats = ('Computed','Description','N_samples','Not_primary',
                'Not_primary_post','Non_FS','T2')
    fail_cats = ('Pre_Failure','Post_Failure','Sequence_Failure',
                 'Laterality_Seperation_Failure','Multiple_laterality',
                 'Adjacent Series','invalid_slices',)
    filt_stats['expected_session_removals'] = sum(
        rem.get(k,{}).get('sessions',0) for k in exp_cats)
    filt_stats['failure_session_removals'] = sum(
        rem.get(k,{}).get('sessions',0) for k in fail_cats)

    # Splitting analysis
    spl_stats = {
        'sessions_in': flow['filtered']['sessions'],
        'sessions_out': flow['split']['sessions'],
        'gained_laterality': max(flow['split']['sessions']
                                -flow['filtered']['sessions'],0),
        'lost': max(flow['filtered']['sessions']
                   -flow['split']['sessions'],0),
    }

    # Ordering analysis
    order = {}
    dgrz = []  # degenerate-zero session IDs for sampling
    pc_raw = []
    if tim is not None and not tim.empty:
        post = tim[tim['Post_scan']==True].copy()
        pre  = tim[tim['Pre_scan']==True]
        g_post = post.groupby('SessionID')
        n_deg, n_unk, n_zero, n_same = _degeneracy(post)
        major_bad   = _major_sanity(post)
        acqtime_bad = _acqtime_order(post)

        order['sessions_after_order'] = flow['timing']['sessions']
        order['degTriTime'] = {
            'total': n_deg, 'all_Unknown': n_unk,
            'all_zero': n_zero, 'all_same_nonzero': n_same,
            'fraction': _safe_ratio(n_deg, flow['timing']['sessions']),
        }
        # Collect zero-TriTime session IDs for detail sampling
        degen_tbl = post.groupby('SessionID')['TriTime'].agg(
            ['nunique','first'])
        degen_tbl.columns = ['nu','sv']
        zero_ids = degen_tbl[(degen_tbl['nu']==1)
                             &(degen_tbl['sv'].str.lower().str.strip()=='0')].index
        dgrz = list(zero_ids[:sample_n])

        order['Major_sanity'] = {
            'mismatch_sessions': len(major_bad),
            'fraction': _safe_ratio(len(major_bad), flow['timing']['sessions']),
            'sample': major_bad[:sample_n],
        }
        order['AcqTime_ordering'] = {
            'out_of_order': len(acqtime_bad),
            'fraction': _safe_ratio(
                len(acqtime_bad), flow['timing']['sessions']),
            'sample': acqtime_bad[:sample_n],
        }
        order['pre_stats'] = {
            'total_pre': len(pre),
            'mean_per_session': round(len(pre)/flow['timing']['sessions'])
                if flow['timing']['sessions'] else 0,
        }
        dist_data = _post_dist(g_post)
        order['post_distribution'] = dist_data[0]
        pc_raw = dist_data[1]

    # Accounting
    acct = _account(flow['raw']['scans'] if flow['raw']['scans'] else 0,
                    flow['timing']['scans'], rem)

    # Detail samples
    detail_sids = []
    if 'Major_sanity' in order:
        detail_sids.extend(order['Major_sanity']['sample'])
    if 'AcqTime_ordering' in order:
        detail_sids.extend(order['AcqTime_ordering']['sample'])
    detail_sids.extend(dgrz)
    details = _details(tim, detail_sids, sample_n) if tim is not None else {}

    report = {
        'pipeline_flow': flow,
        'filtering': filt_stats,
        'splitting': spl_stats,
        'ordering': order,
        'accounting': acct,
        'detail_samples': details,
    }

    # --- Figures -----------------------------------------------------------
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        _fig_funnel(flow,          os.path.join(fig_dir,'01_pipeline.png'))
        _fig_scans_dist(sps, os.path.join(fig_dir,'02_scans_per_session.png'))
        _fig_removals(rem,      os.path.join(fig_dir,'03_removal_breakdown.png'))
        if pc_raw:
            _fig_post_dist(pc_raw, os.path.join(fig_dir,'04_post_scan_dist.png'))
        if 'degTriTime' in order:
            d = order['degTriTime']
            _fig_tritime(flow['timing']['sessions'], d['all_Unknown'],
                         d['all_zero'], d['all_same_nonzero'],
                        os.path.join(fig_dir,'05_tritime_states.png'))

    return report, sps


# ----- text summary ----------------------------------------------------------

def _print(report, fig_dir):
    f = report['pipeline_flow']
    print('=' * 60)
    print('  Step 02 Diagnostic')
    print('=' * 60)
    pkgs = [('Raw','raw'),('Filter','filtered'),('Split','split'),
            ('Timing','timing')]
    
    for lbl, key in pkgs:
        print(f'  {lbl:>8s}: {f[key]["scans"]:>8d} scans, '
              f'{f[key]["sessions"]:>6d} sessions')
    print()

    filt = report['filtering']
    print('  Removal breakdown:')
    for cat, info in filt['removal_breakdown'].items():
        if isinstance(info, dict) and 'scans' in info:
            print(f'    {cat:30s}: {info["scans"]:>8d} scans, '
                  f'{info["sessions"]:>5d} sessions')

    print()
    order = report['ordering'] or {}
    if odt := order.get('degTriTime'):
        print(f'  TriTime degeneracy: {odt["total"]} sessions '
              f'({odt["fraction"]:.2%}): zero={odt["all_zero"]}, '
              f'Unknown={odt["all_Unknown"]}, same_nonzero={odt["all_same_nonzero"]}')
    if mj := order.get('Major_sanity'):
        print(f'  Major mismatches: {mj["mismatch_sessions"]} '
              f'({mj["fraction"]:.2%})')
    if ao := order.get('AcqTime_ordering'):
        print(f'  AcqTime out-of-order: {ao["out_of_order"]} '
              f'({ao["fraction"]:.2%})')

    acct = report['accounting']
    print()
    print(f'  Accounting: removed={acct["removed_scans"]:,} + final={acct["final_scans"]:,} '
          f'= {acct["accounted"]:,} / raw={acct["raw_scans"]:,} '
          f'(unaccounted={acct["pct_unaccounted"]:.2%})')

    if fig_dir:
        print(f'\n  Figures saved to {fig_dir}/')


# ----- entrypoint ------------------------------------------------------------

def main():
    a = _parse_args()
    out_dir = os.path.join(os.path.abspath(a.base_dir), 'review', '02')
    os.makedirs(out_dir, exist_ok=True)

    r, _ = build_report(a.base_dir, a.sample, fig_dir=out_dir, removal_dir=a.removal_dir)
    _print(r, out_dir)

    json_path = os.path.join(out_dir, 'review_02.json')
    with open(json_path, 'w') as f:
        json.dump(r, f, indent=2, default=_serialize)
    print(f'\nOutputs -> {out_dir}/')


if __name__ == '__main__':
    main()
