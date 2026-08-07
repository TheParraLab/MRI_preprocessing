import argparse
from collections import namedtuple
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
import numpy as np


INPUT_DIR = '/FL_system/data/inputs/'
OUTPUT_PATH = Path('/tmp/opencode/input_validation.pdf')

DISPLAY_ORDER = ['post', 'slope1', 'slope2']
CMAPS = {'slope1': 'viridis', 'slope2': 'plasma', 'post': 'gray'}
KINETIC_COLORS = {
    'Type 1 (persistent)': '#2ecc71',
    'Type 2 (plateau)':   '#f39c12',
    'Type 3 (washout)':   '#e74c3c',
}

Meta = namedtuple('Meta', [
    'name', 'shape', 'total_vox',
    'stats_post', 'stats_s1', 'stats_s2',
    'pct_nz_post', 'pct_nz_s1', 'pct_nz_s2',
    'med_post', 'med_s1', 'med_s2',
    't1', 't2', 't3',
    'corr',
    'enh_frac', 'wash_frac', 'snr',
])


# -- helpers -------------------------------------------------------------------

def find_session_dirs(d: str):
    p = Path(d).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f'Not found: {p}')
    dirs = sorted(x for x in p.iterdir() if x.is_dir())
    if not dirs:
        raise FileNotFoundError(f'No session dirs in {p}')
    return dirs


def _load(sd, ch):
    for ext in ('.nii.gz', '.nii'):
        fp = sd / f'{ch}{ext}'
        if fp.exists():
            return np.asanyarray(nib.load(str(fp)).dataobj).squeeze()
    raise FileNotFoundError(f'{ch} not found in {sd}')


def _mask(vol):
    return np.isfinite(vol) & (vol != 0)


def _stats(vol):
    v = vol[_mask(vol)]
    if v.size == 0:
        return dict(zip(('min','max','mean','std','p5','p25','p50','p75','p95'), [0]*9))
    return {k: float(f(v)) for k, f in [
        ('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std),
        ('p5', lambda x: np.percentile(x, 5)),
        ('p25', lambda x: np.percentile(x, 25)),
        ('p50', lambda x: np.percentile(x, 50)),
        ('p75', lambda x: np.percentile(x, 75)),
        ('p95', lambda x: np.percentile(x, 95)),
    ]}


def _medpos(vol):
    v = vol[np.isfinite(vol)]
    p = v[v > 0]
    return float(np.median(p)) if p.size else 0.0


def _pearson3(a, b, c, n=50000):
    sz = len(a)
    if sz > n:
        st = max((sz - 1) // n, 1)
        a, b, c = a[::st], b[::st], c[::st]
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.corrcoef(np.column_stack((a, b, c)).T)
        M[~np.isfinite(M)] = 0.0
    return M


def _kinetics(s1, s2):
    m = _mask(s1) & _mask(s2)
    if m.sum() == 0:
        return 0., 0., 0.
    s1p, s2v = s1[m], s2[m]
    t1t = float(np.median(s1p[s1p > 0])) if (s1p > 0).any() else 0.
    t3t = float(np.percentile(s2v, 25))
    t2l = float(np.percentile(s2v, 50))
    N = int(m.sum())
    return (
        int(((s1 > t1t) & (s2 >= t2l)).sum()) / N,
        int(((s1 > t1t) & (s2 < t2l) & (s2 > t3t)).sum()) / N,
        int(((s1 > t1t) & (s2 <= t3t)).sum()) / N,
    )


def _ew_fractions(s1, s2):
    v1 = s1[_mask(s1)]
    ef = float(np.mean(v1 > np.percentile(v1, 50) + 2 * np.std(v1))) if v1.size > 10 else 0.
    v2 = s2[_mask(s2)]
    wf = float(np.mean(v2 < np.percentile(v2, 50) - 2 * np.std(v2))) if v2.size > 10 else 0.
    return ef, wf


def _short(lbl):
    p = lbl.split('_')
    return f'{p[0]}_{p[-1]}' if len(p) >= 2 else lbl[:16]


# -- single-pass pre-compute ---------------------------------------------------

def compute_metas(sessions):
    """Return list of (session_dir_path, Meta) tuples. Skipped sessions are simply omitted."""
    pairs = []
    for sd in sessions:
        try:
            post, s1, s2 = _load(sd, 'post'), _load(sd, 'slope1'), _load(sd, 'slope2')
            tv = int(np.prod(post.shape))
            sp, ss1, ss2 = _stats(post), _stats(s1), _stats(s2)
            t1f, t2f, t3f = _kinetics(s1, s2)
            ef, wf = _ew_fractions(s1, s2)
            m_all = _mask(s1) | _mask(s2) | _mask(post)
            corr = _pearson3(s1[m_all], s2[m_all], post[m_all]) if m_all.any() else np.eye(3)
            bg = s1[np.isfinite(s1) & (s1 == 0)]
            bstd = float(bg.std()) if bg.size > 5 else 0.
            snr = float(np.abs(ss1['mean']) / bstd) if bstd else 0.
            pairs.append((sd, Meta(
                name=sd.name, shape=post.shape, total_vox=tv,
                stats_post=sp, stats_s1=ss1, stats_s2=ss2,
                pct_nz_post=round(100 * len(post[_mask(post)]) / tv, 1),
                pct_nz_s1=round(100 * len(s1[_mask(s1)]) / tv, 1),
                pct_nz_s2=round(100 * len(s2[_mask(s2)]) / tv, 1),
                med_post=_medpos(post), med_s1=_medpos(s1), med_s2=_medpos(s2),
                t1=t1f, t2=t2f, t3=t3f, corr=corr, enh_frac=ef, wash_frac=wf, snr=snr,
            )))
            del post, s1, s2
        except Exception as exc:
            print(f'  Failed to pre-compute {sd.name}: {exc}')
    return pairs


# -- slice rendering (needs disk once more) ------------------------------------

def _pick_slice(vol):
    n = vol.shape[2]
    mg = max(n // 6, 1)
    idx = random.randint(mg, n - mg)
    return np.rot90(np.squeeze(vol[:, :, idx])), idx


def _scale(sl):
    fv = sl[np.isfinite(sl)]
    if fv.size == 0:
        return np.zeros_like(sl), 0., 0.
    lo, hi = np.percentile(fv, [1, 99])
    if lo == hi:
        return np.zeros_like(sl), lo, hi
    return np.clip((sl - lo) / (hi - lo), 0, 1), lo, hi


def render_session(sd, meta, idx):
    sc_info = {}
    for ch in DISPLAY_ORDER:
        v = _load(sd, ch)
        sl, ci = _pick_slice(v)
        nm, lo, hi = _scale(sl)
        sc_info[ch] = (nm, lo, hi)
        del v

    fig = plt.figure(figsize=(18, 5.5))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 3, 3, 3], wspace=0.15)
    ai = fig.add_subplot(gs[0])
    ai.axis('off')
    lines = [f'Session #{idx}: {meta.name}', '']
    st_map = {'post': meta.stats_post, 'slope1': meta.stats_s1, 'slope2': meta.stats_s2}
    for ch in DISPLAY_ORDER:
        s = st_map[ch]
        lo, hi = sc_info[ch][1], sc_info[ch][2]
        lines.append(f'  {ch}: mean={s["mean"]:.4f}  std={s["std"]:.4f}  range=[{lo:.4f},{hi:.4f}]')
    ai.text(0, 0.95, '\n'.join(lines), transform=ai.transAxes,
            fontsize=8, va='top', family='monospace')

    for ci2, ch in enumerate(DISPLAY_ORDER):
        ax = fig.add_subplot(gs[ci2 + 1])
        nm, lo, hi = sc_info[ch]
        ax.imshow(nm, cmap=plt.get_cmap(CMAPS[ch]), origin='lower', vmin=0, vmax=1)
        ax.set_title(f'{ch.upper()}\n[{lo:.4f}  {hi:.4f}]', fontsize=10)
        ax.axis('off')

    fig.suptitle(f'{meta.name}  (page {idx})', fontsize=13, fontweight='bold')
    return fig


# -- analysis pages (consume metas only) ---------------------------------------

def page_scatter(metas):
    n = len(metas)
    fig, (a, b) = plt.subplots(1, 2, figsize=(18, 7))
    a.scatter([m.med_s1 for m in metas], [m.med_post for m in metas],
              s=max(20, 600 / n), color='#440154', edgecolors='k', lw=0.5, alpha=0.85)
    a.axhline(0, color='gray', ls='--', lw=0.8)
    a.axvline(0, color='gray', ls='--', lw=0.8)
    if n <= 60:
        for m in metas:
            a.annotate(_short(m.name), (m.med_s1, m.med_post), fontsize=5, alpha=0.55)
    a.set_xlabel('Med slope1')
    a.set_ylabel('Med post')
    a.set_title('Post vs Enhancement')

    mn = max(1, max(abs(m.med_s1) for m in metas),
             abs(min((m.med_s2 for m in metas))),
             abs(max((m.med_s2 for m in metas))))
    b.scatter([m.med_s1 for m in metas], [m.med_s2 for m in metas],
              s=max(20, 600 / n), color='#FDE725', edgecolors='k', lw=0.5, alpha=0.85)
    b.plot([-mn, mn], [-mn, mn], color='k', lw=1.2)
    if n <= 60:
        for m in metas:
            b.annotate(_short(m.name), (m.med_s1, m.med_s2), fontsize=5, alpha=0.55)
    b.set_xlabel('Med slope1')
    b.set_ylabel('Med slope2')
    b.set_title('Slope1 vs Slope2 (identity ref.)')

    fig.suptitle(f'Per-session medians ({n} sessions)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def page_corr(metas):
    all_c = np.array([m.corr for m in metas])
    max_g = min(len(metas), 25)
    idxs = np.linspace(0, len(metas) - 1, max_g, dtype=int)
    sub_metas = [metas[i] for i in idxs]
    sc = all_c[idxs]

    n = len(sub_metas)
    cols = max(1, min(n, 5))
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4 * rows), squeeze=False)
    flat = axes.flatten()
    labels = ['slope1', 'slope2', 'post']

    im = None
    for ii in range(n):
        ax = flat[ii]
        c = sc[ii]
        t_im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        if im is None:
            im = t_im
        for r2 in range(3):
            for c2 in range(3):
                cl = 'white' if abs(c[r2, c2]) > 0.6 else 'black'
                ax.text(c2, r2, f'{c[r2, c2]:.2f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=cl)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(_short(sub_metas[ii].name), fontsize=7)

    for ax in flat[len(sub_metas):]:
        ax.axis('off')
    if im is not None:
        cb = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cb, label='Pearson r')

    med_r = float(np.median(all_c[:, 0, 1]))
    extra = f' (showing {max_g} of {len(metas)})' if len(metas) > max_g else ''
    fig.suptitle(
        f'Voxelwise correlation (Pearson r){extra}\n'
        f'Median r(s1,s2)={med_r:.3f}. Negative approx washout/Type-3 kinetics.',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.89, 0.95])
    return fig


def page_kinetics(metas):
    if not metas:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    x = np.arange(len(metas))
    t1s = [m.t1 for m in metas]
    t2s = [m.t2 for m in metas]
    t3s = [m.t3 for m in metas]

    fig, (axb, axl) = plt.subplots(1, 2, figsize=(18, 7), width_ratios=[4, 1])
    axb.bar(x, t1s, color=KINETIC_COLORS['Type 1 (persistent)'], alpha=0.85)
    axb.bar(x, t2s, bottom=t1s, color=KINETIC_COLORS['Type 2 (plateau)'], alpha=0.85)
    axb.bar(x, t3s, bottom=[a + b for a, b in zip(t1s, t2s)],
            color=KINETIC_COLORS['Type 3 (washout)'], alpha=0.85)

    flabel = _short if len(metas) > 40 else lambda x: x
    axb.set_xticks(x)
    axb.set_xticklabels([flabel(m.name) for m in metas], rotation=60, ha='right', fontsize=7)
    axb.set_ylabel('Active volume fraction')
    axb.set_title('BI-RADS kinetic types\nType 1 = persistent | Type 2 = plateau | Type 3 = washout')
    axb.set_ylim(0, 1.05)
    axb.grid(axis='y', alpha=0.4)

    hs = [plt.Line2D([0], [0], marker='s', color='w', markersize=10,
                     markerfacecolor=c, label=l) for l, c in KINETIC_COLORS.items()]
    axl.legend(handles=hs, loc='center', fontsize=10); axl.axis('off')

    fig.suptitle(
        'Kinetic classification (Kuhl 3-type system)\n'
        'High Type-3 approx elevated washout volume (suspicious for malignancy)',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def page_eh_vs_wash(metas):
    n = len(metas)
    efs = [m.enh_frac for m in metas]
    wfs = [m.wash_frac for m in metas]

    fig, ax = plt.subplots(figsize=(10, 7))
    mx = max(0.02, max(max(efs), max(wfs)))
    lo = min(0, min(efs), min(wfs))

    ax.scatter(efs, wfs, s=max(20, 4000 // n), color='#69b3a5',
               edgecolors='k', lw=0.5, alpha=0.8)
    ax.plot([lo, mx], [lo, mx], color='k', lw=1.2)

    if n <= 60:
        for m in metas:
            ax.annotate(_short(m.name), (m.enh_frac, m.wash_frac), fontsize=5, alpha=0.55)

    ax.set_xlabel('Enhancement fraction (> +2 sigma slope1)')
    ax.set_ylabel('Washout fraction (< -2 sigma slope2)')
    ax.set_title(
        f'Enh vs Wash fractions  ({n} sessions)\n'
        'Diagonal = equal extent; deviation suggests anomaly',
        fontsize=11)
    fig.suptitle(
        'Volume-fraction anomaly detection\n'
        'Related to ICNR radiomics enhancement-volume metrics',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def page_tables(metas):
    max_rows = 200
    if not metas:
        return []

    n_pages = -(-len(metas) // max_rows)
    results = []
    for pg in range(n_pages):
        chunk = metas[pg * max_rows: (pg + 1) * max_rows]
        rows = [[_short(m.name), f'{m.shape[0]}x{m.shape[1]}x{m.shape[2]}',
                 m.pct_nz_post, m.pct_nz_s1, m.pct_nz_s2,
                 round(m.stats_s1['mean'], 4), round(m.stats_s1['p50'], 4),
                 round(m.stats_s1['p95'], 4),
                 round(m.stats_s2['mean'], 4), round(m.stats_s2['p50'], 4),
                 round(m.stats_s2['p95'], 4),
                 round(m.snr, 2),
                 f'{m.t1:.0%}', f'{m.t2:.0%}', f'{m.t3:.0%}']
                for m in chunk]

        cols = ['Session', 'Shape', '%nz post', '%nz s1', '%nz s2',
                'Mean s1', 'Med s1', 'P95 s1',
                'Mean s2', 'Med s2', 'P95 s2',
                'SNR', 'T1%', 'T2%', 'T3%']

        fh = max(4, 1.8 * (len(rows) + 1))
        fig, ax = plt.subplots(figsize=(22, fh))
        ax.axis('off')
        tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(6); tbl.scale(1.4, 2)
        for j in range(len(cols)):
            tbl[(0, j)].set_facecolor('#2c3e50')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(rows) + 1):
            bg = '#ecf0f1' if i % 2 else 'white'
            for j in range(len(cols)):
                tbl[(i, j)].set_facecolor(bg)

        sux = f' (page {pg+1}/{n_pages})' if n_pages > 1 else ''
        fig.suptitle(
            f'Per-session summary{sux}\n'
            'SNR=|mean(s1)|/sigma(bkg). T1/T2/T3 = BI-RADS kinetic fractions.',
            fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        results.append(fig)

    return results


# -- terminal summary ---------------------------------------------------------

def print_dist(metas):
    print('\n' + '=' * 78)
    print('  INPUT CHANNEL DISTRIBUTION SUMMARY')
    print('=' * 78)

    hdr = '{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        'Channel', 'Voxels', 'Min', 'Max', 'Mean', 'Std', 'P5', 'P25', 'Med', 'P75', 'P95')
    print(hdr)
    print('-' * 78)

    for ch in DISPLAY_ORDER:
        attr = f'stats_{ch[:4]}' if ch != 'post' else 'stats_post'
        stats_list = [getattr(m, attr) for m in metas]
        nv_list = [int(m.total_vox * getattr(m, f'pct_nz_{ch}') / 100) for m in metas]

        if not any(s['mean'] != 0 for s in stats_list):
            print('{:<10}  (no data)'.format(ch))
            continue

        weights = [w / max(nv_list) for w in nv_list]
        keys = ('min', 'max', 'mean', 'std', 'p5', 'p25', 'p50', 'p75', 'p95')
        agg = {k: float(np.average([s[k] for s in stats_list], weights=weights)) for k in keys}
        tot = sum(nv_list)
        print('{:<10} {:>10,} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}'.format(
            ch, tot, *(agg[k] for k in keys)))

    print('=' * 78 + '\n')


# -- orchestrator -------------------------------------------------------------

def run(input_dir, output_path, limit=None):
    sessions = find_session_dirs(input_dir)
    if limit:
        sessions = sessions[:limit]
    ns = len(sessions)

    print('  Pre-computing per-session stats (single-pass, ~3 loads/session)...')
    pairs = compute_metas(sessions)  # list of (sd_path, Meta)
    ok = len(pairs)
    metas = [m for _, m in pairs]
    print(f'  {ok}/{ns} sessions loaded; skipped {ns - ok if ns >= ok else 0}')
    print(f'Saving PDF to {output_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:

        # 1. per-session slice pages (volumes already freed; one more disk pass for images only)
        for i, (sd, meta) in enumerate(pairs):
            try:
                fig = render_session(sd, meta, i + 1)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                print(f'  [{i + 1}/{ok}] Session slice: {meta.name}')
            except Exception as exc:
                print(f'  [{i + 1}/{ok}] Skipping {meta.name}: {exc}')

        # 2. scatter comparison
        try:
            pdf.savefig(page_scatter(metas), bbox_inches='tight')
            plt.close('all')
            print('  [+1] Scatter comparison page')
        except Exception as exc:
            print(f'  [+1] Scatter ERROR: {exc}')

        # 3. correlation heatmap
        try:
            pdf.savefig(page_corr(metas), bbox_inches='tight')
            plt.close('all')
            print('  [+2] Correlation heatmap page')
        except Exception as exc:
            print(f'  [+2] Correlation ERROR: {exc}')

        # 4. kinetic fractions
        try:
            pdf.savefig(page_kinetics(metas), bbox_inches='tight')
            plt.close('all')
            print('  [+3] Kinetic-type fractions page')
        except Exception as exc:
            print(f'  [+3] Kinetics ERROR: {exc}')

        # 5. enhancement vs washout scatter
        try:
            pdf.savefig(page_eh_vs_wash(metas), bbox_inches='tight')
            plt.close('all')
            print('  [+4] Enh/wash fraction scatter page')
        except Exception as exc:
            print(f'  [+4] Eh/wash scatter ERROR: {exc}')

        # 6. summary table(s) — paginati



def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate slope/post-contrast inputs with DCE-MRI kinetic analysis.')
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR,
                        help='Directory containing session folders (default: %(default)s)')
    parser.add_argument('--output', type=str, default=str(OUTPUT_PATH),
                        help='Output PDF path (default: %(default)s)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only the first N sessions')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    run(args.input_dir, Path(args.output), limit=args.limit)


if __name__ == '__main__':
    main()
