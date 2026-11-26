"""Figures of the report, from results/raw/*.csv -> report/figs/*.pdf"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csvutil import load, median_by

C = {'k49_norm_10NN': '#2a78d6', 'com-Amazon': '#eb6834', 'coPapersDBLP': '#1baf7a'}
LS = {'omp': '-', 'cilk': '--', 'pthreads': '-.'}
LBL = {'omp': 'OpenMP', 'cilk': 'OpenCilk', 'pthreads': 'Pthreads'}
GRAPHS = list(C)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'lines.markersize': 3.5, 'legend.frameon': False})
os.makedirs('report/figs', exist_ok=True)

# ---------- scaling: speedup per phase vs p ----------
d = [r for r in load('results/raw/scaling.csv') if '-direct' not in r['version']]
phases = [('t_total', 'total'), ('t_P', 'P'), ('t_attr', 'attraction'),
          ('t_build', 'tree build'), ('t_walk', 'walk')]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.3))
for ax, (f, name) in zip(axs.flat, phases):
    med = median_by(d, ['graph', 'version', 'threads'], f)
    for g in GRAPHS:
        t1 = med[(g, 'seq', 1)]
        for v in ('omp', 'cilk', 'pthreads'):
            ps = sorted(p for (gg, vv, p) in med if gg == g and vv == v)
            ax.plot(ps, [t1 / med[(g, v, p)] for p in ps], LS[v], color=C[g], marker='o')
    ax.plot([1, 16], [1, 16], ':', color=MUTED, lw=1)
    ax.axvline(8, color=MUTED, lw=0.8, alpha=0.5)
    ax.set_title(f'speedup: {name}', fontsize=8.5, color=INK)
    ax.set_xticks([1, 2, 4, 6, 8, 12, 16]); ax.set_ylim(0, 16.5)
for ax in axs[1]: ax.set_xlabel('threads p')
for ax in axs[:, 0]: ax.set_ylabel('T_seq / T_p')
# last panel: absolute total time
ax = axs[1, 2]
med = median_by(d, ['graph', 'version', 'threads'], 't_total')
for g in GRAPHS:
    for v in ('omp', 'cilk', 'pthreads'):
        ps = sorted(p for (gg, vv, p) in med if gg == g and vv == v)
        ax.plot(ps, [med[(g, v, p)] for p in ps], LS[v], color=C[g], marker='o')
ax.set_yscale('log'); ax.set_xticks([1, 2, 4, 6, 8, 12, 16])
ax.set_title('total time [s]', fontsize=8.5, color=INK); ax.set_xlabel('threads p')
from matplotlib.lines import Line2D
h = [Line2D([], [], color=C[g], marker='o', label=g) for g in GRAPHS] + \
    [Line2D([], [], color=MUTED, ls=LS[v], label=LBL[v]) for v in LS] + \
    [Line2D([], [], color=MUTED, ls=':', label='ideal')]
fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig('report/figs/scaling.pdf'); plt.close(fig)

# ---------- leaf capacity + crossover ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.4))
lc = load('results/raw/leafcap.csv')
med = median_by(lc, ['graph', 'version', 'leafcap'], lambda r: r['t_build'] + r['t_com'] + r['t_walk'])
for g in GRAPHS:
    for v, ls in (('seq', '-'), ('omp', '--')):
        ms = sorted(m for (gg, vv, m) in med if gg == g and vv == v)
        a1.plot(ms, [med[(g, v, m)] for m in ms], ls, color=C[g], marker='o')
a1.set_xscale('log', base=2); a1.set_yscale('log'); a1.axvline(16, color=MUTED, lw=0.8, alpha=0.6)
a1.set_xlabel('leaf capacity m'); a1.set_ylabel('build + centers + walk [s]')
a1.set_title('leaf capacity (solid: seq, dashed: OpenMP p=8)', fontsize=8.5, color=INK)
a1.legend(handles=[Line2D([], [], color=C[g], marker='o', label=g) for g in GRAPHS], fontsize=6.5,
          loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.5)
a1.set_ylim(top=30)
ms_all = [1, 2, 4, 8, 16, 32, 64, 128, 256]
a1.set_xticks(ms_all); a1.set_xticklabels([str(m) for m in ms_all])
xo = load('results/raw/crossover.csv')
for v, col in (('seq', '#2a78d6'), ('omp', '#eb6834')):
    sel = [r for r in xo if r['version'] == v]
    bh = median_by(sel, ['nsub'], 't_bh'); di = median_by(sel, ['nsub'], 't_direct')
    ns = sorted(k[0] for k in bh)
    lab = 'sequential' if v == 'seq' else 'OpenMP p=16'
    a2.plot(ns, [bh[(x,)] for x in ns], '-', color=col, marker='o', label=f'Barnes-Hut, {lab}')
    a2.plot(ns, [di[(x,)] for x in ns], ':', color=col, marker='s', label=f'direct, {lab}')
a2.set_xscale('log'); a2.set_yscale('log'); a2.set_xlabel('points n (subsets of com-Amazon)')
a2.set_ylabel('repulsion time [s]'); a2.legend(fontsize=6.5)
a2.set_title('Barnes-Hut vs direct O(n^2)', fontsize=8.5, color=INK)
fig.tight_layout(); fig.savefig('report/figs/leafcap_crossover.pdf'); plt.close(fig)
print('figures in report/figs/')
