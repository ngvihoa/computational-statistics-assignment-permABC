"""Generate all figures from benchmark_raw.csv — mirrors the notebook cells."""
import warnings; warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.1)
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 200

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'assignment_benchmark')
FIG_DIR  = os.path.join(os.path.dirname(__file__), '..', 'figures', 'assignment_benchmark')
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, 'benchmark_raw.csv'))
print(f'{df.shape[0]} runs loaded')

K_VALUES     = sorted(df.K.unique())
N_OBS_VALUES = sorted(df.n_obs.unique())
ALPHA_VALUES = sorted(df.alpha.unique(), reverse=True)

METHOD_ORDER = ['LSA', 'LSA+Swap', 'Hilbert', 'Hilbert+Swap',
                'Swap only', 'Smart Swap', 'Smart Hilbert', 'Smart H+S']
PALETTE = {
    'LSA': '#1f77b4', 'LSA+Swap': '#aec7e8',
    'Hilbert': '#2ca02c', 'Hilbert+Swap': '#98df8a',
    'Swap only': '#ff7f0e', 'Smart Swap': '#d62728',
    'Smart Hilbert': '#9467bd', 'Smart H+S': '#8c564b',
}

# --- compute relative metrics ---
base = df[df['method'] == 'LSA'].set_index(['K', 'n_obs', 'alpha'])
rows = []
for _, r in df.iterrows():
    idx = (r['K'], r['n_obs'], r['alpha'])
    b = base.loc[idx]
    speedup  = b['time'] / r['time'] if r['time'] > 0 else np.nan
    rel_mse  = r['mse_loc'] / b['mse_loc'] if b['mse_loc'] > 0 else np.nan
    nsim_rat = r['total_nsim'] / b['total_nsim'] if b['total_nsim'] > 0 else np.nan
    rows.append({**r.to_dict(), 'speedup': speedup, 'rel_mse': rel_mse, 'nsim_ratio': nsim_rat})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, 'benchmark_full.csv'), index=False)

summary = df.groupby('method').agg(
    time_median=('time', 'median'), speedup_median=('speedup', 'median'),
    mse_loc_median=('mse_loc', 'median'), rel_mse_median=('rel_mse', 'median'),
    n_iters_median=('n_iters', 'median'), final_eps_median=('final_eps', 'median'),
).round(4).reindex(METHOD_ORDER)
print(summary.to_string())
summary.to_csv(os.path.join(DATA_DIR, 'benchmark_summary.csv'))

# Helpers
n_m = len(METHOD_ORDER)
nc_h, nr_h = 4, (n_m + 3) // 4

# ---- Fig 1: Speedup vs K ----
print('  Fig 1: speedup vs K')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, alpha in zip(axes, ALPHA_VALUES):
    sub = df[(df['alpha'] == alpha) & (df['method'] != 'LSA')]
    agg = sub.groupby(['K', 'method'])['speedup'].median().reset_index()
    for lab in METHOD_ORDER:
        if lab == 'LSA': continue
        d = agg[agg['method'] == lab]
        if d.empty: continue
        ax.plot(d['K'], d['speedup'], 'o-', label=lab, color=PALETTE[lab], lw=2, ms=6)
    ax.axhline(1, ls='--', color=PALETTE['LSA'], alpha=.5, label='LSA (baseline)')
    ax.set_xlabel('K'); ax.set_title(f'α = {alpha}', fontsize=13)
    ax.set_xscale('log'); ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[0].set_ylabel('Speedup vs LSA (median over n_obs)')
axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
fig.suptitle('Speedup relative to pure LSA', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_speedup_vs_K.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 2: Relative MSE vs K ----
print('  Fig 2: relative MSE vs K')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, alpha in zip(axes, ALPHA_VALUES):
    sub = df[(df['alpha'] == alpha) & (df['method'] != 'LSA')]
    agg = sub.groupby(['K', 'method'])['rel_mse'].median().reset_index()
    for lab in METHOD_ORDER:
        if lab == 'LSA': continue
        d = agg[agg['method'] == lab]
        if d.empty: continue
        ax.plot(d['K'], d['rel_mse'], 'o-', label=lab, color=PALETTE[lab], lw=2, ms=6)
    ax.axhline(1, ls='--', color=PALETTE['LSA'], alpha=.5, label='LSA (baseline)')
    ax.set_xlabel('K'); ax.set_title(f'α = {alpha}', fontsize=13)
    ax.set_xscale('log'); ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[0].set_ylabel('Relative MSE (vs LSA, <1 = better)')
axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
fig.suptitle('Posterior quality relative to pure LSA', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_rel_mse_vs_K.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 3: Efficiency frontier ----
print('  Fig 3: efficiency frontier')
K_show = [k for k in K_VALUES if k >= 3]
ncols_f = min(3, len(K_show)); nrows_f = (len(K_show) + ncols_f - 1) // ncols_f
fig, axes = plt.subplots(nrows_f, ncols_f, figsize=(6*ncols_f, 5*nrows_f), squeeze=False)
for idx, K_val in enumerate(K_show):
    ax = axes[idx // ncols_f, idx % ncols_f]
    sub = df[(df['K'] == K_val) & (df['method'] != 'LSA')]
    for lab in METHOD_ORDER:
        if lab == 'LSA': continue
        d = sub[sub['method'] == lab]
        if d.empty: continue
        ax.scatter(d['speedup'], d['rel_mse'], color=PALETTE[lab],
                   label=lab, s=55, alpha=.7, edgecolors='w', lw=.5)
    ax.axhline(1, color='grey', ls='--', alpha=.3)
    ax.axvline(1, color='grey', ls='--', alpha=.3)
    ax.set_xlabel('Speedup vs LSA'); ax.set_ylabel('Relative MSE')
    ax.set_title(f'K = {K_val}', fontsize=13)
for idx in range(len(K_show), nrows_f*ncols_f):
    axes[idx // ncols_f, idx % ncols_f].set_visible(False)
h, l = axes[0, 0].get_legend_handles_labels()
by_label = dict(zip(l, h))
fig.legend(by_label.values(), by_label.keys(),
           loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(.5, 1.04))
fig.suptitle('Efficiency frontier (top-left = faster AND better quality)', fontsize=13, y=1.08)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_efficiency_frontier.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 4: Time heatmaps ----
print('  Fig 4: time heatmaps')
fig, axes = plt.subplots(nr_h, nc_h, figsize=(5.5*nc_h, 4.5*nr_h))
for i, lab in enumerate(METHOD_ORDER):
    ax = axes.flat[i]
    sub = df[df['method'] == lab]
    piv = sub.groupby(['K', 'n_obs'])['time'].median().reset_index()
    piv = piv.pivot(index='K', columns='n_obs', values='time')
    sns.heatmap(piv, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 's'})
    ax.set_title(lab, fontweight='bold', color=PALETTE[lab])
    ax.set_xlabel('n_obs'); ax.set_ylabel('K')
for i in range(n_m, nr_h*nc_h): axes.flat[i].set_visible(False)
fig.suptitle('Median computation time (seconds)', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_heatmap_time.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 5: Speedup heatmaps ----
print('  Fig 5: speedup heatmaps')
fig, axes = plt.subplots(nr_h, nc_h, figsize=(5.5*nc_h, 4.5*nr_h))
for i, lab in enumerate(METHOD_ORDER):
    ax = axes.flat[i]
    sub = df[df['method'] == lab]
    piv = sub.groupby(['K', 'n_obs'])['speedup'].median().reset_index()
    piv = piv.pivot(index='K', columns='n_obs', values='speedup')
    vmax = max(float(piv.max().max()), 2.0)
    sns.heatmap(piv, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                center=1.0, vmin=0.3, vmax=vmax, cbar_kws={'label': '×'})
    ax.set_title(lab, fontweight='bold', color=PALETTE[lab])
    ax.set_xlabel('n_obs'); ax.set_ylabel('K')
for i in range(n_m, nr_h*nc_h): axes.flat[i].set_visible(False)
fig.suptitle('Speedup vs LSA (green = faster, red = slower)', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_heatmap_speedup.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 6: Relative MSE heatmaps ----
print('  Fig 6: relative MSE heatmaps')
fig, axes = plt.subplots(nr_h, nc_h, figsize=(5.5*nc_h, 4.5*nr_h))
for i, lab in enumerate(METHOD_ORDER):
    ax = axes.flat[i]
    sub = df[df['method'] == lab]
    piv = sub.groupby(['K', 'n_obs'])['rel_mse'].median().reset_index()
    piv = piv.pivot(index='K', columns='n_obs', values='rel_mse')
    vmax = max(float(piv.max().max()), 1.5)
    sns.heatmap(piv, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                center=1.0, vmin=0.5, vmax=vmax, cbar_kws={'label': 'rel. MSE'})
    ax.set_title(lab, fontweight='bold', color=PALETTE[lab])
    ax.set_xlabel('n_obs'); ax.set_ylabel('K')
for i in range(n_m, nr_h*nc_h): axes.flat[i].set_visible(False)
fig.suptitle('Relative MSE vs LSA (green = better quality, red = worse)', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_heatmap_rel_mse.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 7: Time vs K grid ----
print('  Fig 7: time vs K grid')
fig, axes = plt.subplots(len(ALPHA_VALUES), len(N_OBS_VALUES),
                         figsize=(4*len(N_OBS_VALUES), 3.8*len(ALPHA_VALUES)), sharex=True)
for i, alpha in enumerate(ALPHA_VALUES):
    for j, n_obs in enumerate(N_OBS_VALUES):
        ax = axes[i, j]
        sub = df[(df['alpha'] == alpha) & (df['n_obs'] == n_obs)]
        for lab in METHOD_ORDER:
            d = sub[sub['method'] == lab]
            if d.empty: continue
            ax.plot(d['K'], d['time'], 'o-', label=lab, color=PALETTE[lab], lw=1.5, ms=4)
        ax.set_title(f'α={alpha}, n={n_obs}', fontsize=10)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(K_VALUES); ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        if j == 0: ax.set_ylabel('Time (s)')
        if i == len(ALPHA_VALUES) - 1: ax.set_xlabel('K')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(.5, 1.03))
fig.suptitle('Computation time vs K', fontsize=14, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_time_vs_K_grid.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 8: MSE vs K grid ----
print('  Fig 8: MSE vs K grid')
fig, axes = plt.subplots(len(ALPHA_VALUES), len(N_OBS_VALUES),
                         figsize=(4*len(N_OBS_VALUES), 3.8*len(ALPHA_VALUES)), sharex=True)
for i, alpha in enumerate(ALPHA_VALUES):
    for j, n_obs in enumerate(N_OBS_VALUES):
        ax = axes[i, j]
        sub = df[(df['alpha'] == alpha) & (df['n_obs'] == n_obs)]
        for lab in METHOD_ORDER:
            d = sub[sub['method'] == lab]
            if d.empty: continue
            ax.plot(d['K'], d['mse_loc'], 'o-', label=lab, color=PALETTE[lab], lw=1.5, ms=4)
        ax.set_title(f'α={alpha}, n={n_obs}', fontsize=10)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(K_VALUES); ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        if j == 0: ax.set_ylabel('MSE (local params)')
        if i == len(ALPHA_VALUES) - 1: ax.set_xlabel('K')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(.5, 1.03))
fig.suptitle('Posterior MSE (local parameters) vs K', fontsize=14, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_mse_vs_K_grid.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 9: Final ε vs K ----
print('  Fig 9: final eps vs K')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, alpha in zip(axes, ALPHA_VALUES):
    agg = df[df['alpha'] == alpha].groupby(['K', 'method'])['final_eps'].median().reset_index()
    for lab in METHOD_ORDER:
        d = agg[agg['method'] == lab]
        if d.empty: continue
        ax.plot(d['K'], d['final_eps'], 'o-', label=lab, color=PALETTE[lab], lw=2, ms=5)
    ax.set_xlabel('K'); ax.set_title(f'α = {alpha}', fontsize=13)
    ax.set_xscale('log'); ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[0].set_ylabel('Final ε (median over n_obs)')
axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
fig.suptitle('Convergence quality: final ε reached', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_final_eps_vs_K.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 10: Best method ----
print('  Fig 10: best method')
non_lsa = df[df['method'] != 'LSA'].copy()
non_lsa['efficiency'] = non_lsa['speedup'] / non_lsa['rel_mse'].clip(lower=0.01)
best_idx = non_lsa.groupby(['K', 'n_obs', 'alpha'])['efficiency'].idxmax()
best = non_lsa.loc[best_idx]
best_counts = best.groupby('method').size().reindex(METHOD_ORDER[1:]).fillna(0).astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.2]})
colors = [PALETTE.get(m, '#999') for m in best_counts.index]
bars = ax1.barh(best_counts.index, best_counts.values, color=colors)
ax1.set_xlabel('# configs where method is most efficient')
ax1.set_title('Best method by efficiency\n(speedup / relative_MSE, excl. LSA)', fontsize=12)
ax1.invert_yaxis()
for bar, v in zip(bars, best_counts.values):
    if v > 0: ax1.text(v + 0.3, bar.get_y() + bar.get_height()/2, str(v), va='center')

mode_df = best.groupby(['K', 'alpha'])['method'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
).reset_index()
pivot = mode_df.pivot(index='K', columns='alpha', values='method')
pivot = pivot[sorted(pivot.columns, reverse=True)]
m2i = {m: i for i, m in enumerate(METHOD_ORDER)}
cmap_l = mpl.colors.ListedColormap([PALETTE[m] for m in METHOD_ORDER])
piv_int = pivot.map(lambda x: m2i.get(x, -1))
ax2.imshow(piv_int.values, cmap=cmap_l, vmin=0, vmax=len(METHOD_ORDER)-1, aspect='auto')
ax2.set_xticks(range(pivot.shape[1])); ax2.set_xticklabels(pivot.columns)
ax2.set_yticks(range(pivot.shape[0])); ax2.set_yticklabels(pivot.index)
ax2.set_xlabel('α (epsilon quantile)'); ax2.set_ylabel('K')
ax2.set_title('Best method per (K, α) regime\n(mode over n_obs)', fontsize=12)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        txt = pivot.iloc[i, j]
        ax2.text(j, i, txt, ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2, foreground='black')])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_best_method.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 11: Radar ----
print('  Fig 11: radar')
rdf = df.groupby('method').agg(
    speedup_med=('speedup', 'median'), rel_mse_med=('rel_mse', 'median'),
    n_iters_med=('n_iters', 'median'), nsim_med=('total_nsim', 'median'),
).reindex(METHOD_ORDER).copy()
rdf['inv_rel_mse'] = 1 / rdf['rel_mse_med'].clip(lower=0.01)
rdf['inv_iters']   = 1 / rdf['n_iters_med'].clip(lower=1)
rdf['inv_nsim']    = 1 / rdf['nsim_med'].clip(lower=1)
cols = ['speedup_med', 'inv_rel_mse', 'inv_iters', 'inv_nsim']
for c in cols:
    mn, mx = rdf[c].min(), rdf[c].max()
    rdf[c] = (rdf[c] - mn) / (mx - mn + 1e-12)
angles = np.linspace(0, 2*np.pi, len(cols), endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})
for lab in METHOD_ORDER:
    vals = rdf.loc[lab, cols].values.tolist() + [rdf.loc[lab, cols[0]]]
    ax.plot(angles, vals, 'o-', label=lab, color=PALETTE[lab], lw=2)
    ax.fill(angles, vals, alpha=0.05, color=PALETTE[lab])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(['Speed', 'Quality', '1/Iters', '1/N_sim'], fontsize=12)
ax.set_title('Method comparison (normalised, higher = better)', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_radar.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 12: Smart strategies nsim ratio ----
print('  Fig 12: smart nsim ratio')
smart_methods = ['Smart Swap', 'Smart Hilbert', 'Smart H+S']
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, alpha in zip(axes, ALPHA_VALUES):
    sub = df[(df['alpha'] == alpha) & df['method'].isin(smart_methods)]
    agg = sub.groupby(['K', 'method'])['nsim_ratio'].median().reset_index()
    for lab in smart_methods:
        d = agg[agg['method'] == lab]
        if d.empty: continue
        ax.plot(d['K'], d['nsim_ratio'], 'o-', label=lab, color=PALETTE[lab], lw=2, ms=6)
    ax.axhline(1, ls='--', color=PALETTE['LSA'], alpha=.5, label='LSA (baseline)')
    ax.set_xlabel('K'); ax.set_title(f'α = {alpha}', fontsize=13)
    ax.set_xscale('log'); ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
axes[0].set_ylabel('N_sim ratio vs LSA')
axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
fig.suptitle('Smart strategies: simulation budget relative to LSA', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_smart_nsim_ratio.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 13: Bar chart by K regime ----
print('  Fig 13: bar by K regime')
df['K_regime'] = pd.cut(df['K'], bins=[0, 5, 15, 100],
                        labels=['Small (K≤5)', 'Medium (K≤15)', 'Large (K>15)'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for ax, metric, title, ylabel in [
    (ax1, 'speedup', 'Mean speedup by K regime', 'Speedup vs LSA'),
    (ax2, 'rel_mse', 'Mean relative MSE by K regime', 'Relative MSE'),
]:
    piv = df[df['method'] != 'LSA'].groupby(['K_regime', 'method'])[metric].mean().reset_index()
    piv = piv.pivot(index='method', columns='K_regime', values=metric)
    piv = piv.reindex(METHOD_ORDER[1:])
    piv.plot(kind='bar', ax=ax, width=0.7, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=13); ax.set_ylabel(ylabel); ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    ax.axhline(1, ls='--', color='grey', alpha=.4)
    ax.legend(title='K regime', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_bar_regime.pdf'), bbox_inches='tight')
plt.close()

# ---- Fig 14: Iterations heatmap ----
print('  Fig 14: iterations heatmap')
fig, axes = plt.subplots(nr_h, nc_h, figsize=(5.5*nc_h, 4.5*nr_h))
for i, lab in enumerate(METHOD_ORDER):
    ax = axes.flat[i]
    sub = df[df['method'] == lab]
    piv = sub.groupby(['K', 'n_obs'])['n_iters'].median().reset_index()
    piv = piv.pivot(index='K', columns='n_obs', values='n_iters')
    sns.heatmap(piv, annot=True, fmt='.0f', cmap='Blues', ax=ax, cbar_kws={'label': 'iters'})
    ax.set_title(lab, fontweight='bold', color=PALETTE[lab])
    ax.set_xlabel('n_obs'); ax.set_ylabel('K')
for i in range(n_m, nr_h*nc_h): axes.flat[i].set_visible(False)
fig.suptitle('Median number of SMC iterations', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_heatmap_iters.pdf'), bbox_inches='tight')
plt.close()

print(f'\nAll figures saved to {FIG_DIR}')
