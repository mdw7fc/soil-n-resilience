#!/usr/bin/env python3
"""Rebuild Fig 6 (main) and Supp Fig S4 (flux decomposition) under matched
20% SC1 scenario using output of run_matched_mems_comparison.py."""
import sys, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJ = REPO_ROOT  # alias
with open(PROJ/'data'/'matched_mems_century.pkl','rb') as f: data = pickle.load(f)
with open(PROJ/'data'/'mems_flux_decomposition.pkl','rb') as f: flux = pickle.load(f)

REGIONS = ['north_america','europe','east_asia','south_asia','southeast_asia',
           'latin_america','sub_saharan_africa','fsu_central_asia']
LBL = {'north_america':'North America','europe':'Europe','east_asia':'East Asia',
       'south_asia':'South Asia','southeast_asia':'SE Asia','latin_america':'Latin America',
       'sub_saharan_africa':'Sub-Saharan Africa','fsu_central_asia':'FSU & Central Asia'}
KEY4 = ['north_america','south_asia','sub_saharan_africa','latin_america']
KEY4_LBL = ['North America','South Asia','Sub-Saharan Africa','Latin America']

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica','Arial','DejaVu Sans'],
                     'axes.spines.top': False, 'axes.spines.right': False})

# === Fig 6: matched-scenario SOC trajectories + bar of amplification ratios ===
fig = plt.figure(figsize=(12, 9))
gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.30, height_ratios=[1, 1.1])

col_cen = '#2166ac'; col_mems = '#b2182b'
for idx, rn in enumerate(KEY4):
    ax = fig.add_subplot(gs[0, idx])
    c_df = data['century_shocked'][rn]
    m_df = data['mems_shocked'][rn]
    c_soc0 = c_df[c_df['year']==0]['soc_total'].iloc[0] if 0 in c_df['year'].values else c_df['soc_total'].iloc[0]
    m_soc0 = m_df[m_df['year']==0]['soc_total'].iloc[0] if 0 in m_df['year'].values else m_df['soc_total'].iloc[0]
    ax.plot(c_df['year'], c_df['soc_total']/c_soc0, color=col_cen, linewidth=1.8, label='Century (3-pool)')
    ax.plot(m_df['year'], m_df['soc_total']/m_soc0, color=col_mems, linewidth=1.8, linestyle='--', label='MEMS (4-pool)')
    ax.set_title(KEY4_LBL[idx], fontsize=10, fontweight='bold')
    if idx == 0: ax.set_ylabel('SOC (fraction of year-0)', fontsize=9)
    ax.set_xlabel('Years', fontsize=8)
    ax.set_xlim(0, 30); ax.set_ylim(0.85, 1.01)
    ax.axhline(1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
    if idx == 0: ax.legend(fontsize=7.5, loc='lower left')
    ax.text(-0.12, 1.08, chr(ord('a') + idx), transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

# Panel e: amplification ratios across all 8 regions
ax_e = fig.add_subplot(gs[1, 1:3])
ratios = []
labels = []
for rn in REGIONS:
    c_loss = data['century_shocked'][rn]['soc_total'].iloc[0] - data['century_shocked'][rn][data['century_shocked'][rn]['year']==30]['soc_total'].iloc[0]
    m_loss = data['mems_shocked'][rn]['soc_total'].iloc[0] - data['mems_shocked'][rn][data['mems_shocked'][rn]['year']==30]['soc_total'].iloc[0]
    r = m_loss / c_loss if c_loss > 0 else float('nan')
    ratios.append(r); labels.append(LBL[rn].replace('Sub-Saharan Africa','SSA').replace('FSU & Central Asia','FSU'))

order = sorted(range(len(ratios)), key=lambda i: ratios[i])
rs = [ratios[i] for i in order]; ls = [labels[i] for i in order]
bars = ax_e.barh(range(len(rs)), rs, color=['#d32f2f' if r<1.0 else '#1565C0' for r in rs], alpha=0.85, edgecolor='black', linewidth=0.5)
for i, r in enumerate(rs):
    ax_e.text(r + 0.03, i, f'{r:.2f}×', va='center', fontsize=8)
ax_e.axvline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
ax_e.axvline(2.0, color='gray', linewidth=0.8, linestyle=':', alpha=0.4)
ax_e.set_yticks(range(len(ls))); ax_e.set_yticklabels(ls, fontsize=8)
ax_e.set_xlabel('MEMS / Century 30-year SOC loss ratio', fontsize=9)
ax_e.set_xlim(0, max(rs)*1.18)
ax_e.text(-0.2, 1.05, 'e', transform=ax_e.transAxes, fontsize=11, fontweight='bold', va='top')
ax_e.set_title(f'Matched scenario: SC1 (20% sustained supply reduction)\nRange: {min(rs):.2f}× to {max(rs):.2f}× | Median: {float(np.median(rs)):.2f}×', fontsize=9.5)

plt.tight_layout()
out = PROJ/'figures'/'figureS5_mems_flux_panelA'
fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out}.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}.png')

# === Supp Fig S4: MEMS flux decomposition ===
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios':[1.3, 1]})

# Panel a: stacked bar of cumulative respiration by mechanism (CUE vs necromass) per region
cue_vals = [flux[rn]['cum_resp_cue'] for rn in REGIONS]
necro_vals = [flux[rn]['cum_resp_necro'] for rn in REGIONS]
# Sort regions by total respiration
sort_ord = sorted(range(len(REGIONS)), key=lambda i: cue_vals[i]+necro_vals[i], reverse=True)
cue_s = [cue_vals[i] for i in sort_ord]
necro_s = [necro_vals[i] for i in sort_ord]
labels_s = [LBL[REGIONS[i]].replace('Sub-Saharan Africa','SSA').replace('FSU & Central Asia','FSU') for i in sort_ord]

y = np.arange(len(REGIONS))
ax_a.barh(y, cue_s, color='#E65100', label='CUE respiration (DOM → microbe)', edgecolor='black', linewidth=0.4)
ax_a.barh(y, necro_s, left=cue_s, color='#6A1B9A', label='Non-recycled necromass', edgecolor='black', linewidth=0.4)
for i, (cv, nv) in enumerate(zip(cue_s, necro_s)):
    total = cv + nv
    pct_cue = cv/total*100 if total>0 else 0
    ax_a.text(total + 2, i, f'CUE {pct_cue:.0f}%', va='center', fontsize=7, color='#333333')
ax_a.set_yticks(y); ax_a.set_yticklabels(labels_s, fontsize=8.5)
ax_a.invert_yaxis()
ax_a.set_xlabel('Cumulative 30-year respired C (t C ha⁻¹)', fontsize=9)
ax_a.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax_a.text(-0.2, 1.05, 'a', transform=ax_a.transAxes, fontsize=11, fontweight='bold', va='top')
ax_a.set_title('Where carbon is lost in MEMS under 20% shock', fontsize=10)

# Panel b: fraction of total respiration by mechanism
cue_pct = [flux[rn]['cue_pct_of_resp'] for rn in REGIONS]
necro_pct = [flux[rn]['necro_pct_of_resp'] for rn in REGIONS]
cue_pct_s = [cue_pct[i] for i in sort_ord]
necro_pct_s = [necro_pct[i] for i in sort_ord]

ax_b.barh(y, cue_pct_s, color='#E65100', edgecolor='black', linewidth=0.4)
ax_b.barh(y, necro_pct_s, left=cue_pct_s, color='#6A1B9A', edgecolor='black', linewidth=0.4)
ax_b.set_yticks(y); ax_b.set_yticklabels([]); ax_b.invert_yaxis()
ax_b.set_xlim(0, 105)
ax_b.set_xlabel('Share of total MEMS respiration (%)', fontsize=9)
ax_b.axvline(80, color='black', linestyle=':', linewidth=0.6, alpha=0.5)
ax_b.text(-0.08, 1.05, 'b', transform=ax_b.transAxes, fontsize=11, fontweight='bold', va='top')
ax_b.set_title('CUE respiration dominates (≥80% across regions)', fontsize=10)

plt.tight_layout()
out2 = PROJ/'figures'/'figureS5_mems_flux_panelB'
fig.savefig(f'{out2}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{out2}.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out2}.png')
