"""Empirical anchor for f_fert (fertilizer-era share of current SOM mineralization)
using the Broadbalk Wheat Experiment long-term SOC trajectories.

Logic
-----
The Broadbalk Nil plot (P3) has received no N, P, or K since 1843 and provides
the closest empirical counterfactual to a fertilizer-free cropland steady state.
The N3PK (P8) and FYM+N3 (P2.1) plots have received synthetic N and/or manure
continuously and are the fertilizer-era end-members.

At approximate steady state:

    J_min(plot) = k_eff(plot) * SOM(plot)

where k_eff is the mean decomposition rate of the SOM stock and SOM is the
total SOM-N pool.  The fertilizer-era share of mineralization on a fertilized
plot is then:

    f_fert_flux = 1 - (k_eff_Nil * SOM_Nil) / (k_eff_fert * SOM_fert)

If we assume k_eff_fert >= k_eff_Nil (fresh fertilizer-era SOM turns over faster
than aged pre-industrial SOM), the ratio (k_eff_Nil / k_eff_fert) is bounded
above by 1.0 and below by the Century-active / Century-slow turnover ratio
(roughly 0.5).  We report f_fert under three assumptions and use them as a
bracket for the Monte Carlo prior.

Author: Matthew Wallenstein
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
BB = ROOT / 'model' / 'data' / 'benchmark_broadbalk' / 'soc_trajectories_broadbalk.csv'
OUT = ROOT / 'model' / 'data' / 'fermi_validation'
OUT.mkdir(parents=True, exist_ok=True)

# Broadbalk treatment -> plot mapping (must match benchmark_broadbalk.py)
PAIRS = [
    # fert_treatment, fert_plot, label
    ('N3PK',    'P8',   'Mineral fertilizer (N3PK)'),
    ('FYM1843', 'P2.2', 'FYM since 1843'),
    ('FYM+N3',  'P2.1', 'FYM + mineral N3'),
]
NIL = ('Nil', 'P3')

# Century-style pool turnover rate ratios (1/yr).
# Active SOM k ~ 0.20/yr, slow SOM k ~ 0.02/yr, passive SOM k ~ 0.0009/yr.
# Fertilizer-era SOM (post-1960) is roughly 60% slow / 30% active / 10% passive.
# Native SOM (aged >1000 yr) is roughly 15% slow / 5% active / 80% passive.
# Weighted k_eff ratios: (fertilizer-era / native) ~ 4-5x.  For a plot that is
# 170 yr old (Broadbalk), native-like SOM is still present, so the effective
# ratio of k_eff_fert / k_eff_Nil is more modest, ~1.2-1.8.
K_EFF_RATIO_LO  = 1.0    # lower bound: same turnover rates
K_EFF_RATIO_MID = 1.35   # central: fresh SOM turns over ~35% faster
K_EFF_RATIO_HI  = 1.8    # upper bound: stronger enrichment in fast pool


def main() -> None:
    df = pd.read_csv(BB)

    # Use the final observed year per treatment
    results = []
    for treatment, plot, label in PAIRS:
        sub = df[(df['treatment'] == treatment) & (df['plot'] == plot)]
        nil = df[(df['treatment'] == NIL[0]) & (df['plot'] == NIL[1])]
        if sub.empty or nil.empty:
            continue

        # Take the last year with observed SOC in the fertilized plot
        obs_rows = sub.dropna(subset=['soc_obs'])
        if obs_rows.empty:
            final_year = int(sub['year'].max())
        else:
            final_year = int(obs_rows['year'].max())

        soc_fert_obs = sub.loc[sub['year'] == final_year, 'soc_obs'].values
        soc_fert_cen = sub.loc[sub['year'] == final_year, 'soc_century'].values[0]
        soc_fert_mems = sub.loc[sub['year'] == final_year, 'soc_mems'].values[0]

        # Nil plot SOC at the same year
        nil_at_year = nil.loc[nil['year'] == final_year]
        if nil_at_year.empty:
            continue
        soc_nil_obs = nil_at_year['soc_obs'].values
        soc_nil_cen = nil_at_year['soc_century'].values[0]
        soc_nil_mems = nil_at_year['soc_mems'].values[0]

        # Fall back to model if obs missing
        soc_fert_obs = soc_fert_obs[0] if len(soc_fert_obs) else np.nan
        soc_nil_obs = soc_nil_obs[0] if len(soc_nil_obs) else np.nan

        # Pick best available SOC estimates
        def pick(obs, cen, mems):
            if not np.isnan(obs):
                return obs, 'observed'
            return (cen + mems) / 2.0, 'model-mean'

        soc_fert, src_fert = pick(soc_fert_obs, soc_fert_cen, soc_fert_mems)
        soc_nil,  src_nil  = pick(soc_nil_obs,  soc_nil_cen,  soc_nil_mems)

        # Stock-basis fertilizer-era share
        stock_frac = max(0.0, 1.0 - soc_nil / soc_fert)

        # Flux-basis with three k_eff ratios
        # f_fert_flux = 1 - (k_Nil * SOC_Nil) / (k_fert * SOC_fert)
        #             = 1 - (1 / ratio) * (SOC_Nil / SOC_fert)
        f_lo  = max(0.0, 1.0 - (1.0 / K_EFF_RATIO_LO)  * (soc_nil / soc_fert))
        f_mid = max(0.0, 1.0 - (1.0 / K_EFF_RATIO_MID) * (soc_nil / soc_fert))
        f_hi  = max(0.0, 1.0 - (1.0 / K_EFF_RATIO_HI)  * (soc_nil / soc_fert))

        results.append({
            'treatment': treatment,
            'plot': plot,
            'label': label,
            'year': final_year,
            'soc_fert_t_ha': soc_fert,
            'src_fert': src_fert,
            'soc_nil_t_ha': soc_nil,
            'src_nil': src_nil,
            'stock_frac': stock_frac,
            'f_fert_flux_lo':  f_lo,
            'f_fert_flux_mid': f_mid,
            'f_fert_flux_hi':  f_hi,
        })

    out_df = pd.DataFrame(results)
    out_csv = OUT / 'broadbalk_f_fert_empirical.csv'
    out_df.to_csv(out_csv, index=False)

    # Summary
    print('Broadbalk empirical f_fert (fertilizer-era share of current SOM mineralization)')
    print('=' * 88)
    for _, r in out_df.iterrows():
        print(f'{r["label"]:<35} ({r["treatment"]}, {r["plot"]})  yr {r["year"]}')
        print(f'  SOC_fert = {r["soc_fert_t_ha"]:.1f} t C/ha ({r["src_fert"]}); '
              f'SOC_Nil = {r["soc_nil_t_ha"]:.1f} t C/ha ({r["src_nil"]})')
        print(f'  Stock-basis fertilizer-era share:  {r["stock_frac"]*100:.1f}%')
        print(f'  Flux-basis bracket (k ratios {K_EFF_RATIO_LO}/{K_EFF_RATIO_MID}/{K_EFF_RATIO_HI}): '
              f'{r["f_fert_flux_lo"]*100:.1f}% / {r["f_fert_flux_mid"]*100:.1f}% / '
              f'{r["f_fert_flux_hi"]*100:.1f}%')
        print()

    # Pooled central estimate
    pooled_mid = out_df['f_fert_flux_mid'].mean()
    pooled_lo  = out_df['f_fert_flux_lo'].min()
    pooled_hi  = out_df['f_fert_flux_hi'].max()
    print(f'Pooled empirical bracket: {pooled_lo*100:.0f}% - {pooled_hi*100:.0f}% '
          f'(central {pooled_mid*100:.0f}%)')
    print(f'Written: {out_csv}')


if __name__ == '__main__':
    main()
