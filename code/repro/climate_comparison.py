#!/usr/bin/env python3
"""Climate-input robustness: expert (representative) vs ERA5 data-based climate,
comparing year-1 and year-10 S3 losses.

Writes outputs/climate_swap_comparison.csv and results/climate_swap_stats.txt.
The two headline numbers are deposited rather than printed: F-009 records what
happens when a number reaches a document by being read off a console, and the
0.74 pp that MANIFEST.md carried for three generations of this analysis is the
example. Read the current values out of the stats file, not out of this
docstring and not out of a run log."""
import os, sys, json, csv, warnings
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,'..','model'))
import numpy as np
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params, calibrate_price_shock
from soil_n_model import get_default_regions
from scipy import stats
RO=['north_america','europe','east_asia','south_asia','southeast_asia','latin_america','sub_saharan_africa','fsu_central_asia']
def run(t_max=10.0):
    regions=get_default_regions(); mp=MonthlyNParams()
    s3=get_scenario_params()['S3']; s3.fert_price_shock=calibrate_price_shock(0.20)
    out={}
    for rk in RO:
        r=regions[rk]; ym=get_calibrated_ym(rk,mp)
        df=CoupledMonthlyModel(region=r,econ=s3,region_key=rk,t_max=t_max,yield_max_override=ym).run()
        f=lambda yr:(1-df[df['year']==yr]['yield_fraction'].iloc[0])*100
        out[rk]=dict(ybase=float(df[df['year']==0]['yield_tha'].iloc[0]),l1=float(f(1)),l10=float(f(10)))
    return out
old=run()                                   # expert representative profiles
apply_era5_climate_file(os.path.join(
    HERE,'..','..','data','era5_regional_climates.json'))
new=run()                                   # ERA5 data-based
d10=[new[k]['l10']-old[k]['l10'] for k in RO]
os.makedirs(os.path.join(HERE,'..','..','outputs'),exist_ok=True)
with open(os.path.join(HERE,'..','..','outputs','climate_swap_comparison.csv'),'w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['region','Ybase_expert','Ybase_ERA5','yr1_loss_expert','yr1_loss_ERA5','yr10_loss_expert','yr10_loss_ERA5','d_yr10_pp'])
    for k in RO: w.writerow([k,round(old[k]['ybase'],2),round(new[k]['ybase'],2),round(old[k]['l1'],2),round(new[k]['l1'],2),round(old[k]['l10'],2),round(new[k]['l10'],2),round(new[k]['l10']-old[k]['l10'],2)])
shift = max(abs(x) for x in d10)
rho = stats.spearmanr([old[k]['l10'] for k in RO],
                      [new[k]['l10'] for k in RO]).correlation
os.makedirs(os.path.join(HERE,'..','..','results'),exist_ok=True)
stats_path = os.path.join(HERE,'..','..','results','climate_swap_stats.txt')
with open(stats_path,'w') as fh:
    fh.write('max_abs_year10_shift_pp\t%.2f\n' % shift)
    fh.write('spearman_rho_year10_ranking\t%.2f\n' % rho)
print('max |year-10 shift| = %.2f pp ; Spearman rho (ranking) = %.2f' % (shift, rho))
print('wrote %s' % stats_path)
