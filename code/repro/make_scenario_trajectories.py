#!/usr/bin/env python3
"""Write data/scenario_trajectories.csv: year (0-30) x production-weighted global
yield loss (%) for S3 (price-mediated), SC1 (permanent 20% supply loss), SC2
(20% loss, 20-yr recovery) and PULSE1 (the S3 shock held for one year and then
removed), plus per-region S3 loss.

PULSE1 was written in the v15 tree that was lost, and its capability died with
it (F-018). It is rebuilt here on the SupplyState seam rather than on a
one-year recovery ramp, which would decay the shock through the very year it is
meant to be at full strength. Year 1 is S3's year 1 by construction, which is
the check that the rebuild is the same scenario: the recovered F-016 narrative
records 2.316 for both, and that number belongs to the superseded eps_F_N
family, so the two columns agreeing with each other matters and their agreeing
with 2.316 does not."""
import os, sys, json, csv, warnings
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,'..','model'))
import numpy as np
from monthly_model_v3 import MonthlyNParams, apply_era5_climate_file
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_scenario_params, calibrate_price_shock, get_supply_constrained_scenarios, get_pulse_scenario
from soil_n_model import get_default_regions
DATA=os.path.join(HERE,'..','..','data')
RO=['north_america','europe','east_asia','south_asia','southeast_asia','latin_america','sub_saharan_africa','fsu_central_asia']
apply_era5_climate_file(os.path.join(DATA,'era5_regional_climates.json'))
reg=get_default_regions(); mp=MonthlyNParams()
area=np.array([reg[k].cropland_mha for k in RO])
def traj(econ):
    per={}; yb=[]
    for k in RO:
        df=CoupledMonthlyModel(region=reg[k],econ=econ,region_key=k,t_max=30.0,yield_max_override=get_calibrated_ym(k,mp)).run()
        per[k]=[float((1-df[df['year']==y]['yield_fraction'].iloc[0])*100) for y in range(31)]
        yb.append(float(df[df['year']==0]['yield_tha'].iloc[0]))
    W=area*np.array(yb); W/=W.sum()
    g=[sum(per[k][y]*W[j] for j,k in enumerate(RO)) for y in range(31)]
    return per,g
s3=get_scenario_params()['S3']; s3.fert_price_shock=calibrate_price_shock(0.20)
sc=get_supply_constrained_scenarios()
p3,g3=traj(s3); _,g1=traj(sc['SC1_20pct']); _,g2=traj(sc['SC2_20pct_recovery']); _,gp=traj(get_pulse_scenario())
assert abs(gp[1]-g3[1])<1e-6, 'PULSE1 year 1 must equal S3 year 1; got %r vs %r'%(gp[1],g3[1])
with open(os.path.join(DATA,'scenario_trajectories.csv'),'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['year','S3_global','SC1_global','SC2_global','PULSE1_global']+['S3_'+reg[k].name.replace(' ','_') for k in RO])
    for y in range(31):
        w.writerow([y,round(g3[y],3),round(g1[y],3),round(g2[y],3),round(gp[y],3)]+[round(p3[k][y],3) for k in RO])
print('scenario_trajectories.csv: S3 yr10/30 = %.2f / %.2f ; SC1 yr10 = %.2f ; SC2 yr10 = %.2f ; PULSE1 yr1/2/5 = %.3f / %.3f / %.3f'%(g3[10],g3[30],g1[10],g2[10],gp[1],gp[2],gp[5]))
