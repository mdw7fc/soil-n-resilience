#!/usr/bin/env python3
"""SC1 (permanent 20% physical supply loss) and SC2 (20% loss, 20-year recovery)
regional yield-loss trajectories under the canonical ERA5 climate. Writes
../../data/SC1_regional_trajectory.csv and SC2_regional_trajectory.csv, and
prints production-weighted global year-10 loss for each."""
import os, sys, json, csv, warnings
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,'..','model'))
import numpy as np
from monthly_model_v3 import MonthlyClimate, MonthlyNParams, REGIONAL_CLIMATES
from coupled_monthly import CoupledMonthlyModel, get_calibrated_ym
from coupled_econ_biophysical import get_supply_constrained_scenarios
from soil_n_model import get_default_regions
DATA=os.path.join(HERE,'..','..','data')
RO=['north_america','europe','east_asia','south_asia','southeast_asia','latin_america','sub_saharan_africa','fsu_central_asia']
clim=json.load(open(os.path.join(DATA,'era5_regional_climates.json')))
for k,c in list(REGIONAL_CLIMATES.items()):
    n=clim[k]; REGIONAL_CLIMATES[k]=MonthlyClimate(c.name,list(map(float,n['temp'])),list(map(float,n['precip'])),list(map(float,n['pet'])),c.planting_month,c.maturity_month)
regions=get_default_regions(); mp=MonthlyNParams()
scen=get_supply_constrained_scenarios()
areas=np.array([regions[k].cropland_mha for k in RO])
for name,econ in [('SC1',scen['SC1_20pct']),('SC2',scen['SC2_20pct_recovery'])]:
    years=list(range(0,31)); traj={}
    ybase=[]
    for rk in RO:
        df=CoupledMonthlyModel(region=regions[rk],econ=econ,region_key=rk,t_max=30.0,
                               yield_max_override=get_calibrated_ym(rk,mp)).run()
        traj[rk]=[float((1-df[df['year']==y]['yield_fraction'].iloc[0])*100) for y in years]
        ybase.append(float(df[df['year']==0]['yield_tha'].iloc[0]))
    with open(os.path.join(DATA,f'{name}_regional_trajectory.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['year']+[regions[k].name for k in RO])
        for i,y in enumerate(years): w.writerow([y]+[round(traj[k][i],3) for k in RO])
    W=areas*np.array(ybase); W/=W.sum()
    g10=sum(traj[k][10]*W[j] for j,k in enumerate(RO))
    print(f'{name}: production-weighted global year-10 loss = {g10:.2f}%')
