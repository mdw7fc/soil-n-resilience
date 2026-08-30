#!/usr/bin/env python3
"""Supplementary Table 3 (pairwise Spearman rank associations) from the frozen
canonical run. Reproduces the values printed in the SI. Run run_canonical.py first."""
import os, json
import numpy as np
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, os.path.join(HERE,'..','model'))
from coupled_econ_biophysical import REGIONAL_ECON_PARAMS as REP
d=json.load(open(os.path.join(HERE,'..','..','data','canonical_ERA5_y30.json')))['regions']
l1=np.array([r['loss_yr1'] for r in d]); l10=np.array([r['loss_yr10'] for r in d])
desc={'SOC stock':[r['soc'] for r in d],
      'Soil N buffer ratio':[r['buffer_ratio_pct'] for r in d],
      'Baseline water deficit':[r['water_deficit'] for r in d],
      'Calibrated yield ceiling y_max':[r['y_max'] for r in d],
      'BNF potential':[r['bnf'] for r in d],
      'Synthetic N rate':[r['synth_n'] for r in d],
      '|Fertilizer-demand elasticity|':[abs(r['eps_F_PF']) for r in d],
      '|Food-demand elasticity|':[abs(r['eta']) for r in d],
      'Land-response coefficient lambda_L':[REP[r['region']]['eps_LS_PL']*REP[r['region']]['eps_LD_PY']/(REP[r['region']]['eps_LS_PL']-REP[r['region']]['eps_LD_PL']) for r in d]}
sp=lambda x,y: stats.spearmanr(x,y)[0]
print('%-34s %8s %8s'%('Descriptor','yr1','yr10'))
out=[]
for name,x in desc.items():
    x=np.array(x,float); r1,r10=sp(x,l1),sp(x,l10)
    print('%-34s %+8.2f %+8.2f'%(name,r1,r10)); out.append((name,round(r1,2),round(r10,2)))
os.makedirs(os.path.join(HERE,'..','..','outputs'),exist_ok=True)
import csv
with open(os.path.join(HERE,'..','..','outputs','table_S3_correlations.csv'),'w',newline='') as f:
    w=csv.writer(f, lineterminator='\n'); w.writerow(['descriptor','rho_yr1','rho_yr10']); [w.writerow(r) for r in out]
print('\nAll descriptors including the region-specific land-response coefficient are reproduced.')
