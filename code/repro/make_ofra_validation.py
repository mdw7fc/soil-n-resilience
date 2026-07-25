import csv, numpy as np, os
HERE=os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROWS=list(csv.DictReader(open(os.path.join(HERE,'..','..','data','ofra_maize_N_responsefunctions.csv'))))
def valid(r):
    try:
        A=float(r['A_control']);B=float(r['B_gain']);C=float(r['C_base'])
        return 0<C<1 and A>=0 and B>0
    except: return False
funcs=[r for r in ROWS if valid(r)]
N=np.linspace(0,120,61)
Y=np.array([float(r['A_control'])+float(r['B_gain'])*(1-float(r['C_base'])**N) for r in funcs])
med=np.median(Y,0); p25=np.percentile(Y,25,0); p75=np.percentile(Y,75,0)
p10=np.percentile(Y,10,0); p90=np.percentile(Y,90,0)
calibration={r['region']:r for r in csv.DictReader(open(os.path.join(
    HERE,'..','..','outputs','Table_S4_calibration_sol.csv')))}
ssa=calibration['sub_saharan_africa']
YMAX=float(ssa['calibrated_y_max_t_ha'])
CTRL=float(ssa['simulated_year2_no_synth_n_t_ha'])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':12})
fig,ax=plt.subplots(figsize=(11,6.8))
ax.fill_between(N,p10,p90,color='#4C6EA0',alpha=0.15,label=f'OFRA maize-N 10–90th pct (n={len(funcs)})')
ax.fill_between(N,p25,p75,color='#4C6EA0',alpha=0.35,label='OFRA maize-N IQR')
ax.plot(N,med,color='#1A2E5A',lw=2.5,label='OFRA maize-N median')
ax.axhline(YMAX,color='#C0392B',lw=2,ls='--',label=f'Model SSA ceiling (y_max={YMAX:.2f})')
ax.axhline(CTRL,color='#C0392B',lw=2,ls=':',label=f'Model SSA control (no synth N ≈{CTRL:.1f})')
ax.set_xlim(-6,124); ax.set_ylim(0,8)
ax.set_xlabel('Applied N (kg N ha$^{-1}$)'); ax.set_ylabel('Maize grain yield (t ha$^{-1}$)')
ax.set_title('SSA maize N-response: model vs OFRA on-farm database (Wortmann et al.)',fontsize=13)
ax.legend(loc='upper left',fontsize=11,framealpha=0.9)
ax.text(0.99,0.02,'Model ceiling sits below the OFRA median but within the interquartile range;\nthe no-synthetic-N control sits at the low edge, consistent with degraded regional-mean soils.\nNote: model x-axis is N availability/uptake; OFRA is applied N.',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=9,color='#555555',style='italic')
plt.tight_layout()
fig.savefig(os.path.join(HERE,'..','..','figures','Figure_S13_OFRA_SSA_validation.png'),dpi=200,bbox_inches='tight',facecolor='white')
print('n=%d  median[0]=%.2f median[-1]=%.2f  IQR[-1]=[%.2f,%.2f]'%(len(funcs),med[0],med[-1],p25[-1],p75[-1]))
print('ceiling %.3f vs median[-1] %.2f (below median) ; within IQR [%.2f,%.2f]: %s'
      %(YMAX,med[-1],p25[-1],p75[-1],p25[-1]<=YMAX<=p75[-1]))
