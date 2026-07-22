import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
HERE=os.path.dirname(os.path.abspath(__file__))
D=json.load(open(os.path.join(HERE,'..','..','data','figS12_curves.json')))
NAMES={'north_america':'N America','europe':'Europe','east_asia':'E Asia','south_asia':'S Asia',
       'southeast_asia':'SE Asia','latin_america':'L America','sub_saharan_africa':'Sub-Saharan Africa','fsu_central_asia':'FSU/C Asia'}
order=['north_america','europe','east_asia','south_asia','southeast_asia','latin_america','sub_saharan_africa','fsu_central_asia']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':11})
fig,axes=plt.subplots(2,4,figsize=(16.5,8.2)); axes=axes.flatten()
for i,rk in enumerate(order):
    ax=axes[i]; d=D[rk]; ssa=(rk=='sub_saharan_africa')
    col='#C0392B' if ssa else '#3B6FB0'
    x=np.array(d['x']); y=np.array(d['y'])
    ax.plot(x,y,color=col,lw=2.2)
    ax.axhline(d['floor'],color='#888',ls=':',lw=1)
    ax.scatter([d['Ncur']],[d['fao']],s=55,c='black',zorder=5,label='FAOSTAT ~2020' if i==0 else None)
    ax.scatter([d['Nns']],[d['y_nosynth']],s=55,facecolors='none',edgecolors='#C0392B',linewidth=1.6,zorder=5)
    ax.set_title(NAMES[rk], fontweight='bold' if ssa else 'normal')
    ax.set_xlim(0,400); ax.set_ylim(0,7)
    ax.text(0.97,0.10,f'y_max={d["ym"]:.2f}',transform=ax.transAxes,ha='right',fontsize=9,color='#555')
    ax.text(0.97,0.03,f'floor={d["floor"]:.2f}',transform=ax.transAxes,ha='right',fontsize=9,color='#555')
    if i==0: ax.legend(loc='upper left',fontsize=9,frameon=False)
    if i%4==0: ax.set_ylabel('Yield (t ha$^{-1}$)')
    if i>=4: ax.set_xlabel('Total N availability (kg N ha$^{-1}$ yr$^{-1}$)')
fig.suptitle('Regional crop nitrogen-response calibration (simulated year-2 yield; c=0.015 uniform, y_max numerically calibrated to FAOSTAT, empirical floor)',fontsize=12.5,y=0.99)
plt.tight_layout(rect=[0,0,1,0.97])
fig.savefig(os.path.join(HERE,'..','..','figures','Figure_S12_crop_response_calibration.png'),dpi=200,bbox_inches='tight',facecolor='white')
print('saved')
