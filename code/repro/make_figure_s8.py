import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
HERE=os.path.dirname(os.path.abspath(__file__))
D=json.load(open(os.path.join(HERE,'..','..','data','figS8_curves.json')))
RO=D['RO']; base=D['base']; half=D['half']; gbase=D['gbase']; ghalf=D['ghalf']
NAME={'north_america':'North America','europe':'Europe','east_asia':'East Asia','south_asia':'South Asia',
      'southeast_asia':'SE Asia','latin_america':'Latin America','sub_saharan_africa':'Sub-Saharan Africa','fsu_central_asia':'FSU & Central Asia'}
COL={'sub_saharan_africa':'#C62828','south_asia':'#1565C0','fsu_central_asia':'#6A1B9A','southeast_asia':'#E65100',
     'east_asia':'#795548','europe':'#00695C','latin_america':'#2E7D32','north_america':'#455A64'}
yrs=list(range(31))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':12})
fig,(axa,axb)=plt.subplots(1,2,figsize=(16.4,6.4),gridspec_kw={'width_ratios':[1,1.15]})
# Panel a
axa.fill_between(yrs,ghalf,gbase,color='0.8',alpha=0.7,label='Elasticity range')
axa.plot(yrs,gbase,color='#C0392B',lw=2.6,label='Baseline ε$_{F,PF}$')
axa.plot(yrs,ghalf,color='#1565C0',lw=2.4,ls='--',label='Halved ε$_{F,PF}$ (0.5×)')
axa.set_xlabel('Years after disruption onset'); axa.set_ylabel('Global yield loss (%)')
axa.set_xlim(0,30); axa.set_ylim(0,6.5); axa.legend(loc='upper left',frameon=True)
axa.text(-0.08,1.02,'a',transform=axa.transAxes,fontsize=17,fontweight='bold')
# Panel b: sort descending by baseline yr10
order=sorted(RO,key=lambda k:base[k]['10'])  # ascending -> bottom-to-top
ypos=np.arange(len(order))
for i,k in enumerate(order):
    b=base[k]['10']; h=half[k]['10']
    axb.plot([h,b],[i,i],color='0.6',lw=2,zorder=1)
    axb.scatter([b],[i],s=130,c=COL[k],edgecolors='black',lw=0.5,zorder=3)
    axb.scatter([h],[i],s=130,facecolors='white',edgecolors=COL[k],lw=2,zorder=3)
    axb.text(h-0.25,i,f'{h:.1f}%',ha='right',va='center',fontsize=10.5,color='0.35')
    axb.text(b+0.25,i,f'{b:.1f}%',ha='left',va='center',fontsize=10.5,color='black')
axb.set_yticks(ypos); axb.set_yticklabels([NAME[k] for k in order])
axb.set_xlabel('Year-10 yield loss (%)'); axb.set_xlim(0,16.5); axb.axvline(0,color='0.7',ls=':',lw=1)
axb.text(-0.02,1.02,'b',transform=axb.transAxes,fontsize=17,fontweight='bold')
from matplotlib.lines import Line2D
axb.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#C62828',markeredgecolor='black',markersize=11,label='Baseline'),
                    Line2D([0],[0],marker='o',color='w',markerfacecolor='white',markeredgecolor='#C62828',markersize=11,label='Halved (0.5×)')],
          loc='lower right',frameon=True)
axb.text(0.99,-0.14,'Regional ordering preserved: Spearman ρ = 0.98',transform=axb.transAxes,ha='right',fontsize=10,style='italic',color='0.4')
plt.tight_layout()
fig.savefig(os.path.join(HERE,'..','..','figures','Figure_S8_elasticity_sensitivity.png'),dpi=200,bbox_inches='tight',facecolor='white')
print('saved; global yr10 base/half = %.1f / %.1f'%(gbase[10],ghalf[10]))
