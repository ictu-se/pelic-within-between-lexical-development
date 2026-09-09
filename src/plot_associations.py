from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]/'results'
c=pd.read_csv(OUT/'within_between_contrasts.csv')
labels=['MATTR','MTLD','Density','Log frequency','Rare-word rate']
fig,ax=plt.subplots(figsize=(6.5,2.9));plt.rcParams['pdf.fonttype']=42
for j,r in c.iterrows():ax.errorbar(r.difference,4-j,xerr=1.96*r.se,fmt='o',color='#24556d',capsize=3)
ax.set_yticks(range(5),labels[::-1]);ax.axvline(0,color='gray',linestyle='--',linewidth=.8)
ax.set_xlabel('Between minus within association (outcome SD per level)');ax.spines[['top','right']].set_visible(False);fig.tight_layout()
(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/Figure_2_contrasts.png',dpi=300);fig.savefig(OUT/'figures/Figure_2_contrasts.pdf');plt.close(fig)

# Recreate the companion primary-estimate plot from the recorded aggregate fits.
d=pd.read_csv(OUT/'hybrid_model_results.csv');d=d[d.analysis=='primary']
plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(6.5,2.9))
metrics=['mattr50','mtld','lexical_density','mean_log_frequency_content','rare_content_rate']
for term,offset,color,marker,label in [('level_within',.12,'#24556d','o','Within learner'),('level_between',-.12,'#788792','s','Between learners')]:
 for i,m in enumerate(metrics):
  r=d[(d.metric==m)&(d.term==term)].iloc[0]
  ax.errorbar(r.estimate_std,4-i+offset,xerr=1.96*r.se,fmt=marker,color=color,capsize=3,label=label if i==0 else None)
ax.set_yticks(range(5),labels[::-1]);ax.axvline(0,color='gray',linestyle='--',linewidth=.8)
ax.set_xlabel('Association (outcome SD per course level)');ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,fontsize=9,loc='upper left');fig.tight_layout()
fig.savefig(OUT/'figures/Figure_1_estimates.png',dpi=300);fig.savefig(OUT/'figures/Figure_1_estimates.pdf');plt.close(fig)
