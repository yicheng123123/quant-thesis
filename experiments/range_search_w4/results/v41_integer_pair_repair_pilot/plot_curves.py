"""Standalone CPU matplotlib figure from locked numerical results."""
import os,csv
from pathlib import Path
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'plot_config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(n):
    with (OUT/n).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
rows=read('ROW_RESULTS.csv');budget=read('BUDGET_RESULTS.csv')
lookup={(r['unit_id'],r['init'],r['algorithm']):r for r in rows}
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
for ax,(init,split) in zip(axes.flat,[(i,s) for i in ['alpha1','scalar10'] for s in ['S0','S1']]):
    for alg,label,color in [('A','A: single','#333333'),('B','B: random pairs','#888888'),('C','C: correlation pairs','#225ea8')]:
        xx=[0];yy=[1.]
        for node in [16,32,64,96,128]:
            rr=[r for r in budget if r['init']==init and r['split']==split and r['algorithm']==alg and int(r['node_over_dim'])==node]
            denominator=sum(float(lookup[(r['unit_id'],init,alg)]['search_initial_mse']) for r in rr)
            xx.append(node);yy.append(sum(float(r['search_mse']) for r in rr)/denominator)
        ax.plot(xx,yy,marker='o',markersize=3,linewidth=1.6,label=label,color=color)
    ax.set_title(('Local alpha=1' if init=='alpha1' else 'Scalar10 search winner')+' / '+split)
    ax.set_xlabel('Candidate evaluation allowance / input dimension')
    ax.set_ylabel('Pooled search SSE / initial SSE (ratio)')
    ax.set_xticks([0,16,32,64,96,128]);ax.set_ylim(.8,1.01);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(frameon=False,fontsize=8)
fig.suptitle('Fixed-grid integer repair: historical cached search',fontsize=13)
fig.text(.05,.015,'Source: v24/v30 fixed M2-prefix inputs, 24 rows per split/init. No new articles. A holds state after stopping or its 32-update cap.\nActual evaluations and accepted-step losses: SEARCH_TRAJECTORIES.csv; common allowance states: BUDGET_RESULTS.csv.',fontsize=8)
fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(OUT/'SEARCH_BUDGET_CURVES.svg');fig.savefig(OUT/'SEARCH_BUDGET_CURVES.png',dpi=150);plt.close(fig)
