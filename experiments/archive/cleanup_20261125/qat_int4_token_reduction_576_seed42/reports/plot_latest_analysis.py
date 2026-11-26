"""Standalone figures for the completed matrix and the timing extrapolation."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/qat_token_reduction_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/'completed_analysis_20261125.json').read_text())
by = {(j['suite'],j['config']):j for j in data['jobs']}
suites = list(dict.fromkeys(j['suite'] for j in data['jobs']))
styles = [('resolution','#2166ac','o'),('uniform','#c45b18','s'),('merge','#25845a','^')]
fig, axes = plt.subplots(4,3,figsize=(12,12),layout='constrained')
for ax,suite in zip(axes.flat,suites):
    ax.axhline(0,color='#777777',lw=.8,ls='--')
    for method,color,marker in styles:
        points=[(100.,0.)]
        for ratio in (50,25,10):
            j=by[suite,f'{method}_r{ratio:03}']
            if j['matched_input_policy']:
                points.append((100*j['actual_retention'],j['delta_pp']))
        points.sort()
        ax.plot(*zip(*points),color=color,marker=marker,label=method,lw=1.5,ms=5)
    ax.set_title(suite.replace('_',' '),fontsize=10)
    ax.set_xlim(0,105); ax.set_xticks([10,25,50,100]); ax.grid(alpha=.18)
    if suite.startswith('videommmu'):
        ax.text(.98,.04,'Original-policy resolution r10 omitted',ha='right',
                transform=ax.transAxes,fontsize=7,color='#555555')
fig.supxlabel('Actual retained visual tokens (% of matched r100 baseline)')
fig.supylabel('Accuracy change from r100 (percentage points; higher is better)')
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
for suffix in ('png','pdf'):
    fig.savefig(HERE/f'accuracy_retention_20261125.{suffix}',dpi=180)
plt.close(fig)

timing=json.loads((HERE/'timing_analysis_20261125.json').read_text())
tb={e['config']:e for e in timing['rows']}
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,metric in zip(axes,('model','request')):
    for method,color,marker in styles:
        xx,yy=[],[]
        for ratio in (50,25,10):
            config=f'{method}_r{ratio:03}'
            e=tb[config][metric];x=e['full_seconds']/3600;y=by['videomme',config]['score_percent']
            lo,hi=[v/3600 for v in e['full_seconds_ci95']]
            ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],color=color,marker=marker,capsize=3,ls='none')
            offset=(4,-12) if method=='merge' else (4,7)
            if method=='merge' and ratio==10:
                offset=(4,5)
            if method=='uniform' and ratio==50:
                offset=(-22,6)
            ax.annotate(f'r{ratio}',(x,y),xytext=offset,textcoords='offset points',fontsize=8,color=color)
            xx.append(x);yy.append(y)
        ax.plot(xx,yy,color=color,alpha=.6,label=method)
    e=tb['baseline_r100'][metric];x=e['full_seconds']/3600
    y=by['videomme','baseline_r100']['score_percent'];lo,hi=[v/3600 for v in e['full_seconds_ci95']]
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],color='#444444',marker='*',ms=10,capsize=3,label='r100')
    ax.set_xlabel(f'Projected {metric} hours for 1,800 questions (lower is better)')
    ax.set_ylabel('Measured full-population accuracy (%)');ax.grid(alpha=.18)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
for suffix in ('png','pdf'):
    fig.savefig(HERE/f'videomme_tradeoff_20261125.{suffix}',dpi=180)
print('Wrote accuracy-retention and VideoMME runtime-tradeoff PNG/PDF figures.')
