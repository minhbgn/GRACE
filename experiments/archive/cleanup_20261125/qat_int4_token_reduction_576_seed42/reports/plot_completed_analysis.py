"""Plot saved accuracy changes against actual token retention; no model imports."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'completed_analysis_20261116.json').read_text())
by = {(j['suite'], j['config']): j for j in data['jobs'] if j['status'] == 'complete'}
suites = ['hallusionbench', 'mmbench', 'scienceqa', 'ai2d', 'mmmu', 'seedbench_img',
          'mmstar', 'tempcompass', 'videomme']
styles = [('resolution', '#2166ac', 'o'), ('uniform', '#c45b18', 's'), ('merge', '#25845a', '^')]
fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
for ax, suite in zip(axes.flat, suites):
    ax.axhline(0, color='#777777', lw=.8, ls='--')
    for method, color, marker in styles:
        points = [(100., 0.)]
        for ratio in [50, 25, 10]:
            j = by[suite, f'{method}_r{ratio:03}']
            if not j.get('matched_input_policy'):
                continue
            points.append((j['actual_retention'] * 100, j['delta_pp']))
        points.sort()
        ax.plot(*zip(*points), color=color, marker=marker, label=method, lw=1.5, ms=5)
    ax.set_title(suite.replace('_', ' '), fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_xticks([10, 25, 50, 100])
    ax.grid(alpha=.18)
    ax.tick_params(labelsize=9)
    if suite == 'videomme':
        ax.text(.38, .16, 'Original resolution 25%/10%\nomitted: different starting policy',
                transform=ax.transAxes, fontsize=8, color='#555555')
fig.supxlabel('Actual retained visual tokens (% of matched r100 baseline)', fontsize=11)
fig.supylabel('Accuracy change from r100 (percentage points; higher is better)', fontsize=11)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
fig.savefig(HERE / 'accuracy_retention_20261116.png', dpi=180)
fig.savefig(HERE / 'accuracy_retention_20261116.pdf')
print(HERE / 'accuracy_retention_20261116.png')
