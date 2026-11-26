from pathlib import Path
import json,hashlib
exp=Path('reproduction/GRACE/experiments/released_bf16_init_32k_b128_p100352_t2048_seed42')
source=exp/'reports/report_analysis.json'
scores=json.loads(source.read_text())['scores']
keys=['hallusionbench','mmbench','scienceqa','ai2d','mmmu','seedbench_img','mmstar','mean']
refs={
 'vanilla_2b':('Vanilla 2B, BF16',[51.4,78.4,81.4,76.9,53.4,71.2,58.3,67.3]),
 'vanilla_8b_teacher':('Vanilla 8B teacher, BF16',[61.1,84.5,85.0,85.7,69.6,77.5,70.9,76.3]),
 'released_bf16':('Released BF16 initialization',[66.9,86.4,86.2,81.3,72.1,76.7,67.3,76.7]),
 'released_grace_w4':('Released full GRACE W4',[65.4,84.6,84.3,79.5,70.5,75.1,65.8,75.0]),
 'released_grace_w8':('Released full GRACE W8',[66.1,85.5,85.3,80.4,71.3,75.9,66.5,75.9]),
}
rows={}
for name,(label,values) in refs.items():
 paper=dict(zip(keys,values))
 rows[name]={'label':label,'paper':paper,'local':{k:scores[name][k] for k in keys},'local_minus_paper_pp':{k:scores[name][k]-paper[k] for k in keys}}
header=['Reference model','HallB','MMBench','ScienceQA','AI2D','MMMU','SEED','MMStar','Mean','Δ mean (pp)']
lines=['| '+' | '.join(header)+' |','| --- | '+' | '.join(['---:']*9)+' |']
for row in rows.values():
 lines.append('| '+' | '.join([row['label'],*[f'{row["paper"][k]:.1f} → {row["local"][k]:.2f}' for k in keys],f'{row["local_minus_paper_pp"]["mean"]:+.2f}'])+' |')
deltas=['| Released checkpoint | HallB | MMBench | ScienceQA | AI2D | MMMU | SEED | MMStar | Δ mean |', '| --- | '+' | '.join(['---:']*8)+' |']
for name in ('released_grace_w4','released_grace_w8'):
 row=rows[name];deltas.append('| '+row['label']+' | '+' | '.join(f'{row["local_minus_paper_pp"][k]:+.2f}' for k in keys)+' |')
section='''### Comparison with the paper's claimed results

**Added 2026-09-25.** The reference is the Qwen3-VL comparison in **Table 6 of
arXiv v5**, also reproduced in the authors' repository and model cards. The
LLaVA and Qwen2-VL result tables concern different backbones. Each cell below
shows **paper → our original custom-protocol evaluation** for the same named
reference model; these are not the later lmms-eval scores.
[Paper, Table 6](https://arxiv.org/html/2601.22709v5#A1.T6),
[repository results](https://github.com/ForeverBlue816/GRACE#-results).

{{comparison}}

The paper's one-decimal values and reported `Avg.` are retained as published.
Local scores and differences are calculated from the unrounded
[report data](reports/report_analysis.json); the
[machine-readable comparison](reports/custom_protocol_paper_comparison.json)
records both sets of values. Because the paper rounds its cells and average
separately, the average of its displayed cells can differ slightly from its
reported `Avg.`. The mean-gap column subtracts the reported `Avg.`, not a
reconstructed mean of rounded cells.

For the low-bit releases, the per-benchmark differences are **local minus paper**:

{{deltas}}

**The seven-suite headline gap is −1.15 pp for W4 and −3.62 pp for W8, with
substantial cancellation between benchmarks.** MMMU is the largest negative
difference for both releases: −9.28 and −14.41 pp. MMBench moves in the opposite
direction, by +7.87 and +6.05 pp. SEED and HallusionBench also contribute to W8's
shortfall. Thus, the W4 mean being relatively close does not establish faithful
reproduction at the task level.

The discrepancy is not unique to the low-bit releases. The vanilla 8B teacher
is −18.27 pp on MMMU and −11.77 pp on MMStar, but +9.23 pp on HallusionBench and
+7.36 pp on ScienceQA. Such large, opposing changes in an independently released
baseline are consistent with evaluation sensitivity; they cannot be explained
solely by our constrained training or the authors' low-bit optimization.

The released W8 checkpoint also remains below released W4 on every local suite:
**72.28 versus 73.85**, whereas the paper reports W8 ahead by 0.9 pp. That
reversed ordering is an observation about these artifacts under our protocol,
not evidence that eight-bit training intrinsically performs worse.

#### Why this is not a matched reproduction of the paper

Our internal comparison holds the local evaluator fixed across all ten models.
Matching benchmark names does not establish that the authors used identical
splits, prompts, image budgets or scoring. In particular:

- This report uses **4,096–100,352 pixels**, eight output tokens for most tasks,
  and custom Transformers evaluators. The exact author settings for the
  published checkpoint scores have not been recovered.
- **ScienceQA** here combines 2,017 image and 2,224 text-only questions. The later
  harness round uses image questions only; the paper's exact evaluation scope
  must be confirmed before calling either configuration an exact match.
- **HallusionBench** here uses rule-based scoring over the local image/non-image
  collection. **MMBench** uses per-row English-dev accuracy. These differ from
  the image-only judged HallusionBench and rolling/judge-assisted MMBench tasks
  selected for the pinned harness workflow.
- **MMMU** uses the local mixed-answer scorer, including simplified open-answer
  matching. **MMStar** uses question-level accuracy, whereas the later pinned
  harness reports an equal mean over L2-category accuracies. Answer extraction
  can also respond differently to models' output formats.

Consequently, a local method numerically exceeding a published headline is not
a matched-method win. For example, LoRA-QAT W8's 76.15 is above the paper's W8
75.9, but the valid controlled contrast in this report is against the **same
release evaluated locally**, at 72.28. Likewise, GRACE-LoRA W8 exceeds that
local release by 1.65 pp while remaining below the paper's headline. The paper's
Qwen3 table has no matched QAT-only row; our QAT ablation supplies that control
for this constrained experiment.

#### What the subsequent checks established

The later [integrity audit](reports/evaluation_integrity_audit.md) verified all
ten checkpoint files against their evaluation manifests and the downloaded
references against official Hugging Face commits. It found no swapped paths or
corrupted reference downloads. For the **four completed lmms-eval suites**, it
also verified source datasets, shared prompts/settings and independent
explicit-choice rescoring. That latter scoring audit concerns the harness
round; it is not an independent validation of every custom scorer in this
original seven-suite report.

The [lmms-eval comparison](lmms_evaluation.md) now gives released W4/W8 four-suite
means of **77.68 / 76.72**. Relative to the paper's same four named columns,
these are +1.50 / −0.31 pp, but MMMU and the two judge-based suites remain
unavailable in that round. These are **four-suite means** and cannot replace,
be compared directly with, or be combined with the seven-suite means above.
The authentic W4/W8 ordering remains reversed in that evaluation as well.

The evidence therefore supports competitive local accuracy under explicitly
recorded protocols. The unresolved paper discrepancy calls for a successful
standard MMMU evaluation, the authors' exact commands/raw predictions, and
controlled changes to resolution, prompts and scoring. It does not justify
assuming that the downloaded models are wrong or that the paper's claimed
numbers have been reproduced.

'''
section=section.replace('{{comparison}}','\n'.join(lines)).replace('{{deltas}}','\n'.join(deltas))
p=exp/'experiment_report.md';text=p.read_text();anchor='### Distillation benefit and gap to the full method'
assert anchor in text and '### Comparison with the paper\'s claimed results' not in text
text=text.replace(anchor,section+anchor,1)
old='This unexpected W8 ordering warrants a focused loading/configuration and protocol audit before presenting “better than full GRACE” as a general claim.'
new='This unexpected W8 ordering motivated the subsequent [integrity audit](reports/evaluation_integrity_audit.md), which verified checkpoint identity and found no download/path error; the intended author evaluation protocol remains unresolved, so “better than full GRACE” is not established as a general claim.'
assert old in text;text=text.replace(old,new)
p.write_text(text)
record={'source':'https://arxiv.org/html/2601.22709v5#A1.T6','local_protocol':'Original seven-suite custom Transformers evaluation; not lmms-eval', 'local_source':'reports/report_analysis.json','local_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'mean_delta_definition':'unrounded local mean minus paper reported one-decimal Avg.','rows':rows}
(exp/'reports/custom_protocol_paper_comparison.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print('Added paper comparison for',len(rows),'reference models with all seven benchmark columns.')
