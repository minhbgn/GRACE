"""Render the audited score comparison and report; never edits evaluation outputs."""
import json
from pathlib import Path
import shutil

EXP=Path(__file__).resolve().parents[1]
A=json.loads((EXP/'reports/comparison_audit.json').read_text())
T=json.loads((EXP/'reports/training_token_audit.json').read_text())
S=['hallusionbench','mmbench','scienceqa','ai2d','mmmu','seedbench_img','mmstar']
N=['Hallusion','MMBench','ScienceQA','AI2D','MMMU','SEED image','MMStar']
M=['qat_w4','grace_w4','qat_w8','grace_w8']
LABEL={'qat_w4':'QAT W4','grace_w4':'GRACE W4','qat_w8':'QAT W8','grace_w8':'GRACE W8'}

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def f(x):return f'{x:.2f}'
def signed(x):return f'{x:+.2f}'
def ev(label,method):return A['evaluations'][label+'/'+method]
def comp(ll,l,rr,r):return A['comparisons'][ll+'/'+l+' minus '+rr+'/'+r]

# Shareable figure: raw audited scores, equal benchmark weights.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':10})
fig,(ax,heat)=plt.subplots(1,2,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1.45]})
x=np.arange(4);old=[ev('baseline',m)['mean'] for m in M];new=[ev('new',m)['mean'] for m in M]
ax.bar(x-.18,old,.36,label='Old: up to 98 visual tokens',color='#a9bbcf')
ax.bar(x+.18,new,.36,label='New: up to 576 visual tokens',color='#245b88')
for i,(a,b) in enumerate(zip(old,new)):
 ax.text(i-.18,a+.08,f(a),ha='center',fontsize=8);ax.text(i+.18,b+.08,f(b),ha='center',fontsize=8)
ax.axhline(ev('baseline','released_bf16')['mean'],color='#b45f06',ls='--',lw=1,label='Released BF16 reference (native budget differs)')
ax.set_xticks(x,[LABEL[m] for m in M],rotation=20);ax.set_ylim(0,86);ax.set_ylabel('Equal-weight seven-suite mean (%)');ax.set_title('Larger-budget controls improve all four means');ax.legend(loc='lower left',fontsize=8)
values=np.array([list(comp('new',m,'baseline',m)['deltas'].values()) for m in M])
im=heat.imshow(values,cmap='RdBu',vmin=-5,vmax=5,aspect='auto')
heat.set_xticks(range(7),N,rotation=35,ha='right');heat.set_yticks(range(4),[LABEL[m] for m in M]);heat.set_title('New minus old scores (percentage points)')
for i in range(4):
 for j in range(7):heat.text(j,i,signed(values[i,j]),ha='center',va='center',color='white' if abs(values[i,j])>3 else 'black',fontsize=9)
fig.colorbar(im,ax=heat,label='Percentage points',shrink=.8)
fig.suptitle('QAT and GRACE: higher visual budget, same 256-update adaptation',fontsize=13)
fig.tight_layout();fig.savefig(EXP/'reports/score_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(EXP/'reports/score_comparison.pdf',bbox_inches='tight');plt.close(fig)

parts=['# Larger visual budget QAT and GRACE results',
'''Prepared **2026-10-12**. The four production runs completed October 5–6 Singapore time, followed by full local evaluation October 11–12. This compares the new 589,824-pixel / 4,096-token experiment with the earlier 100,352-pixel / 2,048-token checkpoints in the [evaluation-only baseline](../reproduction_eval_only/experiment_report.md), using **`local_native_greedy256_v1`** throughout. The directory retains its original `t2048` name; the executed sequence limit was **4,096**.

**The larger-budget setup improves all four seven-suite means by 1.52–2.01 percentage points, but CE-only QAT still outperforms GRACE at both precisions.** QAT W8 reaches **77.37%**, versus **77.40%** for the released BF16 SFT reference under its different native processor. GRACE reaches **74.56% at W4** and **74.86% at W8**, trailing matched QAT by **1.22 and 2.51 pp**. Higher resolution improves absolute scores without resolving the distillation deficit.

All **28/28 new evaluation pairs** have valid completion evidence. The nine historical MMBench result files also pass the current CPU validator; their archived completion markers remain absent. Baseline MMBench scores and means below therefore include validated saved predictions, rather than treating missing bookkeeping as missing accuracy.

![Seven-suite means and per-task improvements](reports/score_comparison.png)

## Seven suite comparison''']
rows=[]
for m in M:rows.append([LABEL[m],f(ev('baseline',m)['mean']),f(ev('new',m)['mean']),signed(comp('new',m,'baseline',m)['mean_delta']),f(ev('new',m)['mean']-ev('baseline','released_bf16')['mean'])])
parts.append(table(['Method','Old mean %','New mean %','New minus old pp','New minus released BF16 pp'],rows))
parts.append('All means weight the seven benchmark metrics equally; they are not pooled accuracy or the LLMC+ aggregate. Scores and deltas are calculated at full precision before rounding. “Old” means the evaluation-only baseline’s latest local protocol, not the older training report’s evaluation recipe.')
rows=[]
for m in M:
 for label,title in [('baseline','Old'),('new','New')]:
  v=ev(label,m);rows.append([title+' '+LABEL[m],*[f(v['suites'][s]['score_percent']) for s in S],f(v['mean'])])
parts.append(table(['Checkpoint',*N,'Mean'],rows))
parts.append('### Per task change from the previous controls')
parts.append(table(['Method',*N],[[LABEL[m],*[signed(comp('new',m,'baseline',m)['deltas'][s]) for s in S]] for m in M]))
parts.append('''**27 of 28 method–task comparisons improve.** The sole decline is GRACE W8 on HallusionBench, **−0.09 pp**, equivalent to one fewer correct answer. The most consistent gains are on SEED image (**+2.99 to +3.94 pp**) and MMStar (**+3.33 to +4.40 pp**). These gains are consistent with additional visual detail helping, but this experiment does not isolate the mechanism: training inputs, evaluation image budgets and the total training length limit all changed.

### GRACE versus matched QAT''')
parts.append(table(['GRACE minus QAT',*N,'Mean'],[[f'W{b}',*[signed(comp('new',f'grace_w{b}','new',f'qat_w{b}')['deltas'][s]) for s in S],signed(comp('new',f'grace_w{b}','new',f'qat_w{b}')['mean_delta'])] for b in (4,8)]))
parts.append('''GRACE wins on **HallusionBench, ScienceQA and SEED**, and loses on **MMBench, AI2D, MMMU and MMStar**, at both precisions. Its W4 mean deficit changes from **−1.16 to −1.22 pp**; its W8 deficit widens from **−2.20 to −2.51 pp**. Thus there is no observed aggregate recovery of the distillation benefit at the larger visual budget.

MMStar accounts for most of the average deficit. Excluding it as a diagnostic only, GRACE still trails QAT by **0.10 pp at W4** and **0.73 pp at W8** across the remaining six suites. The seven-suite mean remains the reported outcome; excluding a difficult task would conceal a material regression.

Moving W4 → W8 improves QAT’s mean by **1.59 pp**, but GRACE’s by only **0.30 pp**. GRACE W8 loses **3.27 pp on MMStar** relative to GRACE W4. More weight precision therefore does not solve the observed GRACE behavior.

## Comparison with frozen reference checkpoints''')
refs=['vanilla_2b','vanilla_8b_teacher','released_bf16','released_grace_w4','released_grace_w8']
refnames=['Vanilla 2B','Vanilla 8B teacher','Released BF16 SFT','Released GRACE W4','Released GRACE W8']
parts.append(table(['Frozen baseline reference',*N,'Mean'],[[name,*[f(ev('baseline',m)['suites'][s]['score_percent']) for s in S],f(ev('baseline',m)['mean'])] for m,name in zip(refs,refnames)]))
parts.append('''These references are reused from the evaluation-only baseline; they were not rerun in the new experiment. New GRACE exceeds the matching released GRACE mean by **+0.51 pp at W4** and **+3.74 pp at W8**, but still underperforms the stronger matched local QAT controls.

QAT W8 is **0.03 pp below released BF16 on the mean**, with task-specific gains and losses. It improves AI2D (+0.71), MMMU (+0.44) and MMStar (+1.73), while falling on HallusionBench (−1.77), MMBench (−0.52), ScienceQA (−0.35) and SEED (−0.45). This is near-equality of the observed aggregate, not demonstrated statistical or functional equivalence.

Native image budgets differ: released BF16 and vanilla checkpoints use minimum/maximum **65,536 / 16,777,216** pixels; released GRACE W4 uses **784 / 200,704**; released GRACE W8 **784 / 50,176**; new local QAT/GRACE **4,096 / 589,824**. Thus reference comparisons describe checkpoint-plus-processor behavior under the agreed protocol, not an isolated precision effect. Vanilla models also have more unresolved and capped outputs in this protocol.

## MMStar remains the main GRACE weakness

Each category has 250 questions. The following uses the new checkpoints:''')
categories=list(A['mmstar_categories']['new/qat_w4'])
parts.append(table(['Category','QAT W4','GRACE W4','GRACE minus QAT','QAT W8','GRACE W8','GRACE minus QAT'],[[cat,f(A['mmstar_categories']['new/qat_w4'][cat]['accuracy']),f(A['mmstar_categories']['new/grace_w4'][cat]['accuracy']),signed(A['mmstar_categories']['new/grace_w4'][cat]['accuracy']-A['mmstar_categories']['new/qat_w4'][cat]['accuracy']),f(A['mmstar_categories']['new/qat_w8'][cat]['accuracy']),f(A['mmstar_categories']['new/grace_w8'][cat]['accuracy']),signed(A['mmstar_categories']['new/grace_w8'][cat]['accuracy']-A['mmstar_categories']['new/qat_w8'][cat]['accuracy'])] for cat in categories]))
parts.append('''GRACE remains below QAT in **all six categories** at both bit widths. Math has the largest losses: **−19.20 pp at W4** and **−25.20 pp at W8**. At W8, logical reasoning is also **−15.60 pp** and science/technology **−12.80 pp**. All 1,500 new MMStar responses per checkpoint are bare A/B/C/D choices with zero unresolved answers, inference errors or length stops; the regression is in selected answers, not parser failures or the generation cap.''')
parts.append(table(['Paired MMStar comparison','Only left correct','Only right correct','Net correct'],[[f'New GRACE W{b} minus new QAT W{b}',*(lambda p:[p['only_left_correct'],p['only_right_correct'],p['only_left_correct']-p['only_right_correct']])(comp('new',f'grace_w{b}','new',f'qat_w{b}')['mmstar_paired'])] for b in (4,8)]))
parts.append('''The larger budget nevertheless raises GRACE’s MMStar score by **4.40 pp at W4** and **3.33 pp at W8** versus its old checkpoint. That is an absolute improvement, while the W8 deficit against QAT grows because QAT improves more.

### HallusionBench gains also appear in grouped consistency

The primary HallusionBench metric remains row accuracy (aAcc) over image and text rows. Auxiliary qAcc and fAcc require all answers in a question group or image-figure group to be correct:''')
parts.append(table(['New checkpoint','aAcc %','qAcc %','fAcc %'],[[LABEL[m],*[f(A['hallusion_grouped']['new/'+m][k]) for k in ('aAcc','qAcc','fAcc')]] for m in M]))
parts.append('''GRACE improves both grouped metrics relative to QAT at each precision. This supports retaining the per-task evidence even though GRACE loses on the overall mean. Group definitions match the existing local audit; these are auxiliary measurements, not substitutes for the primary seven-suite score.

## Executed configuration and data comparability

Both branches start from the authors’ released **BF16 SFT checkpoint without GDKD/RCKA**. QAT uses CE-only LoRA-QAT; GRACE adds GDKD, RCKA and the adaptive controller. The saved launch recipes confirm:''')
parts.append(table(['Setting','Old controls','New controls'],[
 ['Maximum image area / merged visual ceiling','100,352 / 98','589,824 / 576'],['Minimum image area','4,096','4,096'],['Total training sequence cap','2,048','4,096'],['Training subset / seed / sampling seed','32,768 / 42 / 2026','Same'],['GPUs / microbatch / accumulation / effective batch','2 / 1 / 64 / 128','Same'],['Optimizer updates / RCKA warmup','256 / 25','Same'],['LoRA rank / alpha / dropout','32 / 64 / 0.05','Same'],['LoRA / merger / scale LR','2e-4 / 1e-5 / 2e-4','Same'],['W4/W8 group size','128','Same'],['Training implementation hashes and environment','Pinned','Identical to old controls']]))
parts.append('''The saved resolved configs differ only in `max_pixels` and `model_max_length`; all training implementation hashes match. The old evaluation-only report’s opening table still lists first-pilot values (batch 64, 512 updates, 50,176 pixels and 1,024 tokens). Those are not the baseline checkpoints compared here; their executed recipes confirm batch 128, 256 updates, 100,352 pixels and 2,048 tokens.

The new and old subsets share **32,767 of 32,768 identities**. One example was added and one removed, **no common conversation text changed**, and **32,512 common records changed token hashes**. There are **32,669 identical positions**, so **99 sequence positions differ**, including the replacement and reordered common examples. Within the new experiment, all four arms use the exact same manifest; across old/new experiments, data order is almost but not completely matched.

The added example (annotation `69378`) has **2,209 total tokens** and is the only new example above 2,048. Its 2,069 nonvisual tokens explain why the longer cap admits it. The removed example (`68543`) had 328 old total tokens. Full-subset token measurements are:''')
oldtok=json.loads((EXP.parent/'released_bf16_init_32k_b128_p100352_t2048_seed42/reports/token_lengths.json').read_text())
parts.append(table(['Token measure','Old mean','New mean','Old median','New median','Old max','New max'],[[label,f(oldtok[k]['mean']),f(T['statistics'][k]['mean']),f(oldtok[k]['median']),f(T['statistics'][k]['median']),oldtok[k]['maximum'],T['statistics'][k]['max']] for label,k in [('Merged visual','visual'),('Nonvisual including formatting','nonvisual'),('Total input','total')]]))
parts.append('''Mean visual count increases **2.82×**, while mean nonvisual count stays almost unchanged. The new visual 95th percentile is **540**; **143/32,768 examples (0.44%)** reach 576. **2,728 (8.33%)** have at least 400 visual tokens. The maximum total sequence is **2,209**, with **zero examples reaching 4,096**. Consequently, the 4,096 limit is headroom rather than typical compute usage. The distribution also explains why production memory is lower than the synthetic 4,096-token stress test.

No token reduction is implemented here. This establishes useful unpruned controls for later LLMC+ integration, but 576 is a ceiling, not a fixed baseline count. Retention fractions and actual retained-count distributions will be more informative than assuming every image starts with 576 tokens.

## Training cost and diagnostics''')
rows=[]
for m in M:
 n=A['training']['new/'+m];o=A['training']['baseline/'+m]
 rows.append([LABEL[m],256,f(o['wall_hours']),f(n['wall_hours']),f(n['gpu_hours']),f(n['summary']['peak_reserved_gib']),f(o['summary']['peak_reserved_gib']),signed(100*(n['wall_hours']/o['wall_hours']-1))+'%'])
parts.append(table(['Method','Updates','Old elapsed h','New elapsed h','New GPU h','New reserved GiB/GPU','Old reserved GiB/GPU','Elapsed change'],rows))
parts.append('''All four runs completed in one successful invocation, without a pause/resume or memory-stop reason. Combined measured launch time is **8.94 elapsed GPU-job hours / 17.87 GPU-hours** across the four sequential two-GPU runs. Per-run launch timing includes loading, training, checkpointing and export; it excludes prior CPU preparation, memory probes, evaluation and the authors’ SFT stage. These are observed wall times on a shared machine, not controlled throughput benchmarks.

GRACE costs **27.8% more elapsed time at W4** and **27.3% more at W8** than QAT, and uses about **2.84×** its peak reserved memory. Relative to the smaller-input controls, new QAT takes **1.2–2.0%** longer and GRACE about **5.0%** longer. The larger visual budget thus improves measured quality at modest additional adaptation cost, while distillation remains substantially more expensive than matched CE-only QAT.

The adaptive beta reaches its upper bound at **update 4 for W4** and **update 5 for W8**, remaining capped for **253/256 (98.83%)** and **252/256 (98.44%)** recorded updates respectively. This supports investigating whether the current KD strength is well matched to the adaptation budget; it does not prove that saturation causes the benchmark regression. Mean logged training CE is **0.606/0.580 for QAT W4/W8** versus **0.954/0.925 for GRACE**. These are training observations, not held-out perplexity, and scalar loss values alone cannot establish gradient conflict. Compare per-term gradients or use a focused loss ablation before making a causal claim.

## Evaluation integrity and limitations

A CPU audit revalidated **91/91 saved pairs**, covering **382,460 prediction rows** across 13 checkpoint evaluations. It checked source identities and labels, coverage, duplicate IDs, native processor settings, decoding/token telemetry, deterministic rescoring, summary arithmetic and completion/provenance hashes where present. All four new exports passed the full-training export check at 256 steps and match their pinned recipe, subset and configured model paths. The audit checked saved checkpoint provenance and export structure; it did not rehash every current checkpoint weight byte.

The new policy and baseline have identical dataset fingerprints, metrics, decoding, attention backend (`flash_attention_2`) and software environment. Their policy differences are the experiment configuration and the evaluation runner hash: the current runner normalizes JSON histogram keys for MMBench validation. Scorer and worker source hashes are unchanged. New saved evaluator hashes match current sources; the baseline’s runner hash is historical. Original predictions, scores, frozen protocols and completion records are unchanged by this report.

New inference has **zero recorded errors**. QAT W8 has **two unresolved, length-capped ScienceQA responses**; all other new outputs resolve without length stops. These same two outputs are counted in both diagnostics. For the old QAT W8, the corresponding totals were six unresolved and five capped. Most observed score changes therefore concern chosen answers, not recovered output formatting. The baseline’s nine MMBench files remain without completion markers, despite passing this audit; the new matrix is formally complete.

MMBench uses **circular accuracy over 1,164 base questions**, requiring all 4,329 supplied rotations to be correct in their groups. It is not the higher row-level accuracy. HallusionBench includes 1,129 image and text rows; ScienceQA all 4,241 test rows; AI2D 3,088 test questions; MMMU 900 validation questions; SEED 14,233 image questions; MMStar 1,500 questions. No paid semantic judging was used.

These are single-training-seed results. Paired answer counts show which predictions changed but do not measure training-seed uncertainty. No confidence interval or equivalence claim is inferred from the near-equal BF16/QAT means. Both training and evaluation image budgets changed, the training cap increased, and a small amount of subset/order changed; therefore higher-resolution inference alone, adaptation to richer inputs and extra text allowance are not isolated effects. The known native `Q(W)+Δ` versus merged/baked `Q(W+Δ)` export issue remains a separate validation question. BF16-stored simulated low-bit exports establish neither packed INT4/INT8 memory savings nor low-bit kernel acceleration.

## Implications for the next experiment

1. **Use new QAT W8 as the strongest measured local quality control**, and QAT W4 for the initial constrained token-reduction study. Retain GRACE as an experimental arm because its task-specific benefits remain relevant, but do not describe distillation as an aggregate improvement.
2. **Keep this completed experiment frozen.** It supplies the larger-budget unpruned controls. Continue the approved token-only LLMC+ reproduction separately, then measure pruning on these checkpoints at matched input budgets and actual retained fractions.
3. **Make the next diagnostic narrow.** To separate resolution from training effects, evaluate old and new checkpoints at a deliberately shared processor budget in a separately named diagnostic. To explain GRACE’s deficit, prioritize MMStar math/logic disagreement sets, native/export correspondence and a matched loss-strength/component check rather than another simultaneous hyperparameter sweep.
4. **Measure latency and token distributions alongside quality.** The new full-subset mean is about 255 visual tokens, so a pruning policy calibrated around a universal 576-token input would misstate actual reduction. Vision-tower and selector overhead still need measurement before any speed claim.

## Supervisor summary

> Increasing the visual ceiling from 98 to 576 tokens and the total training cap from 2,048 to 4,096 improved all four constrained QAT/GRACE checkpoints by 1.52–2.01 percentage points on the same local seven-benchmark protocol. QAT W8 reached 77.37%, close to the released BF16 reference’s 77.40% under different native image budgets. GRACE still trailed matched QAT by 1.22 pp at W4 and 2.51 pp at W8, driven mainly by MMStar deficits, while improving HallusionBench, ScienceQA and SEED. Each QAT run took about 1.96 hours on two A100s; GRACE about 2.50 hours. The new setup is a completed unpruned baseline for token reduction, not evidence yet that pruning or distillation improves the overall trade-off.

## Artifacts

- [New local score matrix](local_development/local_native_greedy256_v1/full/local_scores.md) and [baseline report](../reproduction_eval_only/experiment_report.md).
- [Comparison audit JSON](reports/comparison_audit.json): raw-precision scores, result paths and hashes, completion checks, protocol differences, paired/category diagnostics, cost and subset comparisons.
- [Score CSV](reports/scores_comparison.csv), [PNG figure](reports/score_comparison.png) and [PDF figure](reports/score_comparison.pdf).
- [Full training token audit](reports/training_token_audit.json) and [saved subset comparison](subset_comparison.json).
- Reproduction scripts: [compile comparison](reports/compile_comparison.py), [audit tokens](reports/audit_training_tokens.py), [render report](reports/render_comparison.py). Use `.venv-lmms-eval/bin/python` for compilation, `/home/air/miniconda3/envs/grace/bin/python` for the processor audit, and `/home/air/miniconda3/envs/fm/bin/python` for rendering (Matplotlib is already installed there). These are CPU reporting operations and do not launch inference.
''')
report=EXP/'experiment_report.md'
if report.exists() and 'report scaffold' in report.read_text():
 shutil.copy2(report,EXP/'reports/pre_completion_report_20261005.md')
report.write_text('\n\n'.join(parts)+'\n')
print(report)
