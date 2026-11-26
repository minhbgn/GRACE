from pathlib import Path
root=Path('reproduction/GRACE')
p=root/'evaluation/docs/README_video_evaluation.md'
s=p.read_text()
s=s.replace('## Selected input policy','## Default input policy — all suites')
s=s.replace('- Retain source spatial resolution, rounded to the nearest 32-pixel patch grid\n  (minimum 32 per dimension). This small grid rounding is recorded explicitly.',
'''- Use a **256 px maximum longest side for every suite, including TempCompass**.
  Preserve aspect ratio before rounding to the 32-pixel patch grid (minimum 32
  per dimension); smaller source frames are not deliberately enlarged. This is
  spatial downsampling, with no temporal clipping or fixed per-video token cap.''')
s=s.replace('`--longest-side N` allows an explicitly separate spatial-resolution experiment.',
'''The default is `--longest-side 256`. Higher-resolution experiments must pass an
explicit value, such as `--longest-side 320`; `--longest-side 0` explicitly selects
source resolution. The same selected value applies to every suite in the run.''')
a=s.index('## Commands\n')
b=s.index('`--methods` selects',a)
s=s[:a]+'''## Commands

From the workspace root, this **single full-run command** evaluates all ten models
on TempCompass, Video-MME and all three Video-MMMU sets at **2 fps / 256 px**:

```bash
bash reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh run --gpu 2
```

GPU 2 is an example: choose an available PCI GPU index with `gpuwho`.
Omitting `--suites` and `--methods` selects all five available tasks and all ten
checkpoints (50 model/task pairs), sequentially on one GPU. There is no separate
TempCompass configuration. Generation remains greedy with 4,096 output tokens
and each checkpoint's native 262,144-token context.

Optional planning, CPU preparation, a one-question-per-task smoke run and summary
use the same defaults and output base:

```bash
RUN=reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh
bash "$RUN" plan
bash "$RUN" check
bash "$RUN" prepare
bash "$RUN" run --smoke 1 --gpu 2
bash "$RUN" summarize
```

Later experiments select resolution explicitly; the policy hash automatically
separates their outputs:

```bash
bash "$RUN" run --longest-side 320 --gpu 2
# Explicit source-resolution input for a historical-policy comparison:
bash "$RUN" prepare --longest-side 0
```

Full-population projections from the saved source-size/frame-count audits show
**zero context overflows across all 5,180 questions at 256 px**, including the
4,096-token output reserve. The largest proposed input is 152,914 tokens. These
are CPU projections, not a fresh full decode or GPU-memory guarantee. See
[resolution feasibility](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/video_resolution_feasibility_20261017.json).
The longest vanilla-8B input still needs memory profiling before committing to
the full matrix; the runner records OOM failures without reducing inputs.

Video-MME's 900 videos / 2,700 questions are complete and extracted; the runner
detects them automatically. Prior source-resolution overflow counts are
historical evidence, not blockers attributed to the new 256 px default.

'''+s[b:]
s=s.replace('## Validation on 2026-10-17','## Historical source-resolution validation on 2026-10-17')
s=s.replace('The GPU/preflight evidence below was recorded with v1 before the directory\nreorganization.',
'''The evidence below used the earlier **2 fps / source-resolution** policy,
not the new unified 256 px default. GPU smoke/preflight evidence was recorded
with v1 before the directory reorganization.''')
s=s.replace('The context is 262,144 tokens, reserving 4,096 for output. Video-MMMU cannot be\nfully evaluated under this policy on these checkpoints. Its overflow questions\nmust remain blocked unless a separate lower-resolution/rate policy is approved.',
'''The context is 262,144 tokens, reserving 4,096 for output. These overflow counts
apply to the archived source-resolution policy. The user subsequently selected
256 px uniformly for all suites; its CPU projections fit context.''')
s=s.replace('The earlier Video-MME audit lacked **253 of 900** referenced videos; the completed\ndownload described above supersedes that missing-data finding.',
'''The earlier Video-MME audit lacked **253 of 900** referenced videos. After the
download completed, the source-resolution audit found 1,623/2,700 overflows and
zero first-frame/metadata probe errors; see the
[saved audit](../../experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/videomme_input_audit_20261017.json).
All 900 extracted videos are now present.''')
p.write_text(s)
p=root/'evaluation/docs/README.md';s=p.read_text();s=s.replace('Video-MME now has all 900 videos extracted. Its explicit commands use\n`--suites videomme`; see the [video guide](README_video_evaluation.md#commands)\nfor the completed-data output base and context-preflight requirements.',
'''All five video tasks now share **2 fps / 256 px longest side** by default,
including TempCompass. A single `run --gpu N` command selects all ten models and
all available tasks; see the [video guide](README_video_evaluation.md#commands).
Higher resolution is selected explicitly with `--longest-side N`.''');p.write_text(s)
p=root/'evaluation/docs/README_datasets.md';s=p.read_text();s=s.replace('audits 2 fps/source-resolution inputs without adaptive processor ceilings.',
'''defaults to the same 2 fps / 256 px longest-side policy for all suites, without
adaptive total-video token ceilings.''');s=s.replace('2 fps/source resolution. See the [updated run commands](README_video_evaluation.md#commands).\nThe full CPU audit found zero first-frame/metadata probe errors, but 1,623 of\n2,700 proposed inputs exceed model context under that policy. Download completion\ndoes not remove this context blocker.',
'''the selected resolution. See the [updated run commands](README_video_evaluation.md#commands).
The historical source-resolution audit found zero first-frame/metadata probe
errors but 1,623 context overflows. The user has since selected **256 px for all
suites**; projections at that default show zero context overflows.''');p.write_text(s)
p=root/'experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/README.md';s=p.read_text();a=s.index('## Video evaluation (inference only)');b=s.index('## Supervisor report checklist',a)
s=s[:a]+'''## Video evaluation (inference only)

`run_video_evaluation.sh` now uses one **2 fps / 256 px longest-side** default for
all ten models and all five available video tasks, including TempCompass. It
preserves full-video coverage, has no maximum frame count or fixed per-video
visual-token cap, and disables tokenizer truncation. From the workspace root:

```bash
bash reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/run_video_evaluation.sh run --gpu 2
```

Select an available PCI GPU index; 2 is only an example. Later resolution studies
pass `--longest-side 320` (or another explicit value); `--longest-side 0` requests
source resolution. Policy hashes separate output directories automatically.

Video-MME has all 900 extracted videos / 2,700 questions. Across all 5,180
questions, 256 px CPU projections show zero context overflows with a 4,096-token
output reserve; maximum proposed input is 152,914 tokens. Worst-case vanilla-8B
GPU memory still needs profiling. Earlier source-resolution audits and their
blocked results remain historical. See [the video evaluation guide](../../evaluation/docs/README_video_evaluation.md)
for commands, token accounting, scoring and preserved evidence.

'''+s[b:];p.write_text(s)
