# Video benchmark startup and memory investigation — 2026-10-23

The apparent startup failures have an established termination mechanism:
**all 15 `grace_w4`, `qat_w8`, and `grace_w8` jobs in the October 19 parallel
retry were killed with SIGKILL**. They did not report a Python loading or
quantization exception. All four local QAT/GRACE checkpoints successfully load,
transfer to CUDA, and generate in new isolated checks using the same package
versions and video-input implementation.

The separate `qat_w4` VideoMME failure is a CUDA allocation failure during a very
long language-model prefill. These evaluations use dense BF16 inference for
every checkpoint, including the models labelled W4 and W8.

**Implementation follow-up:** the launch scripts now select a separate resilient
v3 runner with the verified allocator configuration, per-GPU locks and occupancy
checks, durable validated-prefix resume, parent-written signal/exit diagnostics,
and one bounded SIGKILL retry. The v2 evaluator source files and its saved
results remain intact. See the [updated run guide](../../../evaluation/docs/README_video_evaluation.md)
and `resilient_runtime_validation_20261023/` in this reports directory for the
real-worker recovery checks. The investigation below describes the original
failure evidence and diagnostic experiments; it does not claim that software
can prevent an external SIGKILL.

## What the historical run actually completed

Evidence comes from
`video_evaluation/parallel_retry_20261019T034141Z_kbsu66/`, relative to this
experiment. The group's `full/<method>` paths are symlinks to the shared results;
searches must follow those symlinks. Earlier October 19 analysis reports predate
the later retry outcomes and should not be treated as current status.

| Local model | TempCompass | VideoMME | MMMU Perception | MMMU Comprehension | MMMU Adaptation |
| --- | --- | --- | --- | --- | --- |
| QAT W4 | Complete: 1,580 | CUDA OOM after 1,851 / 2,700 | Complete: 300 | Complete: 300 | SIGKILL after 160 / 300 |
| GRACE W4 | SIGKILL, 0 predictions | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 |
| QAT W8 | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 |
| GRACE W8 | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 | SIGKILL, 0 |

The three QAT W4 completion markers were revalidated against predictions,
scoring, population identities, and saved hashes. Vanilla 2B has now completed
VideoMME with all 2,700 questions; its completion marker was also revalidated.

## SIGKILL explains the silent startup logs

`gpu4/worker_exits.log` captures `wait4` and `SIGCHLD` from the matrix parent.
It explicitly records `WIFSIGNALED` and `WTERMSIG == SIGKILL` for each of the
15 zero-prediction jobs. Examples, in Singapore time:

| Job | Worker PID | Termination on October 20 |
| --- | ---: | --- |
| QAT W4 Adaptation, already at 160 predictions | 3953713 | 17:06:57, SIGKILL |
| GRACE W4 TempCompass | 4128284 | 17:07:27, SIGKILL |
| QAT W8 TempCompass | 4138181 | 17:14:19, SIGKILL |
| GRACE W8 TempCompass | 4148265 | 17:22:35, SIGKILL |
| GRACE W8 Adaptation | 4156058 | 17:30:20, SIGKILL |

This sequence began with an already-running QAT job and then affected every
subsequent local job. It is not specific to a model bit width or benchmark.

SIGKILL cannot run the worker's exception handler. That explains why logs end
at the weight-loading progress bar, with neither traceback nor `failure.json`.
That progress bar alone does not locate the kill within final loader work,
the following `.to('cuda:0')`, or the first generation.

**The historical sender/reason for SIGKILL is not recorded.** The parent trace
identifies the child and signal, not the sender. In particular, `si_uid=1000`
in this SIGCHLD trace is not identification of the killer. The accessible kernel,
system, and authentication logs contained no matching OOM-killer/kill record in
the failure window. The currently existing shared-user memory cgroup has zero
OOM kills and zero limit failures, but this does not exclude a historical
host OOM or a now-removed per-session limit. A kernel OOM kill, an external
supervisor, and an explicit kill cannot be distinguished retroactively from
these artifacts. There is no evidence sufficient to attribute the kills to a
particular person or container.

## New isolated startup checks

The standalone diagnostic uses the saved first TempCompass question and the
existing processor, input-building code, checkpoint loader, generation settings,
and 4,096-token output allowance. It records stages before/after loading and
CUDA transfer, token hashes, allocator statistics, generated IDs, and subprocess
exit status. Each checkpoint runs in a separate process on an initially empty
A100 GPU 0. Outputs are separate from benchmark results.

| Checkpoint | Load + CUDA transfer + generation | Loaded parameter bytes | First-question peak allocated |
| --- | --- | ---: | ---: |
| Vanilla 2B | Passed, EOS | 4,255,064,064 | 4.0362 GiB |
| QAT W4 | Passed, EOS | 4,255,064,064 | 4.0362 GiB |
| GRACE W4 | Passed, EOS | 4,255,064,064 | 4.0362 GiB |
| QAT W8 | Passed, EOS | 4,255,064,064 | 4.0362 GiB |
| GRACE W8 | Passed, EOS | 4,255,064,064 | 4.0362 GiB |

All five runs have the identical 324-token input and identical input-ID hash.
All local models answer `A` followed by EOS. This is a startup/generation check,
not a full-population accuracy result or a reproduction of the historical kill
conditions.

The environment matches the pinned failed run: Torch `2.5.1+cu121`, Transformers
`5.9.0`, FlashAttention `2.7.2.post1`, and Decord `0.6.0`. The five pinned
evaluation source hashes also still match. No evaluator or checkpoint was
modified for this investigation.

## Why W4/W8 do not save memory here

`evaluation/workers/video_worker.py` loads every model with
`Qwen3VLForConditionalGeneration.from_pretrained(..., dtype=torch.bfloat16)`.
It never invokes the repository's `maybe_convert_to_awq` conversion hook or
loads the QAT sidecar into packed low-bit layers.

Safetensors headers confirm that every tensor in the four local model exports
is BF16. Their `qat.bits` metadata describes quantization-aware training;
`qat_lora.merged` is true. Inference runs the merged export through ordinary
linear layers. There are no live LoRA adapters or low-bit inference modules
in the diagnostic model module inventory.

Local exports have 626 saved tensors (4,877,393,920 bytes), compared with
vanilla's 625 tensors (4,255,064,064 bytes). The difference is a saved
622,329,856-byte `lm_head.weight`. **It does not cause higher runtime memory:**
the loaded input embeddings and output head share the same Parameter in all
five models, and both parameter bytes and CUDA allocation after loading are
exactly equal. The sidecar is not loaded by this evaluator.

The current setup can measure behavior of the exported QAT/GRACE weights, but
its memory figures are BF16 inference figures. Packed low-bit deployment would
require an explicit loader/conversion path and separate validation.

## VideoMME question 618-1

The historical QAT W4 CUDA OOM occurs after 1,851 successful predictions, on
question `618-1`, in the language-model MLP:

```python
self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

It requests another 1.73 GiB with 1.48 GiB free on a 39.39-GiB device. PyTorch
reports 30.44 GiB allocated and 6.97 GiB reserved but unallocated. This is a
fragmentation/cached-memory symptom, rather than evidence of additional
quantization modules. It occurs during prefill, not model loading or a long
generated response.

This question samples 7,081 frames over a 3,540-second video at the declared
2-fps/256-pixel policy. It has 113,312 visual tokens and 37,906 text tokens,
151,218 input tokens total. Its input plus the output allowance fits the native
262,144-token context; that does not guarantee sufficient GPU memory.

Vanilla's historical successful run processes the same question with a
33.3253-GiB peak allocated and two output tokens. All 1,851 earlier saved QAT
input-ID hashes match vanilla's. Both models have the same 30.2416-GiB maximum
peak allocation over that shared prefix. Their output lengths differ, so their
allocation histories need not match. Weight bit-width labels do not explain
the difference in success.

A new QAT W4 process on an empty GPU also reproduced the MLP OOM for the exact
same input-ID hash. It reported 31.60 GiB allocated and 7.16 GiB reserved but
unallocated. Therefore the memory issue can develop within a single long-video
inference; it does not require the previous 1,851 questions.

The focused comparison on the same empty GPU produced:

| New isolated run on question 618-1 | Allocator | Result |
| --- | --- | --- |
| QAT W4 | Default | CUDA OOM, 31.60 GiB allocated + 7.16 GiB reserved but unused |
| Vanilla 2B | Default | Same CUDA OOM and same memory figures |
| QAT W4 | `expandable_segments:True` | Passed, answer `C` + EOS, peak allocated **33.3223 GiB** |

All three checks use the identical historical input-ID hash
`836c0a976dc48785bbdb111ad502cb938115dec10bad0d217ac9893b1b3bf405`.
The expandable-segments run finished with approximately 3.72 GiB free and
35.16 GiB reserved in total, below the device capacity. Resolution, frame
sampling, prompt, checkpoint, attention implementation, and output allowance
were unchanged. The positive allocator test supports fragmentation as the
cause of this specific allocation failure. Vanilla's historical successful
full run does not imply that a fresh process will reproduce the same allocator
state; the new default-allocator Vanilla check fails too.

This **verifies an allocator fix for the failing question**, not successful
completion of all VideoMME questions or the 8B teacher. Detailed stage/exit
records are in `allocator_diagnostic_20261023/`.

## Practical follow-up

For startup failures, retain explicit subprocess exit status and signal, plus
stage markers before/after checkpoint loading, CUDA transfer, preprocessing,
and generation. The existing matrix prints `FAILED` but does not save the exit
status in the attempt directory. The diagnostic script shows a working way to
preserve those facts without changing the existing results. Recheck GPU
availability before retrying the affected full jobs. Identifying any recurrent
SIGKILL sender requires host-side signal auditing or supervisor/OOM evidence at
the time it happens.

For the long-video allocation issue, the focused allocator comparison verifies
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing Torch. Keep
the video input policy and generated-output allowance unchanged when comparing
that change. A one-question pass does not certify the full long-video matrix,
and an 8B teacher has substantially larger weight/KV-cache requirements.

For a full retry, prepend that environment variable to the normal video
evaluation command and use a fresh `--output` directory to retain the original
memory telemetry for comparison. Existing successful scores can remain intact.
No full retry was launched during this investigation.

The focused check can be repeated from the workspace root as follows, choosing
an empty GPU and a new output directory:

```bash
reproduction/GRACE/.venv-lmms-eval/bin/python \
  reproduction/GRACE/experiments/released_bf16_init_32k_b128_p589824_t4096_seed42/reports/investigate_video_startup_20261023.py \
  --gpu 0 --long-only --long-methods qat_w4 \
  --allocator expandable_segments:True \
  --output /tmp/grace-video-allocator-recheck
```

## Artifacts

- `investigate_video_startup_20261023.py`: standalone, reproducible diagnostic.
- `startup_diagnostic_20261023/`: five successful startup checks and the
  reproduced QAT long-video OOM; per-stage logs and parent exit records.
- `allocator_diagnostic_20261023/`: focused long-video allocator comparison.
- `video_startup_investigation_20261023.json`: machine-readable evidence index.

Only diagnostic scripts and reports were created. Checkpoints, benchmark
predictions, completion markers, pinned evaluation source, and running benchmark
jobs were not edited or stopped.
