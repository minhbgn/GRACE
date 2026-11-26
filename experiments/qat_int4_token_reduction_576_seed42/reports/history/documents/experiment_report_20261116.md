# What the token-reduction experiment tells us

Personal experiment notes, 2026-11-16 (Asia/Singapore). This follows the [November 5/9 report](experiment_report.md). The older snapshots stay available there; this document is the current interpretation of the completed image, TempCompass and VideoMME results, plus the Video-MMMU results available today.

The useful result is that simple reduction already saves substantial work on short-answer tasks. The best method depends on the task and how much we keep. For images, merging is often a good choice at 50%, while resizing wins at 25% and 10%. TempCompass tolerates uniform selection down to 25% unusually well. On VideoMME, resizing to roughly half the adaptive baseline is a strong measured tradeoff; uniform selection and especially averaging become costly in accuracy at 25% and 10%.

There are two catches in interpreting VideoMME. Its saved resolution-25%/10% runs use the original spatial policy, whereas the repaired baseline adapts spatial size to video length. Also, the uniform/merge timings mix answers computed under different GPU scheduling conditions. The tables below retain the measurements and spell out what can actually be compared.

Reading guide: [method definitions](#2-resolution-uniform-and-merge-in-plain-terms), [token counts and rates](#4-actual-r100-token-counts-and-token-rates), [image results](#5-image-accuracy-and-inference-speed), [TempCompass](#6-tempcompass-keeping-one-quarter-works-surprisingly-well), [VideoMME](#7-videomme-useful-half-resolution-result-larger-losses-below-that-for-uniformmerge), and [integration plan](#11-a-concrete-next-step-integration-plan).

![Accuracy changes versus actual token retention across seven image suites, TempCompass and VideoMME](reports/accuracy_retention_20261116.png)

Each panel uses its own vertical scale. Points show measured settings; connecting lines are visual guides, not estimates between settings. The unmatched original VideoMME resolution-25%/10% runs are omitted. [PDF version](reports/accuracy_retention_20261116.pdf).

## 1. What finished, and what this experiment actually changes

| Suite | Completed / 10 |
| --- | --- |
| hallusionbench | 10/10 |
| mmbench | 10/10 |
| scienceqa | 10/10 |
| ai2d | 10/10 |
| mmmu | 10/10 |
| seedbench_img | 10/10 |
| mmstar | 10/10 |
| tempcompass | 10/10 |
| videomme | 10/10 |
| videommmu_perception | 10/10 |
| videommmu_comprehension | 3/10 |
| videommmu_adaptation | 1/10 |

That is **104/120 QAT configurations: 70 image and 34 video**. TempCompass, VideoMME and Video-MMMU perception are complete. Comprehension and adaptation are not complete, so this is not a final report for every Video-MMMU setting. Each image configuration covers 29,420 rows; each TempCompass configuration covers 1,580 questions; each VideoMME configuration covers 1,800 questions. Video-MMMU uses 300 questions per split. Long VideoMME videos, audio and subtitles are outside this experiment.

This is an **inference-only experiment on a frozen QAT W4 checkpoint**. No weights, quantization scales or reducer were trained during this sweep. It measures what happens when the already trained model sees fewer visual embeddings. The matched vanilla/BF16 sweeps are still incomplete; these results establish reduction effects within QAT W4, not an INT4-versus-BF16 speed or accuracy advantage.

The [parent training report](../released_bf16_init_32k_b128_p589824_t4096_seed42/experiment_report.md) gives the training details. The starting model was the authors' released BF16 **SFT** checkpoint. Our selected checkpoint then received CE-only LoRA-QAT, without GDKD/RCKA distillation. The project name “GRACE” does not mean this particular QAT model used KD.

| Choice | Value used | Why it matters |
| --- | --- | --- |
| Training data and exposure | 32,768 saved examples; batch 128; 256 updates; seed 42 | This is the existing control, not extra training for token reduction. |
| Training sequence cap | 4,096 total tokens | A training limit; evaluation videos can be much longer. Training averaged about 254.64 visual and 478.91 total tokens. |
| LoRA | Rank 32, alpha 64, dropout 0.05; 112 attention linears | The learned low-rank updates are merged for inference. |
| Training learning rates | LoRA 2e-4; merger 1e-5; scales 2e-4 | Useful values to preserve for a later matched joint-QAT arm. |
| Quantization | 4-bit weights, groups of 128; 196 decoder linears | Vision, embeddings and output head remain floating point. The decoder has 28 layers. |
| Runtime | Packed Triton W4A16; FP16 tiles/scales, FP32 accumulation | Packed INT4 storage with on-the-fly tile dequantization. It is not INT4×INT4 arithmetic. |
| Attention and generation | FlashAttention 2; batch one; greedy, one beam; max 256 new tokens; seed 42 | No sampling or beam-search variation; capped answers remain in the scores. |
| Image/frame ceiling | 589,824 pixels; image minimum 4,096 pixels | At most 576 merged tokens per image or two-frame video group. Actual counts are often smaller. |
| Sweep | Shared r100 baseline; resolution, uniform, merge at r050/r025/r010 | `r025` means **keep 25%**, i.e. remove roughly 75%. |
| Fixed-output profile | First 3 questions; 1 warmup and 3 timed repeats each; exactly 32 generated tokens | A small diagnostic, not a representative timing sample. |
| Saved software | Python 3.11.0; torch 2.5.1+cu121; transformers 5.9.0; Triton 3.1.0; lmms-eval 0.7.1 | The implementations and backend matter for both answers and speed. |

The packed backend was chosen because the environment lacked `awq_ext`. It checks every module's integer-code conversion, uses a locked CPU packing cache, and aborts if packed CUDA execution is unavailable. It does not silently fall back to a dense BF16 matrix. The numerical kernel gates recorded maximum absolute error 0.0078125. Those gates validate kernel arithmetic on their test inputs, not identical answers for all prompts. This is a correctness-first kernel, so its speed is not a claim about the fastest available INT4 serving stack. See [the kernel](../../evaluation/protocols/int4_gemm.py) and [packing/runtime code](../../evaluation/protocols/token_reduction_runtime.py).

Some timing vocabulary used below: **prefill** is reading the input prompt and building the attention cache; **decode** is generating subsequent answer tokens. The **KV cache** stores earlier tokens' attention keys and values so the model can reuse them. **M-RoPE** supplies the visual tokens' temporal and spatial position coordinates. A **percentage-point** change is subtraction between percentage accuracies: 61% to 60% is a loss of 1 pp.

## 2. Resolution, uniform and merge in plain terms

A visual token here means a **merged visual embedding sent into the language decoder**. The vision encoder starts from 16×16 spatial patches and merges 2×2 of them, so one decoder visual token corresponds to a 32×32 area of the resized input. For video, a temporal patch combines two sampled frames as well. These counts exclude text, timestamps and vision-boundary markers.

For an image of processed height H and width W, the visual count is `(H/32) × (W/32)`. For F sampled video frames it is `ceil(F/2) × (H/32) × (W/32)`. A 768×768 image has 24×24 = 576 tokens. A video with that spatial grid has up to 576 tokens **per two-frame group**, not 576 for the entire clip. At 2 sampled fps that is roughly 576 tokens per second before aspect-ratio rounding, source-size limits and the adaptive cap.

### Resolution: resize the whole picture before encoding

Your understanding that it reduces visual tokens is right. The mechanism is **shrinking the full image/frame**, with its whole field of view retained. It does not crop a smaller viewing window or change an attention-window setting.

The code starts from the actual baseline dimensions, sets a token-area budget, scales both dimensions by approximately the square root of the requested fraction, and rounds down to a valid 32-pixel grid. Thus 50% of tokens means about 70.7% of each side, not half of each side. Halving both sides would keep about 25% of the area.

For a square 24×24-token baseline, the current rounding produces 16×16 = 256 tokens at r050, 12×12 = 144 at r025, and 7×7 = 49 at r010. The nominal 50% example actually keeps 44.44%; the nominal 10% example keeps 8.51%. Different aspect ratios give different rounded fractions. This is why the measured token count belongs beside each result.

The smaller picture enters the vision encoder, so resolution can save both vision and decoder work. It can also lose fine detail before the encoder has a chance to interpret it. The reduction path can go below the usual training image minimum at very low budgets; the absolute floor is one merged spatial token. Images use PIL bicubic resizing; the video path resizes frames with antialiased bicubic interpolation, clamps/rounds to uint8, then disables additional processor resizing.

### Uniform: encode first, then keep evenly spaced embeddings

Yes, this is token pruning. The full baseline image/frame grid is encoded first. The resulting visual sequence is divided into K nearly equal contiguous bins, and the lower middle member of each bin is kept. For a span of N visual embeddings, `K = max(1, ceil(N × retention))`.

It does not score importance, inspect the question, or sample randomly. A 576-token span keeps 288, 144 or 58 tokens at our three budgets. Because rounding happens separately for each span, actual retention can be a little above the target.

“Uniform” describes spacing in the **flattened token sequence**. It is not a guaranteed evenly spaced 2-D checkerboard. Contiguous visual spans are separated by text and special markers; in video, the timestamp/boundary tokens keep the temporal groups separate. Every temporal group remains present, so this is spatial selection within groups, not frame dropping.

### Merge: average each bin instead of selecting one member

This uses the same bins and K as uniform. Each output embedding is the unweighted arithmetic mean of the bin's embeddings. The same averaging is applied to every aligned DeepStack feature stream. DeepStack is Qwen3-VL's way of feeding intermediate vision features into the first decoder layers; forgetting to reduce those streams would misalign visual information.

For example, a bin containing `[v0, v1, v2, v3]` becomes `(v0+v1+v2+v3)/4`, anchored at v1's original position. Uniform would just keep v1. The location is the lower middle member's original multimodal position; the code does not average coordinates or record token mass for later attention weighting.

These are contiguous bins in sequence order, not object regions or similarity clusters. They can cross a spatial row boundary. Averaging can smooth away a small distinctive feature, and no mechanism checks whether the bin members belong together. This is much simpler than ToMe-style similarity matching or an LLMC+ reducer.

### What stays aligned in all three methods

Uniform/merge compact the physical sequence **before decoder layer 0**. All 28 decoder layers therefore process fewer visual tokens and build shorter KV caches. This is actual removal, not just setting attention weights to zero. They still pay for the full baseline vision encoding.

Text, question tokens, video timestamps and vision-boundary markers are kept. Original temporal/height/width M-RoPE coordinates are selected at the representative indices; the sequence indices used for causal attention/cache progression are compacted. New output coordinates continue beyond the original prompt's multimodal maximum. At r100 there is no selection or averaging. The tiny-model tests and saved CUDA memory-equivalence gates check this plumbing; they are not a full-checkpoint benchmark rerun.

Source: [group planning, compaction and image preprocessing](../../evaluation/protocols/token_reduction_runtime.py), [video sampling/preprocessing](../../evaluation/protocols/video_protocol.py), and [video memory runtime](../../evaluation/protocols/video_memory_runtime.py).

## 3. The video choices that are easy to miss

Frames are sampled uniformly across the **full clip including both endpoints**. The requested count is `min(source_frames, ceil(duration_seconds × 2))`. Thus “2 fps” specifies approximate coverage density, not exactly one decoded frame every half second. There is no duration truncation or fixed frame-count limit. Two-frame temporal patches round up when the sampled count is odd.

The original video jobs hit vision/prefill OOMs on the roughly 40 GiB cards. TempCompass also initially failed because the runner requested a YAML name missing from the pinned harness. The recovery used the correct MC adapter/local scorer and a separate pinned output namespace, while preserving five successful original video passes.

| Recovery setting | Actual behavior and tradeoff |
| --- | --- |
| Vision chunk = 1 temporal group | Transfer/encode one two-frame group at a time. Vision attention already separates these groups, so this bounds working memory without shortening the clip. Repeated calls/transfers add overhead. |
| CPU feature staging | Merged and DeepStack features are staged on the host; compacted decoder features are transferred as needed. This saves GPU working memory but does not make transfers free. |
| Prefill chunk = 512 tokens | Run causal prefill in chunks with the full growing GPU KV cache. This bounds intermediate activations, not the final KV size. No cache eviction. |
| Adaptive baseline cap = 131,072 tokens | Reserve exact prompt/timestamp tokens and 256 output tokens; shrink spatial dimensions if the original grid would exceed the remaining visual allowance. Native model context is 262,144; this lower operational cap was chosen for feasibility. |
| Shared adaptive starting point | Compute the baseline before the requested reduction. All repaired methods on the same question start from the same grid, regardless of their retention. Exact text length can make grids differ between questions about the same clip. |
| Durable answer checkpoints | Later attempts replay verified answers and their original telemetry. This saves recomputation but makes a final attempt's elapsed time unsuitable as the time to compute its full population. |
| Parallel scheduling | Earlier attempts shared GPUs; later runs used one worker per GPU on multiple cards. Different workers can still share CPU/memory bandwidth. A scheduler memory estimate does not change token budgets. |

The adaptive policy is a genuine input change relative to the original policy; chunking/staging are execution changes. Keeping those separate is essential. TempCompass never activates the adaptive cap. VideoMME activates it on **846/1,800 questions: 846/900 medium and 0/900 short**. Its r100 contains only **50.14% of the original policy's summed visual tokens**. “100%” means 100% of the declared adaptive baseline, not maximum spatial detail for every video.

For intuition, a BF16 KV cache with 28 layers, 8 KV heads and head dimension 128 needs about `28 × 2 × 8 × 128 × 2 = 114,688` bytes per cached token, before other memory. That is roughly 10.68 GiB for 100,000 tokens. This architecture-based estimate explains why long prompts remain expensive despite small INT4 weight storage. Weight quantization does not quantize the KV cache in this experiment.

## 4. Actual r100 token counts and token rates

### Counts per evaluated question

The means below include every evaluated row, including text-only image questions and repeated questions about the same video. “Text” includes prompt formatting, timestamps and boundary markers. “Total input” is visual plus text. “Input + output” also includes the generated IDs, including EOS. These totals are not a claim that every generated token occupies a KV slot: the final sampled token need not be fed back into the decoder.

| Suite | Rows | Visual sum (exact) | Mean visual | Mean text | Mean total input | Mean input + output |
| --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 1129 | 380,159 | 336.721878 | 41.483614 | 378.205492 | 380.205492 |
| mmbench | 4329 | 653,110 | 150.868561 | 75.057750 | 225.926311 | 227.926311 |
| scienceqa | 4241 | 323,985 | 76.393539 | 79.534308 | 155.927847 | 157.927847 |
| ai2d | 3088 | 842,370 | 272.788212 | 53.295337 | 326.083549 | 328.083549 |
| mmmu | 900 | 250,776 | 278.640000 | 107.396667 | 386.036667 | 388.204444 |
| seedbench_img | 14233 | 7,772,322 | 546.077566 | 55.012717 | 601.090283 | 603.090283 |
| mmstar | 1500 | 285,192 | 190.128000 | 77.406000 | 267.534000 | 269.534000 |
| tempcompass | 1580 | 4,291,838 | 2,716.353165 | 153.131013 | 2,869.484177 | 2,871.484177 |
| videomme | 1800 | 141,289,224 | 78,494.013333 | 2,959.166667 | 81,453.180000 | 81,455.180000 |
| videommmu_perception | 300 | 33,545,343 | 111,817.810000 | 5,178.966667 | 116,996.776667 | 117,001.306667 |
| videommmu_comprehension | 300 | 33,529,601 | 111,765.336667 | 5,182.033333 | 116,947.370000 | 116,955.280000 |

The visual sum is an exact integer; dividing it by the row count gives the exact mean as a fraction. Displayed decimal means are rounded to six places. Full sums for text/output and all configurations are in the [analysis JSON](reports/completed_analysis_20261116.json) and [CSV](reports/completed_analysis_20261116.csv).

HallusionBench has **178 text-only rows**; ScienceQA has **2,224**. These dilute both the average visual count and the whole-suite speed benefit of visual reduction. A 576-token ceiling does not make those suites 576-token workloads. MMMU can include multiple images; the 576 ceiling is per image, not a universal per-question guarantee.

### VideoMME split by video length

| Duration | Rows | Visual sum (exact) | Mean visual | Mean text | Mean input | Adapted rows |
| --- | --- | --- | --- | --- | --- | --- |
| short | 900 | 37,836,849 | 42,040.943333 | 785.762222 | 42,826.705556 | 0 |
| medium | 900 | 103,452,375 | 114,947.083333 | 5,132.571111 | 120,079.654444 | 846 |

Long VideoMME videos were not evaluated, so there is no measured long-video baseline, token rate or speedup to report. Both short and medium have 900 questions. Mean source length is about 80.67 s for short and 516.81 s for medium. Mean total input is 42,826.71 versus 120,079.65 tokens: medium questions are already near the operational cap.

### “Visual tokens per second” has two different meanings

**Source-video rate** describes how densely we represent a clip. For question i, it is `visual_tokens_i / source_duration_i`. The mean-rate column below averages those per-question rates. The pooled column is `sum(visual_tokens) / sum(source_duration)`; it weights longer clips more strongly. Both use saved source durations, not an assumed exact two frames per second.

**Model throughput** describes measured processing speed. It is `sum(retained visual tokens) / sum(model_seconds)`. The model timer also includes processing text and generating answers, so this is a workload-normalized rate, not isolated vision-encoder throughput or generated tokens/s.

| Video population | Mean source s | Mean frames | Mean visual tokens/source s | Pooled visual tokens/source s | Visual tokens/model s |
| --- | --- | --- | --- | --- | --- |
| tempcompass | 11.970608 | 24.432278 | 241.853560 | 226.918571 | 899.216892 |
| videomme | 298.740010 | 597.966667 | 389.445804 | 262.750254 | 867.535576 |
| videommmu_perception | 506.654278 | 1013.800000 | 295.044934 | 220.698442 | 820.994901 |
| videommmu_comprehension | 506.654278 | 1013.800000 | 294.994792 | 220.594874 | 820.938784 |
| videomme short | 80.666365 | 161.850000 | 524.531466 | 521.170666 | 963.336121 |
| videomme medium | 516.813654 | 1034.083333 | 254.360142 | 222.414951 | 837.089148 |

Short VideoMME averages **524.531466 visual tokens per source second**, versus **254.360142** for medium. That difference mainly reflects the adaptive spatial policy and clip geometry; it is not evidence that medium videos inherently contain less visual information. The source-rate columns count each evaluated question, so a clip asked about three times is represented three times. There is no meaningful source-video-seconds denominator for a still image.

For completeness, measured processing rates for **every available r100 benchmark** are below. “Input tokens/prefill s” includes retained visual and text tokens, using summed tokens divided by summed prefill time. It is not output-generation throughput.

| Suite | Mean model s/question | Visual tokens/model s | Input tokens/prefill s |
| --- | --- | --- | --- |
| hallusionbench | 0.460411 | 731.351166 | 1049.249465 |
| mmbench | 0.349103 | 432.161022 | 911.844365 |
| scienceqa | 0.272766 | 280.070022 | 857.437312 |
| ai2d | 0.424810 | 642.142087 | 1011.058973 |
| mmmu | 0.487483 | 571.589714 | 1040.123697 |
| seedbench_img | 0.637399 | 856.728532 | 1130.297030 |
| mmstar | 0.381174 | 498.795996 | 958.113981 |
| tempcompass | 3.020799 | 899.216892 | 1093.237072 |
| videomme | 90.479302 | 867.535576 | 989.595607 |
| videommmu_perception | 136.197935 | 820.994901 | 968.040396 |
| videommmu_comprehension | 136.143326 | 820.938784 | 967.611174 |

### Average total tokens across benchmarks

There are two reasonable averaging choices; they answer different questions.

| r100 population | Rows | Row-weighted mean visual | Row-weighted mean total input | Equal-suite mean visual | Equal-suite mean total input |
| --- | --- | --- | --- | --- | --- |
| Seven image suites | 29,420 | 357.169069 | 420.710979 | 264.516822 | 334.400593 |
| Four available video baselines | 3,980 | 53,431.157286 | 55,611.244472 | 76,198.378291 | 79,566.702711 |
| All 11 available baselines | 33,400 | 6,681.554491 | 6,997.307485 | 27,876.830084 | 29,146.146818 |

Row-weighted means describe the actual saved question workload; equal-suite means give each benchmark/split one vote. Video-MMMU adaptation has no completed r100 and is excluded, rather than treated as zero. These are token-count summaries, not averages of benchmark quality.

## 5. Image accuracy and inference speed

Scores are percentages. HallusionBench uses saved row accuracy; MMBench uses circular accuracy on the supplied rotations. Other suites use their saved local primary metric. Unlike metrics are not averaged into one accuracy number here. All 70 image configurations completed; no answer hit the 256-token cap.

| Suite | baseline_r100 | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 64.92 | 64.48 | 63.77 | 62.27 | 63.60 | 62.80 | 59.61 | 63.95 | 61.56 | 59.96 |
| mmbench | 89.26 | 87.63 | 84.97 | 78.61 | 87.97 | 83.68 | 75.26 | 88.92 | 83.42 | 69.85 |
| scienceqa | 83.49 | 82.86 | 82.10 | 79.56 | 82.36 | 81.00 | 78.68 | 83.31 | 81.68 | 79.04 |
| ai2d | 81.22 | 79.76 | 78.56 | 75.06 | 79.79 | 76.13 | 72.18 | 80.76 | 77.62 | 72.93 |
| mmmu | 67.78 | 66.67 | 66.11 | 63.00 | 65.22 | 63.44 | 61.67 | 66.00 | 63.56 | 61.00 |
| seedbench_img | 75.30 | 73.86 | 72.63 | 69.87 | 73.80 | 71.43 | 63.85 | 74.00 | 70.91 | 61.29 |
| mmstar | 67.27 | 64.87 | 63.13 | 59.20 | 64.13 | 60.73 | 56.07 | 65.53 | 62.40 | 56.87 |

At 50%, merge is the highest-scoring reducer on MMBench, ScienceQA, AI2D, SEED and MMStar. Resolution wins on HallusionBench and MMMU. At 25% and 10%, resolution wins on all seven suites, even though grid rounding often gives it **fewer** tokens than uniform/merge at the same nominal fraction.

The severity depends on the benchmark. ScienceQA merge-50% loses only about 0.19 pp, while MMStar loses about 1.73 pp. At 10%, MMBench drops about 10.65 pp with resolution and 19.42 pp with merge. This makes MMBench/MMStar useful stress tests for the next content-aware selector, alongside a detail-reading diagnostic. These suite-level losses alone do not tell us whether OCR, spatial relations or another skill caused the mistakes.

### Normal-answer model speedup, for every image benchmark

Speedup is `sum(baseline model seconds) / sum(reduced model seconds)` on the same question population. 2× means half the measured model time. Each cell is relative to that suite's r100 on the same packed QAT model. It excludes CPU media preparation, load/packing and file I/O.

| Suite | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 1.468× | 1.826× | 2.212× | 1.410× | 1.778× | 2.105× | 1.408× | 1.775× | 2.107× |
| mmbench | 1.257× | 1.405× | 1.551× | 1.231× | 1.395× | 1.526× | 1.234× | 1.394× | 1.527× |
| scienceqa | 1.140× | 1.198× | 1.263× | 1.106× | 1.191× | 1.259× | 1.106× | 1.190× | 1.257× |
| ai2d | 1.397× | 1.672× | 1.960× | 1.334× | 1.642× | 1.898× | 1.336× | 1.636× | 1.895× |
| mmmu | 1.330× | 1.530× | 1.733× | 1.289× | 1.507× | 1.704× | 1.291× | 1.495× | 1.698× |
| seedbench_img | 1.592× | 2.080× | 2.498× | 1.471× | 1.991× | 2.384× | 1.470× | 1.991× | 2.373× |
| mmstar | 1.290× | 1.447× | 1.635× | 1.261× | 1.424× | 1.604× | 1.265× | 1.421× | 1.602× |

Most image answers contain exactly two generated tokens: a choice/yes-no token plus EOS. MMMU averages 2.168 at baseline, so its natural-answer ratios can also reflect output-length changes. These are historical observed timings across the original GPU-0/2 runs; method order, card assignment and concurrent host load were not randomized. They are useful workload measurements, not a precision-controlled serving benchmark.

### Fixed 32-output-token speedup

| Suite | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 1.088× | 1.130× | 1.161× | 1.066× | 1.118× | 1.138× | 1.070× | 1.114× | 1.146× |
| mmbench | 1.005× | 1.010× | 1.012× | 1.004× | 1.010× | 1.006× | 1.006× | 1.011× | 1.008× |
| scienceqa | 1.017× | 1.024× | 1.031× | 1.015× | 1.021× | 1.024× | 1.020× | 1.021× | 1.027× |
| ai2d | 1.035× | 1.037× | 1.065× | 1.025× | 1.045× | 1.071× | 1.033× | 1.044× | 1.065× |
| mmmu | 1.026× | 1.026× | 1.029× | 1.007× | 1.031× | 1.030× | 1.012× | 1.028× | 1.030× |
| seedbench_img | 1.086× | 1.123× | 1.149× | 1.073× | 1.119× | 1.145× | 1.078× | 1.121× | 1.146× |
| mmstar | 1.033× | 1.042× | 1.068× | 1.031× | 1.047× | 1.063× | 1.038× | 1.044× | 1.067× |

The difference between these two tables matters. For example, SEED resolution-10% shows **2.498×** on normal answers but **1.149×** on the three 32-token profiles. On short answers, input prefill is a large share of time. With 32 output tokens, repeated autoregressive decoder/weight work dominates and removing input tokens saves a smaller fraction. The profile also uses a different, tiny question population, so answer length is not the only uncontrolled difference. Neither timing definition should silently replace the other.

ScienceQA has the weakest normal-answer gains, consistent with its many text-only rows and low starting visual count. SEED has the largest visual workload and stronger gains. Image maximum allocated memory at r100 is about 2.12–2.29 GiB; resolution saves roughly 0.44–7.03% across suite/budget combinations, and uniform/merge roughly 0.34–1.73%. Weight storage and other fixed allocations remain, so a 90% visual-token cut does not imply a 90% memory cut.

## 6. TempCompass: keeping one quarter works surprisingly well

All methods use the same repaired input/scoring policy, with no adaptive spatial changes and no capped answers. The score is deterministic local MC accuracy, not the official API-judged result. The model timers come from the original serial recovery runs.

| Configuration | Score % | Delta pp | Mean visual | Retained % | Model s | Observed time ratio | 32-token ratio | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 61.329 | +0.000 | 2,716.353 | 100.000 | 3.021 | 1.000× | 1.000× | 3.160 |
| resolution_r050 | 60.570 | -0.759 | 1,252.280 | 46.102 | 1.679 | 1.800× | 1.224× | 2.563 |
| resolution_r025 | 60.190 | -1.139 | 657.394 | 24.201 | 1.132 | 2.668× | 1.302× | 2.336 |
| resolution_r010 | 58.354 | -2.975 | 222.729 | 8.200 | 0.727 | 4.153× | 1.389× | 2.171 |
| uniform_r050 | 61.835 | +0.506 | 1,358.177 | 50.000 | 1.791 | 1.687× | 1.169× | 2.614 |
| uniform_r025 | 61.646 | +0.316 | 681.326 | 25.082 | 1.173 | 2.575× | 1.282× | 2.345 |
| uniform_r010 | 59.367 | -1.962 | 273.156 | 10.056 | 0.804 | 3.755× | 1.324× | 2.261 |
| merge_r050 | 61.582 | +0.253 | 1,358.177 | 50.000 | 1.788 | 1.690× | 1.184× | 2.614 |
| merge_r025 | 61.329 | +0.000 | 681.326 | 25.082 | 1.169 | 2.583× | 1.295× | 2.345 |
| merge_r010 | 56.329 | -5.000 | 273.156 | 10.056 | 0.807 | 3.744× | 1.342× | 2.261 |

Uniform-25% is a practical candidate: **61.646% versus 61.329%**, with about **681 visual tokens instead of 2,716** and **2.575×** normal-answer model speedup. Merge-25% matches aggregate baseline accuracy with 2.583× observed speedup. Resolution-25% is slightly faster at 2.668× but loses 1.139 pp. At 10%, merge loses 5.000 pp, compared with 1.962 pp for uniform.

The small positive changes are not established accuracy gains. Uniform-25% fixes 52 baseline errors but introduces 47 new ones: just five net extra correct answers. Merge-25% fixes and introduces 71 each. Equal totals do not mean the same predictions.

The category breakdown helps explain why the overall score can conceal a problem:

| Configuration | action | direction | speed | order | attribute_change | Fixes | New errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 95.858 | 44.478 | 43.533 | 57.285 | 64.236 | 0 | 0 |
| resolution_r050 | 94.083 | 45.672 | 43.849 | 54.967 | 62.847 | 36 | 48 |
| resolution_r025 | 93.491 | 44.776 | 42.271 | 55.960 | 63.194 | 47 | 65 |
| resolution_r010 | 91.716 | 44.478 | 39.432 | 53.974 | 60.764 | 68 | 115 |
| uniform_r050 | 95.858 | 45.373 | 46.688 | 56.954 | 62.847 | 40 | 32 |
| uniform_r025 | 95.266 | 44.776 | 47.003 | 58.940 | 60.764 | 52 | 47 |
| uniform_r010 | 94.970 | 45.970 | 43.849 | 52.980 | 56.944 | 76 | 107 |
| merge_r050 | 95.266 | 45.075 | 46.057 | 57.947 | 62.153 | 40 | 36 |
| merge_r025 | 94.675 | 46.567 | 46.688 | 56.623 | 60.417 | 71 | 71 |
| merge_r010 | 92.899 | 42.687 | 42.902 | 48.344 | 52.431 | 79 | 158 |

At uniform-25%, speed rises from **43.533% to 47.003%** and order from **57.285% to 58.940%**, but attribute change falls from **64.236% to 60.764%**. At merge-10%, order falls to **48.344%** and attribute change to **52.431%**. This suggests keeping category-level checks when choosing a reducer: tolerating broad action recognition does not establish preservation of small changes or event order. A plausible explanation is that averaging removes distinctive visual cues, but that would need error inspection to verify.

## 7. VideoMME: useful half-resolution result, larger losses below that for uniform/merge

These are 1,800 short+medium questions, with two generated tokens per answer and no cap hits. Rows marked “policy differs” are the preserved original resolution runs. Their “retained %” denominator is their own original policy; recovery rows use the adaptive baseline denominator.

**Timing caveat for the following table:** the uniform/merge full-population ratios are descriptive ratios of mixed historical timers. They are **not controlled standalone speedups**. Even their fixed-32 profiles were recorded under different scheduling conditions. The baseline and resolution-50% are the original serial recovery pair and provide the cleaner full-population comparison.

| Configuration | Score % | Delta pp | Mean visual | Retained % | Model s | Observed time ratio | 32-token ratio | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_r100 | 61.500 | +0.000 | 78,494.013 | 100.000 | 90.479 | 1.000× | 1.000× | 16.170 |
| resolution_r050 | 61.222 | -0.278 | 35,840.322 | 45.660 | 43.943 | 2.059× | 1.982× | 9.150 |
| resolution_r025 | 61.722 | policy differs | 38,893.043 | 24.842 | 37.531 | N/A | N/A | 29.835 |
| resolution_r010 | 59.000 | policy differs | 13,557.088 | 8.659 | 14.469 | N/A | N/A | 13.915 |
| uniform_r050 | 60.611 | -0.889 | 39,279.495 | 50.041 | 65.611 | 1.379× | 0.721× | 9.506 |
| uniform_r025 | 57.444 | -4.056 | 19,689.425 | 25.084 | 46.465 | 1.947× | 1.040× | 6.248 |
| uniform_r010 | 52.944 | -8.556 | 7,942.640 | 10.119 | 35.431 | 2.554× | 1.438× | 4.948 |
| merge_r050 | 60.444 | -1.056 | 39,279.495 | 50.041 | 64.730 | 1.398× | 0.755× | 9.506 |
| merge_r025 | 56.722 | -4.778 | 19,689.425 | 25.084 | 45.221 | 2.001× | 1.063× | 6.248 |
| merge_r010 | 48.611 | -12.889 | 7,942.640 | 10.119 | 34.436 | 2.627× | 1.482× | 4.948 |

Resolution-50% retains **45.660%** of adaptive-baseline visual tokens. Accuracy changes by **−0.278 pp**; model time falls **90.479 → 43.943 s/question, 2.059×**, and maximum allocated memory falls **16.170 → 9.150 GiB**. The historical attempt-time estimate is 55.40 → 30.49 hours, about 1.82×; that broader interval also includes setup, media work and profiling and is not a repeatable serving-latency estimate.

Uniform is consistently ahead of merge at the same actual token count: +0.167 pp at 50%, +0.722 pp at 25%, and **+4.333 pp at 10%**. Both methods degrade substantially at the smaller budgets. This makes “preserve distinctive tokens instead of averaging everything” a useful hypothesis for DART/content-aware pruning. It does not prove DART will win.

### Short and medium videos behave differently

| Configuration | short | medium | Fixes | New errors |
| --- | --- | --- | --- | --- |
| baseline_r100 | 68.556 | 54.444 | 0 | 0 |
| resolution_r050 | 68.667 | 53.778 | 69 | 74 |
| resolution_r025 | 68.444 | 55.000 | 87 | 83 |
| resolution_r010 | 64.778 | 53.222 | 91 | 136 |
| uniform_r050 | 66.889 | 54.333 | 53 | 69 |
| uniform_r025 | 63.556 | 51.333 | 64 | 137 |
| uniform_r010 | 57.333 | 48.556 | 71 | 225 |
| merge_r050 | 67.333 | 53.556 | 50 | 69 |
| merge_r025 | 62.333 | 51.111 | 74 | 160 |
| merge_r010 | 51.333 | 45.889 | 77 | 309 |

The original resolution-25%/10% group scores and error swaps above are descriptive only; they do not isolate a reduction effect against the repaired baseline. For the matched recovery methods, uniform-50% barely changes medium accuracy (54.444 → 54.333), but short falls 68.556 → 66.889. At uniform-10%, short loses **11.222 pp** and medium **5.889 pp**. Merge-10% loses **17.222 pp** on short and **8.556 pp** on medium. Longer clips are not automatically more sensitive in percentage-point loss; their lower baseline accuracy, different task mix and already reduced spatial input complicate that interpretation.

### Why the old resolution-25% result cannot be called the best tradeoff

It scores 61.722%, but keeps **38,893 visual tokens/question**, more than recovery resolution-50%'s **35,840**. The nominal labels refer to different starting points. Relative to the adaptive r100 total, the original r025 actually keeps **49.549%**, and original r010 keeps **17.271%**. Their original unchunked runtime also differs. The 61.722% score is a valid saved result, but “25% beats 50% with fewer tokens” would be false for these two rows. A matched repaired resolution-25%/10% speedup is **not available**; computing 90.479/37.531 or 90.479/14.469 would mix input/runtime policies.

### What speed evidence survives the replay and concurrency history?

The audit matches each completed question's exact timer tuple to its earliest saved attempt. That separates historical shared-GPU records from records first computed with one evaluation worker per GPU. A final attempt having `parallel_evaluation=1` does not make a restored prefix newly measured under that condition.

| Configuration | Original serial rows | One-worker rows | Shared-GPU rows | One-worker paired ratio |
| --- | --- | --- | --- | --- |
| baseline_r100 | 1800 | 0 | 0 | 1.000× |
| resolution_r050 | 1800 | 0 | 0 | 2.059× |
| uniform_r050 | 0 | 874 | 926 | 1.760× |
| uniform_r025 | 0 | 816 | 984 | 2.824× |
| uniform_r010 | 0 | 768 | 1032 | 4.494× |
| merge_r050 | 0 | 873 | 927 | 1.802× |
| merge_r025 | 0 | 808 | 992 | 2.953× |
| merge_r010 | 0 | 750 | 1050 | 4.612× |

The one-worker paired ratios compare **the same question indices** against their baseline timings. For uniform-50%/25%/10% these are **1.760× / 2.824× / 4.494×** over 874/816/768 questions. Merge gives **1.802× / 2.953× / 4.612×** over 873/808/750 questions. Each method's subset differs and mostly covers medium videos; do not rank these methods from those ratios or generalize them to all 1,800 questions. Dates, cards and host load also differ. These are the best available subset estimates, not a clean complete-population benchmark. No supported full-population standalone speedup exists for these six mixed-history runs.

The baseline spends **90.97%** of model time in decoder prefill, **8.45%** in vision/embedding, about **0.45%** in reduction/staging and **0.13%** in decode. It generates only two tokens. That is why reducing long input helps much more than speeding output generation would. With aggressive uniform/merge, the unshrunk vision cost becomes a larger share; the saved mixed-history merge-10% timers attribute about 59.66% to vision/embedding. Treat that percentage as descriptive because contention also changes component balance.

Each question independently decodes/prepares its video and computes vision features. Questions sharing a clip do not share a vision cache or prompt KV cache. CPU decoding/resizing, memory traffic, staging and many 512-token prefill calls help explain the long elapsed runs. Model timers include synchronized vision/embedding, reduction/host staging and decoder work, including transfers inside those paths. They exclude initial input construction/media decoding, model load/packing, durable answer writes and final scoring/hashing. The wall-minus-model residual cannot be assigned to preprocessing alone. See the earlier [runtime analysis](reports/runtime_analysis_20261111.md) and [within-suite comparison](reports/within_suite_runtime_ratios_20261111.md).

## 8. Other video results already completed

Every row here covers all 300 questions in its split. “Original” marks a different spatial/execution policy. Recovery timings were first recorded with one worker per GPU, but on different cards/dates and with multiple workers sharing the host; they remain observed ratios, not controlled latency benchmarks.

| Split | Configuration | Origin | Score % | Cap hits | Mean visual | Model s | Observed ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| videommmu_perception | baseline_r100 | recovery | 44.000 | 2 | 111,817.810 | 136.198 | 1.000× |
| videommmu_perception | resolution_r050 | recovery | 45.333 | 0 | 50,910.127 | 67.093 | 2.030× |
| videommmu_perception | resolution_r025 | recovery | 38.333 | 0 | 26,176.910 | 41.660 | 3.269× |
| videommmu_perception | resolution_r010 | original | 38.000 | 1 | 24,842.913 | 26.388 | N/A |
| videommmu_perception | uniform_r050 | recovery | 41.333 | 1 | 55,961.920 | 73.938 | 1.842× |
| videommmu_perception | uniform_r025 | recovery | 34.333 | 1 | 28,044.453 | 44.895 | 3.034× |
| videommmu_perception | uniform_r010 | recovery | 34.000 | 1 | 11,360.583 | 29.300 | 4.648× |
| videommmu_perception | merge_r050 | recovery | 41.333 | 0 | 55,961.920 | 74.198 | 1.836× |
| videommmu_perception | merge_r025 | recovery | 33.000 | 0 | 28,044.453 | 45.163 | 3.016× |
| videommmu_perception | merge_r010 | recovery | 31.667 | 0 | 11,360.583 | 28.800 | 4.729× |
| videommmu_comprehension | baseline_r100 | recovery | 23.667 | 5 | 111,765.337 | 136.143 | 1.000× |
| videommmu_comprehension | resolution_r050 | recovery | 20.333 | 3 | 50,870.200 | 66.804 | 2.038× |
| videommmu_comprehension | resolution_r010 | original | 21.667 | 4 | 24,842.913 | 26.929 | N/A |
| videommmu_adaptation | resolution_r010 | original | 27.667 | 40 | 24,842.913 | 31.452 | N/A |

Perception is another case where modest resizing is promising: **45.333% at resolution-50% versus 44.000% baseline**, four additional correct answers, with an observed 2.030× model-time ratio. Baseline has two capped answers and resolution-50% none; that difference also belongs in the interpretation. Resolution-25% loses 5.667 pp, uniform-25% 9.667 pp, and merge-25% 11.000 pp. At equal counts, averaging is again worse than uniform at the smaller budgets. The original r010 cannot provide a matched repaired 10%-resolution comparison.

Comprehension moves in the opposite direction at half resolution: **23.667 → 20.333%, −3.333 pp**, with five versus three capped answers. This blocks any claim that resolution-50% preserves quality across all video tasks. Its other repaired cells are missing. Adaptation has only the original r010 score, **27.667%**, with **40/300 answers capped (13.33%)**. There is no completed adaptation baseline, so its accuracy loss, speedup and r100 token rate are unavailable.

## 9. Token counts at every evaluated budget

These are mean decoder visual tokens per question. Image means include text-only rows. Uniform and merge have identical counts by design. Video original-resolution cells retain the policy caveat above; missing cells are not estimated.

| Suite | baseline_r100 | res_r050 | res_r025 | res_r010 | uniform_r050 | uniform_r025 | uniform_r010 | merge_r050 | merge_r025 | merge_r010 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hallusionbench | 336.722 | 157.317 | 79.570 | 28.800 | 168.461 | 84.392 | 34.023 | 168.461 | 84.392 | 34.023 |
| mmbench | 150.869 | 68.048 | 35.817 | 12.532 | 75.473 | 37.802 | 15.526 | 75.473 | 37.802 | 15.526 |
| scienceqa | 76.394 | 34.996 | 17.180 | 6.623 | 38.304 | 19.248 | 7.810 | 38.304 | 19.248 | 7.810 |
| ai2d | 272.788 | 125.861 | 64.426 | 22.930 | 136.519 | 68.427 | 27.633 | 136.519 | 68.427 | 27.633 |
| mmmu | 278.640 | 128.134 | 65.732 | 23.324 | 139.427 | 69.887 | 28.222 | 139.427 | 69.887 | 28.222 |
| seedbench_img | 546.078 | 260.031 | 129.535 | 49.876 | 273.244 | 136.673 | 55.018 | 273.244 | 136.673 | 55.018 |
| mmstar | 190.128 | 86.625 | 45.109 | 15.844 | 95.131 | 47.663 | 19.431 | 95.131 | 47.663 | 19.431 |
| tempcompass | 2,716.353 | 1,252.280 | 657.394 | 222.729 | 1,358.177 | 681.326 | 273.156 | 1,358.177 | 681.326 | 273.156 |
| videomme | 78,494.013 | 35,840.322 | 38,893.043 | 13,557.088 | 39,279.495 | 19,689.425 | 7,942.640 | 39,279.495 | 19,689.425 | 7,942.640 |
| videommmu_perception | 111,817.810 | 50,910.127 | 26,176.910 | 24,842.913 | 55,961.920 | 28,044.453 | 11,360.583 | 55,961.920 | 28,044.453 | 11,360.583 |
| videommmu_comprehension | 111,765.337 | 50,870.200 | — | 24,842.913 | — | — | — | — | — | — |
| videommmu_adaptation | — | — | — | 24,842.913 | — | — | — | — | — | — |

The key comparison unit for later methods should be **actual retained counts**, not just configuration labels. At 25%, image resolution keeps about 22.5–23.7% versus uniform/merge's 25.0–25.2%; TempCompass resolution keeps 24.201% versus 25.082%. Better accuracy despite fewer tokens makes the resize baseline worth keeping in the next study.

## 10. How basic are these methods, compared with LLMC+?

They are deliberately simple, deterministic controls. They ask whether reducing input is useful at all, and how much quality a content-blind rule loses. They require no calibration, training, attention score or feature-similarity search. That makes their selection overhead small and their behavior easy to audit, while giving them no way to protect a tiny answer-critical region.

LLMC+ is a **benchmark and toolkit**, not one token reducer. It compares methods with attention/similarity scoring, pruning/merging, different insertion points, and video segmentation. Its controlled LLM-side comparison places reduction at the fifth layer; that differs from our reduction before layer 0. Its Table 5 includes LLaVA-1.5-7B with 576 starting tokens and retained budgets 192/128/64. It also studies quantization combined with reduction, so simply composing quantization and pruning is already established prior work. [LLMC+ v2, Sections 3–5](https://arxiv.org/html/2508.09981v2).

| Approach | How it chooses what remains | Main difference from our sweep |
| --- | --- | --- |
| Our resolution | Smaller full input grid | Saves vision work; fine detail is lost before encoding. |
| Our uniform | Evenly spaced sequence representatives | No content/question score; removes tokens before every decoder layer. |
| Our merge | Unweighted means of contiguous bins | No similarity matching, token mass correction or learned grouping. |
| FastV | Attention-guided token ranking after early decoder processing | Pays for early full-token layers and scoring before later-layer savings. |
| DART | Keeps pivots plus tokens with low duplication to them | Content-dependent diversity selection rather than uniform spacing or averaging. |

[FastV](https://arxiv.org/abs/2403.06764) prunes visual tokens using early-layer attention. In the inspected [LightCompress implementation](https://raw.githubusercontent.com/ModelTC/LightCompress/main/llmc/compression/token_reduction/fastv.py), scores average heads from the last prompt position, and the retained count is `round(N × (1-rate))`. The hook explicitly obtains attentions by invoking the preceding block's forward, so reproducing that implementation can add more work than a cheap index selection. Full-attention materialization/backend changes must be measured, especially for our long videos. `rate` is the fraction removed, whereas our `retention` is the fraction kept.

[DART](https://arxiv.org/html/2502.11494v1) uses pivot tokens and retains visually less duplicated tokens. The inspected [toolkit code](https://raw.githubusercontent.com/ModelTC/LightCompress/main/llmc/compression/token_reduction/dart.py) selects image/text pivots using key-state norms and expands the retained set using low cosine similarity in normalized hidden states. Pivot counts and integer rounding affect the realized count; it is not enough to copy our retention value into a differently named field. DART avoids needing a full attention-score matrix, which makes it a useful first integration candidate, subject to cache/API compatibility and measured selection cost.

The [toolkit configuration guide](https://llmc-en.readthedocs.io/en/stable/advanced/token_reduction.html) separates token-only reduction from a quantization-plus-reduction path. For our QAT composition, use the selector machinery while preserving the trained weights/scales; applying a new RTN/GPTQ/AWQ calibration would change the experiment. The [model registry inspected for this report](https://raw.githubusercontent.com/ModelTC/LightCompress/main/llmc/models/__init__.py) lists Qwen2-VL, Qwen2.5-VL and Qwen3 text support, but no explicit Qwen3-VL wrapper. That makes Qwen3-VL a port/validation task; it is not plug-and-play compatibility established by the name “Qwen3”. Recheck and pin the exact repository commit during implementation.

Our observed failures suggest useful things to test, rather than guaranteeing an advanced method will win:

- **Preserve unusual detail.** Uniform beating merge at aggressive video budgets suggests that retaining distinctive embeddings may help more than averaging unrelated patches. Test a DART-style selector at the same counts.
- **Let early processing happen before selection.** Our layer-0 pruning is maximally early and saves the most decoder-layer work, but contextualized features could support better decisions later. Compare placement with its extra compute included.
- **Separate spatial and temporal reduction.** Every current video temporal group survives. We have not tested removing repeated information across time. A later video-specific method could address that, but it must also protect event order and brief attribute changes.
- **Keep resizing as a strong control.** LLMC+ selectors that retain full vision inputs must earn back their selection cost and full vision cost. Beating uniform alone is insufficient if resolution is already better at the target latency.
- **Treat output length as part of the workload.** A method useful for two-token answers may have a much smaller benefit for longer explanations. Report both natural-answer and fixed-length timing.

## 11. A concrete next-step integration plan

This follows the [existing LLMC+/QAT proposal](../../../../plans/3_llmc_grace_proposal.md). It is a plan based on the results, not an LLMC+ installation or launch performed for this report.

### First: run a defined LLMC+ token-only reproduction

Use an isolated `reproduction/LLMC/` checkout/environment and pin its commit, model revision, paper version, configs and scorer. Start with LLaVA-1.5-7B: full 576-token baseline, then FastV and DART retaining 192, 128 and 64 tokens. That is seven configurations. Reproduce the selected Table 5 slice on GQA, MMBench EN, MME, POPE, TextVQA, VizWiz and ScienceQA. Verify the paper's aggregate and MME scale rather than averaging raw incompatible metrics. A subset is a partial reproduction, and the current GRACE seven-suite scores are not directly comparable to the paper's mean.

Before a broad run, verify the full-token reference, actual retained counts, zero/one-based layer numbering, removed-versus-kept fields, output parsing and attention backend on a fixed smoke manifest. LLaVA's 576 starting tokens are genuinely fixed in that setting; Qwen3-VL's are not. A BF16 7B model is plausibly feasible at batch one on an A100 40GB, but measure the longest intended input before estimating cost. Use the GPU-ownership workflow before any launch.

### Second: port one selector onto the frozen QAT model

Start with DART as the first candidate because it offers content-dependent selection without full attention matrices; retain FastV as a contrasting attention-based candidate. This is a reason to test DART first, not a predicted winner. Add the selector behind a new optional adapter in the GRACE evaluation codebase, rather than changing the sources pinned by the finished runs.

Suggested code boundaries:

| Area | Proposed responsibility |
| --- | --- |
| New selector module under `evaluation/protocols/` | Return ordered original token indices/selection metadata, with explicit budget rounding and placement. Keep content scoring separate from sequence/cache updates. |
| Qwen3-VL model adapter | Map image/video spans, original multimodal positions, DeepStack features and decoder-layer state. Support multiple images and timestamp-separated video groups explicitly. |
| Existing packed W4 loader/kernel | Preserve QAT sidecar/scales, module coverage and strict fused execution. Do not replace QAT with toolkit PTQ. |
| New runner/config namespace | Add method/placement/budget fields and a fresh protocol/output base. Preserve dataset, prompts, scores and completion/replay verification. |
| Shared train/eval selector implementation | Reuse selection semantics later during joint QAT; avoid two slightly different definitions of the same method. |

Prove r100/no-op parity first: same question identities, retained full sequence, close logits and matching greedy answers on an explicit test set. Then check nontrivial reduction, all text/answer labels, positions, DeepStack alignment, cache lengths, EOS/caps and positive packed calls. The saved layer-token telemetry currently describes the expected layer-0 path; for a new late-layer reducer, instrument **actual layer/cache shapes** so the report shows the full-token early layers and compact later layers.

Two implementation traps deserve explicit tests. Official hooks assume a single visual interval and specific cache/forward APIs; Qwen3-VL video has multiple timestamp-separated spans and a newer cache API. Also, FastV's prompt attention and DART's text pivots must use only the available prompt, never ground-truth answer tokens during teacher-forced training.

### Third: handle late-layer selection and chunked video prefill as a separate milestone

The current video loop feeds each 512-token chunk through **all 28 layers** before the next chunk. A selector that needs the final prompt token's attention over the whole video cannot simply run unchanged inside each chunk: the last token of an intermediate chunk is not the final question token. DART likewise needs the intended global pivot/candidate population, not an accidental chunk-local subset.

Start on images where the full prompt can be processed with straightforward layer placement. For video, design and validate a staged/layer-aware prefill path or another implementation that preserves the selected algorithm's global selection semantics. Work out how early-layer full-token caches and later-layer compact caches coexist, how generation advances positions, and which earlier computation can actually be skipped. A layer-0 DART-inspired feature selector or chunk-local approximation could be a separate ablation, but it would not reproduce the paper's fifth-layer method.

Avoid allocating an L×L attention matrix for a 100k-token prompt. If FastV requires attention that FlashAttention 2 does not return, either implement/verify the required last-query scoring carefully or report the backend/scoring overhead explicitly. Do not assume a method's nominal FLOP reduction becomes a speedup on our streamed runtime.

### Fourth: compare content-aware reduction at matched budgets and measured cost

For Qwen3-VL, screen r100/r075/r050/r025 on development examples before confirming a setting. Compare the selected method with uniform, merge and resolution; match actual retained counts where possible and also compare at measured latency/memory. For a fifth-layer method, its “retained count” applies after insertion: its total layer-token work is larger than our layer-0 control even at the same final K. Log that difference instead of presenting equal K as equal compute.

Keep the current image ceiling and declared adaptive video policy. Prioritize the seven image suites, TempCompass category checks, VideoMME short/medium, and a detail-reading task such as TextVQA. Existing Video-MMMU perception results give an additional stress test, while incomplete comprehension/adaptation remain separate pending cells. Do not select hyperparameters on those final benchmark outcomes and then call the same scores independent confirmation.

For speed, use a fresh explicit-ID manifest stratified by clip length, starting tokens, aspect ratio and cap activation. The first-three-question profiler and prefix-only smoke selection are insufficient for representative video timing. Run one evaluation at a time on a fixed device, warm up outside timing, alternate method order, recompute every sample, and report repeated natural-answer latency plus fixed-output latency. Include selector time, vision, prefill, decode, per-question input-to-answer latency and peak allocated/reserved memory. Label checkpoint/setup costs separately. This is the targeted measurement needed to fill the unsupported speedup cells in the current report.

### Fifth: only then test whether joint QAT recovers quality

Keep the completed unpruned W4 checkpoint as arm A. Evaluate A both unpruned and with the chosen selector: the latter is sequential composition. Train arm B with the same selector active from the **same BF16 SFT initialization**, matching the saved 32,768-example order, batch 128, 256 updates, seed, LoRA scope and learning rates. Compare A+selector with B+selector at the same inference setting. Keep QAT W8 as a quality reference.

Do not add extra training to the finished W4 checkpoint and call it equal-exposure joint training without a matched continuation control. Hard top-k selection has no ordinary gradient through the indices; QAT can adapt weights/scales/features to retained inputs without learning the discrete selection rule. Verify effective updates, finite losses, supervised-token preservation and packed-export parity before interpreting quality changes. Report GPU-hours as well as quality: equal update counts are not equal compute when tokens differ.

New GRACE/KD training remains optional. If later justified, align teacher/student supervised tokens and retained/merged visual identities explicitly; reduced token counts do not permit an unaligned CKA comparison. The immediate research question is narrower and useful: **does a content-aware selector improve the frozen QAT tradeoff, and does joint CE-only QAT improve it further?**

## 12. Evidence and how to refresh the calculations

The new [CPU-only audit script](reports/compile_completed_analysis.py) independently checks completion-artifact SHA-256 hashes, successful exits, validation payloads, full telemetry populations/order, stored layer-token consistency, positive fused calls, 196 packed modules, token/timing sums and fixed-profile means. It also verifies **32 pinned source files**, the original/recovery protocol link and the five reused completion markers. TempCompass/VideoMME scores, category scores and error swaps are independently recounted from saved samples; image and Video-MMMU primary scores use their saved validated scorer outputs. This is an artifact audit, not a rerun of model generation or an independent review of every scorer's implementation.

The outputs are [analysis JSON](reports/completed_analysis_20261116.json), [flat CSV](reports/completed_analysis_20261116.csv), and [generated tables](reports/completed_tables_20261116.md). They include exact token sums, source durations, per-group statistics and timer-source provenance. Rebuild them from the workspace root with:

```bash
python3 reproduction/GRACE/experiments/qat_int4_token_reduction_576_seed42/reports/compile_completed_analysis.py
```

This command only writes those three analysis files. It does not refresh this narrative automatically, run inference, change a completion marker or modify the pinned evaluator. If more jobs finish, treat the regenerated files as a new snapshot and update the dated interpretation. The older `compile_video_progress.py` contains November 9-specific narrative assumptions; do not use it to generate the conclusions in this follow-up unchanged.

The [plot script](reports/plot_completed_analysis.py) reads that analysis JSON and generates the PNG/PDF figure using Matplotlib. It performs no model work.
