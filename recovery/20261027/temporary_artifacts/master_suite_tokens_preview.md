# Master image and video suite input-token table

Updated 2026-10-25 16:11:16 SGT.

**Means are tokens per question over the whole declared suite**, including text-only image-suite questions with zero visual tokens and repeated media used by different questions. **Text means input text**, including question/options, instructions, chat/vision markers, and video timestamps. Generated answers are excluded.

| Suite | Modality | Questions | Mean visual input tokens | Mean text input tokens |
| --- | --- | ---: | ---: | ---: |
| ScienceQA | image | 4,241 | 76.39 | 79.53 |
| AI2D | image | 3,088 | 272.79 | 53.30 |
| MMMU | image | 900 | 278.64 | 107.40 |
| MMStar | image | 1,500 | 190.13 | 77.41 |
| SEEDBench image | image | 14,233 | 546.08 | 55.01 |
| HallusionBench | image | 1,129 | 336.72 | 41.48 |
| MMBench | image | 4,329 | 150.87 | 75.06 |
| TempCompass | video | 1,580 | 489.12 | 153.13 |
| VideoMME short | video | 900 | 2,606.40 | 781.76 |
| VideoMME medium | video | 900 | 16,656.11 | 5,128.58 |
| Video-MMMU Perception | video | 300 | 16,939.89 | 5,186.78 |
| Video-MMMU Comprehension | video | 300 | 16,939.89 | 5,190.00 |
| Video-MMMU Adaptation | video | 300 | 16,939.89 | 5,162.19 |

**Excluded:** VideoMME long (900 questions). Short and medium are separate 900-question suites above.

Image reference: QAT W4 native processor, 4,096–589,824 pixels, **576 merged visual tokens maximum per image**. Multi-image questions can exceed 576. The whole-population visual audit is reused; full text inputs are reconstructed on CPU with the real suite prompt builders and native chat template. MMBench counts all 4,329 evaluation rotations, not 1,164 base questions.

Video reference: complete Vanilla 2B runs, **2 fps / 256 px longest side**, without frame-count clipping. All six included video populations already have complete reference coverage: **no extrapolation from missing questions is needed**. These counts can be used as projections for other models under the same input policy.

The values are reference input profiles, not universal model-independent constants. Image native processor budgets can differ across checkpoints; different tokenizers/chat templates can change text input. Generated answer length is model-dependent and is not projected here.

Reconstruction and aggregation only; no model weights loaded and no inference run.

Source image audit: `inference_image_visual_tokens.json`. Source video audit: `video_token_rates_20261023.json`. The companion JSON records reference policies, source checksums, completion provenance and cross-model checks.
