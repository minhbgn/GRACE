---
license: mit
language:
- en
- zh

base_model:
- Qwen/Qwen3-VL-2B-Instruct
base_model_relation: finetune

pipeline_tag: image-text-to-text
library_name: transformers

datasets:
- Lin-Chen/ShareGPT4V

metrics:
- accuracy

tags:
- qwen3-vl
- vision-language-model
- multimodal
- bf16
- full-precision
- grace
- knowledge-distillation
- llava-eval
- icml-2026
---

# Qwen3-VL-2B-GRACE-BF16

This repository contains a **full-precision BF16 checkpoint** of Qwen3-VL-2B trained using the **GRACE framework**.

> **Start here:** Upload an image and try the GRACE 2B student in the
> [GRACE-VLM Space](https://huggingface.co/spaces/ForeverBlue/GRACE-VLM). For
> genuine packed INT4 deployment, use the primary
> [Qwen3-VL-2B-GRACE-W4G128-AWQ](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ)
> checkpoint and its copy-ready GRACE loader command.

This model is associated with our ICML 2026 paper:

**[Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](https://arxiv.org/abs/2601.22709)**  
Yanlong Chen, Amirhossein Habibian, Luca Benini, Yawei Li  
Accepted to the International Conference on Machine Learning (ICML 2026)

- **Paper:** https://arxiv.org/abs/2601.22709
- **DOI:** https://doi.org/10.48550/arXiv.2601.22709
- **Code:** https://github.com/ForeverBlue816/GRACE

---

## Model Details

- **Base model:** Qwen/Qwen3-VL-2B-Instruct
- **Training framework:** GRACE
- **Precision:** BF16 full precision
- **Training data:** ShareGPT4V
- **Evaluation protocol:** LLaVA-style multimodal evaluation
- **Repository:** ForeverBlue/Qwen3-VL-2B-GRACE-BF16

---

## 📊 Results

Comparison on 7 VLM benchmarks. The 8B model is the distillation **teacher**
(reference upper bound); all GRACE-Qwen3 variants are **2B** students. Best result
among the 2B Qwen3-VL models is in **bold**.

We release GRACE on Qwen3-VL here because it is the most current backbone and
gives a fairer, up-to-date point of comparison, with the vanilla
Qwen3-VL-2B-Instruct as the baseline. The paper itself reports GRACE on
LLaVA-1.5 and Qwen2-VL; we additionally release the LLaVA-1.5 W4G128 INT4
checkpoint from the paper in the model zoo below.

| Model | Params | Precision | HallB | MMBench | ScienceQA | AI2D | MMMU | SEED | MMStar | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-8B *(teacher, ref.)* | 8B | BF16 | 61.1 | 84.5 | 85.0 | 85.7 | 69.6 | 77.5 | 70.9 | 76.3 |
| Qwen3-VL-2B *(baseline)*      | 2B | BF16 | 51.4 | 78.4 | 81.4 | 76.9 | 53.4 | 71.2 | 58.3 | 67.3 |
| **Qwen3-VL-2B-GRACE**         | 2B | BF16 | **66.9** | **86.4** | **86.2** | **81.3** | **72.1** | **76.7** | **67.3** | **76.7** |
| Qwen3-VL-2B-GRACE (W8G128)    | 2B | INT8 | 66.1 | 85.5 | 85.3 | 80.4 | 71.3 | 75.9 | 66.5 | 75.9 |
| Qwen3-VL-2B-GRACE (W4G128)    | 2B | INT4 | 65.4 | 84.6 | 84.3 | 79.5 | 70.5 | 75.1 | 65.8 | 75.0 |

> GRACE lifts the Qwen3-VL-2B baseline by **+9.4 avg** and matches or slightly
> exceeds the 8B teacher on average (76.7 vs. 76.3) at roughly 1/4 the
> parameters. The W4G128 INT4 model retains **98%** of the BF16 average.

---

<a id="model-zoo"></a>
## 🤗 Model Zoo

| Model | Backbone | Bits | Group | Checkpoint description | HF Hub |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B-GRACE-BF16 | Qwen3-VL-2B | bf16 | — | Full-precision GRACE checkpoint; used as the student initialization for the W8/W4 Qwen3-VL runs. | [ForeverBlue/Qwen3-VL-2B-GRACE-BF16](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-BF16) |
| Qwen3-VL-2B-GRACE-W8G128 | Qwen3-VL-2B | int8 | 128 | INT8 QAT checkpoint with group size 128; high-retention quantized Qwen3-VL student. | [ForeverBlue/Qwen3-VL-2B-GRACE-W8G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W8G128) |
| Qwen3-VL-2B-GRACE-W4G128 | Qwen3-VL-2B | int4 | 128 | INT4 QAT checkpoint with group size 128; compact Qwen3-VL release retaining about 98% of the BF16 average. | [ForeverBlue/Qwen3-VL-2B-GRACE-W4G128](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128) |
| LLaVA-1.5-7B-GRACE-W4G128 | LLaVA-1.5-7B | int4 | 128 | INT4 QAT checkpoint from the GRACE paper with learned scales; released for reproducing the LLaVA-1.5 experiments. | [ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128](https://huggingface.co/ForeverBlue/LLaVA-1.5-7B-GRACE-W4G128) |

The BF16 Qwen3-VL checkpoint is the full-precision GRACE student used as the
initial student weights for the W8 and W4 Qwen3-VL runs. The LLaVA-1.5 W4G128
checkpoint corresponds to the paper setting and includes GRACE-specific QAT
quantized weights for reproducing the INT4 LLaVA experiments.

---

## Intended Use

This model is intended for research on:

- Efficient vision-language models
- Knowledge distillation for VLMs
- Multimodal alignment
- Full-precision GRACE training
- BF16 baseline / teacher-student comparison studies

---

## Training Details

This checkpoint is a **full-precision BF16 model trained under the GRACE framework**.

Configuration:

- **Precision:** BF16
- **Training method:** GRACE
- **Backbone:** Qwen3-VL-2B-Instruct
- **Dataset:** ShareGPT4V
- **Evaluation:** LLaVA-style multimodal benchmarks

Unlike the QAT releases, this model **does not use weight quantization**.

---

## Files

- `model.safetensors` / `model-*.safetensors`
- `config.json`
- `generation_config.json`
- tokenizer files
- processor files

---

## Loading

```python
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
import torch

repo_id = "ForeverBlue/Qwen3-VL-2B-GRACE-BF16"

processor = AutoProcessor.from_pretrained(
    repo_id,
    trust_remote_code=True
)

model = AutoModelForImageTextToText.from_pretrained(
    repo_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

---

## Important Notes

This is the full-precision BF16 GRACE checkpoint. It does not include INT8 or INT4
QAT weight compression. For quantized versions, please refer to the W8G128 and
W4G128 checkpoints listed in the Model Zoo.

The standard `from_pretrained` call should load this BF16 checkpoint directly in
a Qwen3-VL-compatible Transformers environment. For reproducing the GRACE
training or distillation pipeline, please refer to the official code repository:

https://github.com/ForeverBlue816/GRACE

---

## Limitations

- This model is released for research purposes.
- Performance may vary depending on the evaluation codebase, preprocessing, generation parameters, and multimodal benchmark implementation.
- Users should follow the license and usage restrictions of the original Qwen3-VL-2B-Instruct base model.
- This checkpoint is not optimized for low-bit inference; use the W8G128 or W4G128 release for quantized deployment studies.

---

## Citation

If you use this model, please cite:

```bibtex
@article{chen2026gated,
  title={Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs},
  author={Chen, Yanlong and Habibian, Amirhossein and Benini, Luca and Li, Yawei},
  journal={arXiv preprint arXiv:2601.22709},
  year={2026}
}
```
