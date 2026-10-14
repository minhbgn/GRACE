---
license: mit
language:
- en
- zh

base_model:
- Qwen/Qwen3-VL-2B-Instruct
base_model_relation: quantized

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
- quantization-aware-training
- qat
- int8
- w8g128
- grace
- llava-eval
- icml-2026
---

# Qwen3-VL-2B-GRACE-W8G128

This repository contains a **GRACE-trained Qwen3-VL-2B checkpoint** using **quantization-aware training (QAT)** with **W8G128 group-wise INT8 quantization**.

> This W8G128 repository is a QAT research checkpoint. The primary deployment
> entry point is the real packed
> [Qwen3-VL-2B-GRACE-W4G128-AWQ](https://huggingface.co/ForeverBlue/Qwen3-VL-2B-GRACE-W4G128-AWQ)
> build; try the BF16 student in the [GRACE-VLM Space](https://huggingface.co/spaces/ForeverBlue/GRACE-VLM)
> presents the quality comparison and copy-ready command.

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
- **Method:** GRACE
- **Quantization:** W8G128 group-wise INT8 QAT
- **Training data:** ShareGPT4V
- **Training / evaluation protocol:** LLaVA-style multimodal evaluation
- **Library:** Hugging Face Transformers
- **Repository:** ForeverBlue/Qwen3-VL-2B-GRACE-W8G128

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
> parameters. The W8G128 INT8 model retains **99%** of the BF16 average.

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

This model is intended for **research purposes**, including:

- Efficient vision-language models
- Quantization-aware training
- Low-bit multimodal model deployment
- Knowledge distillation for VLM compression
- Multimodal model efficiency studies

---

## Out-of-Scope Use

This checkpoint is **not intended** for:

- Safety-critical deployment
- Medical / legal / financial decision-making
- Production systems requiring reliability guarantees

Like other VLMs, the model may generate hallucinated, biased, or incorrect outputs.

---

## Training Data

The model was trained using **ShareGPT4V multimodal instruction data** under a LLaVA-style multimodal fine-tuning pipeline.

Dataset:

- `Lin-Chen/ShareGPT4V`

---

## Quantization Details

This checkpoint uses **quantization-aware training (QAT)** with **group-wise W8G128 quantization**.

Configuration:

- **Weight precision:** INT8
- **Group size:** 128
- **Quantization scheme:** Group-wise QAT
- **Method:** GRACE
- **Backbone:** Qwen3-VL-2B-Instruct

Depending on the inference backend, specialized quantized kernels or custom loading logic may be required to obtain real INT8 deployment benefits.

---

## Repository Files

This repository may contain:

- `model.safetensors` / `model-*.safetensors` — model weights
- `qat_quantized_weights.bin` — QAT quantized weight artifact
- `config.json` — model configuration
- `generation_config.json` — generation configuration
- tokenizer files
- processor / preprocessing configuration files

---

## Loading

Please use a **Qwen3-VL-compatible Transformers environment** or the official Qwen3-VL codebase.

```python
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText

repo_id = "ForeverBlue/Qwen3-VL-2B-GRACE-W8G128"

processor = AutoProcessor.from_pretrained(
    repo_id,
    trust_remote_code=True
)

model = AutoModelForImageTextToText.from_pretrained(
    repo_id,
    trust_remote_code=True,
    device_map="auto"
)
```

Recommended:

- recent `transformers` version
- Qwen3-VL compatible environment
- CUDA GPU inference backend for large-scale evaluation

---

## Evaluation

The checkpoint follows a **LLaVA-style multimodal evaluation protocol**.

Representative evaluation may include benchmarks such as:

- HallusionBench
- MMBench
- ScienceQA
- AI2D
- MMMU
- SEED-Bench
- MMStar

Please refer to the associated GRACE paper and the results table above for detailed evaluation settings and results.

---

## Important Notes

This checkpoint includes QAT-specific quantized weights in `qat_quantized_weights.bin`. Depending on the inference codebase, additional GRACE-specific quantization-aware loading logic may be required.

The standard `from_pretrained` call may load the model configuration and checkpoint files, but fully reproducing the intended INT8 QAT behavior may require the GRACE repository:

https://github.com/ForeverBlue816/GRACE

---

## Limitations

- This model is released for research purposes.
- The quantized checkpoint may require custom loading logic for QAT-specific weights.
- Performance may vary depending on the evaluation codebase, preprocessing, generation parameters, and multimodal benchmark implementation.
- Users should follow the license and usage restrictions of the original Qwen3-VL-2B-Instruct base model.
- Specialized kernels or custom loading code may be required to realize practical INT8 speed or memory benefits.

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

Please also cite the original Qwen3-VL work when using this model.

---

## License

Released under the **MIT license**.

Users should additionally comply with:

- Qwen3-VL base model license
- ShareGPT4V dataset terms
- applicable downstream usage restrictions
