"""Group-wise weight-only LSQ QAT for Qwen3-VL.

Each eligible nn.Linear under model.language_model.layers.* is *wrapped*
(weight Parameter is shared, not copied) so DeepSpeed ZeRO-2/3 partitioning,
bf16 mixed precision, and HF Trainer all keep working unchanged. The vision
tower (model.visual), the multimodal merger (model.visual.merger), lm_head
and token embeddings are intentionally left in bf16.

Public API:
  - QuantizedLinear
  - QATLoraLinear
  - replace_linears_with_qat
  - replace_linears_with_qat_lora
  - collect_qat_param_groups
  - merge_lora_weights_inplace
  - materialize_quantized_state_dict
  - bake_qat_weights_inplace
  - strip_log_scale_from_checkpoint_dir
  - strip_lora_from_checkpoint_dir
  - strip_qat_lora_from_checkpoint_dir
"""

from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- core ops ----------

class _STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


# ---------- module ----------

class QuantizedLinear(nn.Module):
    """Wraps an nn.Linear and applies group-wise LSQ fake quantization on the
    weight at forward time. The original weight Parameter is *shared* (not
    copied), so the optimizer and DeepSpeed see the same parameter object.

    Weight is viewed flat as (num_groups, group_size); log_scale is a 1D fp32
    Parameter of length num_groups.
    """

    # MSE-optimal scale init: 20 candidate scales per group, sampled as
    # ratios of (max_abs / qmax). The optimal ratio for 4-bit usually lies
    # below 1.0 because outliers push max-init off the MSE optimum; we let
    # the lower bound dip to 0.3 to give the grid search room.
    _INIT_NUM_CANDIDATES: int = 20
    _INIT_SEARCH_LO: float = 0.3
    _INIT_SEARCH_HI: float = 1.2

    def __init__(self, lin: nn.Linear, bits: int = 4, group_size: int = 128):
        super().__init__()
        assert bits in (4, 8), f"QuantizedLinear supports bits in {{4,8}}, got {bits}"
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bits = bits
        self.qmin = -(1 << (bits - 1))         # -8  / -128
        self.qmax = (1 << (bits - 1)) - 1      #  7  /  127

        # Share weight + bias (no copy) — keeps DS-managed params managed.
        self.weight = lin.weight
        self.bias = lin.bias

        if group_size <= 0 or self.in_features % group_size != 0:
            self.group_size = self.in_features
        else:
            self.group_size = group_size
        self.num_groups = (self.out_features * self.in_features) // self.group_size

        self.log_scale = nn.Parameter(
            torch.zeros(self.num_groups, dtype=torch.float32)
        )
        self.register_buffer(
            "_scale_initialized",
            torch.zeros((), dtype=torch.bool),
            persistent=False,
        )
        # Avoid a CUDA-synchronizing ``Tensor.item()`` in every layer on every
        # forward. The tensor is retained for compatibility/debug visibility;
        # this Python flag is the hot-path readiness check.
        self._scale_ready = False
        # Plain Python flags are intentionally absent from state_dict. After a
        # checkpoint load, a static quantizer must bake the restored weight on
        # its first forward rather than trusting a stale cached state.
        self._quantization_static = False
        self._static_weight_baked = False

    @torch.no_grad()
    def init_scales_from_weight(self) -> None:
        """Per-group scale init via MSE-optimal grid search.

        For each group, evaluate `_INIT_NUM_CANDIDATES` scale candidates
        spanning `_INIT_SEARCH_LO .. _INIT_SEARCH_HI` (as a multiplier of
        max_abs / qmax) and pick the one minimizing
        ||W - dequant(quant(W, s))||² on that group. Compared to a fixed
        max- or 99th-percentile heuristic, this lets each group trade
        outlier clipping against grid-point density on its own merits,
        which matters most at 4-bit where the grid is coarse and the
        optimal trade-off is far from `max/qmax`. Runs once at init
        (and on the first forward for ZeRO-3 partitioned shards).
        """
        if self._scale_ready:
            return
        if bool(self._scale_initialized.item()):
            self._scale_ready = True
            return
        W = self.weight.detach()
        if W.numel() == 0:
            # ZeRO-3 partitioned shard — defer to first forward after DS init.
            return
        Wf = W.float().view(-1, self.group_size)            # (G, gs)
        num_groups = Wf.shape[0]

        max_abs = Wf.abs().amax(dim=-1).clamp(min=1e-8)     # (G,)
        ratios = torch.linspace(
            self._INIT_SEARCH_LO, self._INIT_SEARCH_HI,
            self._INIT_NUM_CANDIDATES,
            device=Wf.device, dtype=Wf.dtype,
        )                                                    # (C,)
        # Per-group candidate scales: (G, C)
        candidates = (max_abs / self.qmax).unsqueeze(-1) * ratios

        # Chunked search to cap peak memory at ~chunk * C * gs floats.
        # With chunk=4096, C=20, gs=128, fp32 this is ~40 MB / step.
        best_scale = torch.empty(num_groups, device=Wf.device, dtype=Wf.dtype)
        chunk = 4096
        for i in range(0, num_groups, chunk):
            Wc = Wf[i:i + chunk]                            # (g, gs)
            Sc = candidates[i:i + chunk]                    # (g, C)
            # Quantize each group under each candidate scale.
            wq = (Wc.unsqueeze(1) / Sc.unsqueeze(-1))       # (g, C, gs)
            wq = wq.round().clamp(self.qmin, self.qmax)
            wdq = wq * Sc.unsqueeze(-1)                     # (g, C, gs)
            mse = (wdq - Wc.unsqueeze(1)).pow(2).mean(dim=-1)  # (g, C)
            best_idx = mse.argmin(dim=-1)                   # (g,)
            best_scale[i:i + chunk] = Sc.gather(
                1, best_idx.unsqueeze(-1)
            ).squeeze(-1)

        best_scale = best_scale.clamp(min=1e-8)
        self.log_scale.data.copy_(torch.log(best_scale))
        self._scale_initialized.fill_(True)
        self._scale_ready = True

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int = 4, group_size: int = 128) -> "QuantizedLinear":
        q = cls(lin, bits=bits, group_size=group_size)
        q.init_scales_from_weight()
        return q

    def _quantize_weight(self) -> torch.Tensor:
        scale = torch.exp(self.log_scale.float()).unsqueeze(-1)   # (num_groups, 1)
        w = self.weight
        w_groups = w.view(-1, self.group_size).float()
        w_q = _ste_round(w_groups / scale).clamp(self.qmin, self.qmax)
        w_dq = (w_q * scale).view_as(w)
        return w_dq.to(w.dtype)

    def freeze_quantizer_(self) -> "QuantizedLinear":
        """Freeze weight/scale and enable an in-place fake-quant cache.

        With ordinary replicated parameters, the first forward bakes the
        initialized fake-quantized value into the already-frozen weight and
        later forwards use it directly. This avoids repeatedly quantizing
        large frozen MLP matrices. ZeRO-3 parameters are gathered only for a
        module invocation, so they safely fall back to dynamic fake quant.
        """
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.log_scale.requires_grad_(False)
        self._quantization_static = True
        self._static_weight_baked = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._scale_ready:
            self.init_scales_from_weight()
        if self._quantization_static and not hasattr(self.weight, "ds_id"):
            if not self._static_weight_baked:
                with torch.no_grad():
                    self.weight.copy_(self._quantize_weight())
                self._static_weight_baked = True
            return F.linear(x, self.weight, self.bias)
        w = self._quantize_weight()
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, group_size={self.group_size}"
        )


class QATLoraLinear(QuantizedLinear):
    """QAT linear with a trainable low-rank residual over a frozen base.

    Training evaluates ``fake_quant(base_weight) @ x + LoRA(x)``. This is
    the efficient QLoRA-style formulation: it avoids materializing a dense
    ``base_weight + B @ A`` tensor on every forward. ``log_scale`` remains
    trainable for these adapted layers, so the quantizer can follow the new
    effective transformation. At export, :func:`merge_lora_weights_inplace`
    folds ``B @ A`` into the base weight before the usual QAT bake.

    Non-target linears use :class:`QuantizedLinear` with both their base
    weight and initialized ``log_scale`` frozen.
    """

    def __init__(
        self,
        lin: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        r: int = 64,
        lora_alpha: float = 128.0,
        lora_dropout: float = 0.0,
    ):
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError(
                f"LoRA dropout must be in [0, 1), got {lora_dropout}"
            )
        super().__init__(lin, bits=bits, group_size=group_size)
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = self.lora_alpha / self.r
        self.lora_dropout = nn.Dropout(float(lora_dropout))

        factory_kwargs = {
            "device": self.weight.device,
            "dtype": self.weight.dtype,
        }
        self.lora_A = nn.Parameter(
            torch.empty((self.r, self.in_features), **factory_kwargs)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_features, self.r), **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA changes the layer through A/B; the pretrained dense weight and
        # bias never receive gradients. The layer's QAT scale is intentionally
        # trainable and is placed in its own optimizer group by the collector.
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.log_scale.requires_grad_(True)

    @classmethod
    def from_linear(
        cls,
        lin: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        r: int = 64,
        lora_alpha: float = 128.0,
        lora_dropout: float = 0.0,
    ) -> "QATLoraLinear":
        q = cls(
            lin,
            bits=bits,
            group_size=group_size,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        q.init_scales_from_weight()
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        lora_input = self.lora_dropout(x).to(self.lora_A.dtype)
        lora_hidden = F.linear(lora_input, self.lora_A)
        lora_output = F.linear(lora_hidden, self.lora_B)
        return base + (lora_output * self.lora_scaling).to(base.dtype)

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", lora_r={self.r}, lora_alpha={self.lora_alpha}, "
              f"lora_dropout={self.lora_dropout.p}"
        )


# ---------- replacement / utilities ----------

# Qwen3-VL submodule skips:
#   visual         — model.visual.* (vision transformer + patch embed)
#   merger         — model.visual.merger.* (multimodal projector)
#   lm_head        — output projection (stays bf16)
#   embed_tokens   — token embeddings (stays bf16)
# Everything else under language_model.layers.* (q/k/v/o_proj +
# gate/up/down_proj) gets wrapped.
DEFAULT_QAT_SKIP_KEYWORDS: Tuple[str, ...] = (
    "visual",
    "merger",
    "lm_head",
    "embed_tokens",
)


def _module_should_skip(name: str, skip_keywords: Iterable[str]) -> bool:
    return any(k in name for k in skip_keywords)


def replace_linears_with_qat(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    skip_keywords: Iterable[str] = DEFAULT_QAT_SKIP_KEYWORDS,
    verbose: bool = True,
) -> List[str]:
    """Replace eligible nn.Linear submodules with QuantizedLinear in-place."""
    replaced: List[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if _module_should_skip(full_name, skip_keywords):
                continue
            qlin = QuantizedLinear.from_linear(child, bits=bits, group_size=group_size)
            setattr(parent, child_name, qlin)
            replaced.append(full_name)

    if verbose:
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0
        if rank == 0:
            print(f"[QAT] Replaced {len(replaced)} Linear -> QuantizedLinear "
                  f"(bits={bits}, group_size={group_size}).")
    return replaced


def replace_linears_with_qat_lora(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    r: int = 64,
    lora_alpha: float = 128.0,
    lora_dropout: float = 0.0,
    skip_keywords: Iterable[str] = DEFAULT_QAT_SKIP_KEYWORDS,
    verbose: bool = True,
    method: str = "qat",
) -> Tuple[List[str], List[str]]:
    """Install QAT on eligible linears and LoRA on selected leaf modules.

    Every eligible linear is fake-quantized. Selected modules get trainable
    LoRA A/B parameters and a trainable QAT scale; all other modules get an
    initialized, frozen scale. Dense base weights stay frozen in both cases.

    Returns ``(quantized_names, lora_names)``.
    """
    targets = {name.strip() for name in target_modules if name.strip()}
    if not targets:
        raise ValueError("At least one LoRA target module name is required.")

    quantized: List[str] = []
    adapted: List[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if _module_should_skip(full_name, skip_keywords):
                continue

            if child_name in targets:
                replacement = QATLoraLinear.from_linear(
                    child,
                    bits=bits,
                    group_size=group_size,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                adapted.append(full_name)
            else:
                replacement = QuantizedLinear.from_linear(
                    child, bits=bits, group_size=group_size
                )
                replacement.freeze_quantizer_()

            setattr(parent, child_name, replacement)
            quantized.append(full_name)

    if not adapted:
        raise ValueError(
            "None of the requested LoRA target modules were found among the "
            f"eligible QAT linears: {sorted(targets)}"
        )

    if verbose:
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0
        if rank == 0:
            method_label = str(method).strip().upper()
            if not method_label:
                raise ValueError("method must be a non-empty name for verbose logging")
            print(
                f"[{method_label}-LoRA] Quantized {len(quantized)} linears; adapted "
                f"{len(adapted)} linears (targets={sorted(targets)}, r={r}, "
                f"alpha={lora_alpha}, bits={bits}, group_size={group_size})."
            )
    return quantized, adapted


def collect_qat_param_groups(
    model: nn.Module,
    base_lr: float,
    scale_lr: Optional[float] = None,
    weight_decay: float = 0.0,
) -> list:
    """Return optimizer param groups separating QAT log_scale params from others."""
    if scale_lr is None:
        scale_lr = base_lr
    scale_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".log_scale") or n.endswith("log_scale"):
            scale_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": scale_params, "lr": scale_lr, "weight_decay": 0.0},
    ]


@torch.no_grad()
def merge_lora_weights_inplace(
    model: nn.Module,
    is_zero3: bool = False,
    output_chunk_size: int = 1024,
) -> int:
    """Fold every QAT-LoRA residual into its frozen dense base weight.

    The output-row chunks cap temporary memory during ``B @ A``. Adapter
    parameters are zeroed after merging, making the operation idempotent for
    the remainder of the process. Under ZeRO-3, parameters are gathered and
    the rank-0 modification is redistributed on context exit.
    """
    if output_chunk_size <= 0:
        raise ValueError("output_chunk_size must be positive")

    def _merge(module: QATLoraLinear) -> None:
        for start in range(0, module.out_features, output_chunk_size):
            end = min(start + output_chunk_size, module.out_features)
            delta = torch.matmul(module.lora_B[start:end], module.lora_A)
            delta = delta.mul(module.lora_scaling).to(module.weight.dtype)
            module.weight.data[start:end].add_(delta)
        module.lora_A.zero_()
        module.lora_B.zero_()

    count = 0
    if is_zero3:
        import deepspeed
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = 0
        for _, module in model.named_modules():
            if not isinstance(module, QATLoraLinear):
                continue
            params = [module.weight, module.lora_A, module.lora_B]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if rank == 0:
                    _merge(module)
            count += 1
    else:
        for _, module in model.named_modules():
            if not isinstance(module, QATLoraLinear):
                continue
            _merge(module)
            count += 1
    return count


@torch.no_grad()
def materialize_quantized_state_dict(model: nn.Module) -> dict:
    """Produce a state dict with fake-quantized weights for downstream low-bit
    deployment. Includes log_scale for true-INT conversion."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            sd[name + ".weight"] = module._quantize_weight().detach().cpu()
            if module.bias is not None:
                sd[name + ".bias"] = module.bias.detach().cpu()
            sd[name + ".log_scale"] = module.log_scale.detach().cpu()
    return sd


def bake_qat_weights_inplace(model: nn.Module, is_zero3: bool = False) -> int:
    """Overwrite each QuantizedLinear's weight.data with its fake-quantized
    value, so a vanilla nn.Linear forward gives the same outputs as QAT.

    Under ZeRO-3, gather the partitioned weight + log_scale and write back on
    modifier_rank=0 so the change replicates to all shards.
    """
    count = 0
    if is_zero3:
        import deepspeed
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = 0
        for _, module in model.named_modules():
            if not isinstance(module, QuantizedLinear):
                continue
            params = [module.weight, module.log_scale]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if rank == 0:
                    wq = module._quantize_weight().detach()
                    module.weight.data.copy_(wq.to(module.weight.dtype))
            count += 1
    else:
        for _, module in model.named_modules():
            if not isinstance(module, QuantizedLinear):
                continue
            wq = module._quantize_weight().detach()
            module.weight.data.copy_(wq.to(module.weight.dtype))
            count += 1
    return count


def _is_weight_shard(path: str) -> bool:
    """Restrict log_scale stripping to actual weight files. HF checkpoints
    also drop training_args.bin / optimizer.pt / scheduler.pt / rng_state.pt
    in the same dir; those contain pickled non-tensor objects that torch.load
    with weights_only=True (PyTorch 2.6 default) refuses to deserialize."""
    base = os.path.basename(path)
    if base.endswith(".safetensors"):
        return base.startswith("model") or base.startswith("pytorch_model")
    if base.endswith(".bin"):
        return base.startswith("pytorch_model")
    return False


def _strip_keys_from_file(path: str, key_fragments: Tuple[str, ...]) -> int:
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file, save_file
        except ImportError:
            return 0
        sd = load_file(path)
        keep = {
            k: v for k, v in sd.items()
            if not any(fragment in k for fragment in key_fragments)
        }
        removed = len(sd) - len(keep)
        if removed:
            save_file(keep, path, metadata={"format": "pt"})
        return removed
    if path.endswith(".bin"):
        # weights_only=False: we trust our own checkpoint files.
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(sd, dict):
            return 0
        keep = {
            k: v for k, v in sd.items()
            if not any(fragment in k for fragment in key_fragments)
        }
        removed = len(sd) - len(keep)
        if removed:
            torch.save(keep, path)
        return removed
    return 0


def _strip_keys_from_checkpoint_dir(
    output_dir: str, key_fragments: Tuple[str, ...]
) -> int:
    """Strip matching auxiliary tensors from HF weight files and indexes."""
    import json
    import glob
    total = 0
    candidates = glob.glob(os.path.join(output_dir, "*.bin")) + \
                 glob.glob(os.path.join(output_dir, "*.safetensors"))
    for path in candidates:
        if not _is_weight_shard(path):
            continue
        total += _strip_keys_from_file(path, key_fragments)
    for idx_name in ("pytorch_model.bin.index.json", "model.safetensors.index.json"):
        idx_path = os.path.join(output_dir, idx_name)
        if not os.path.isfile(idx_path):
            continue
        with open(idx_path) as f:
            idx = json.load(f)
        wmap = idx.get("weight_map", {})
        new_wmap = {
            k: v for k, v in wmap.items()
            if not any(fragment in k for fragment in key_fragments)
        }
        if len(new_wmap) != len(wmap):
            idx["weight_map"] = new_wmap
            if "metadata" in idx and "total_size" in idx["metadata"]:
                idx["metadata"].pop("total_size", None)
            with open(idx_path, "w") as f:
                json.dump(idx, f, indent=2)
    return total


def strip_log_scale_from_checkpoint_dir(output_dir: str) -> int:
    """Remove QAT scale tensors from a merged, vanilla HF checkpoint."""
    return _strip_keys_from_checkpoint_dir(output_dir, ("log_scale",))


def strip_lora_from_checkpoint_dir(output_dir: str) -> int:
    """Remove already-merged native LoRA A/B tensors from an HF checkpoint."""
    return _strip_keys_from_checkpoint_dir(output_dir, ("lora_A", "lora_B"))


def strip_qat_lora_from_checkpoint_dir(output_dir: str) -> int:
    """Remove QAT and merged-LoRA auxiliaries in one pass over large shards."""
    return _strip_keys_from_checkpoint_dir(
        output_dir, ("log_scale", "lora_A", "lora_B")
    )
