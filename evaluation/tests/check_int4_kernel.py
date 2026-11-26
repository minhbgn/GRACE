#!/usr/bin/env python3
"""Verify packed INT4 kernel against dequantized FP16 GEMM; CUDA execution (requires an idle GPU)."""
import argparse
import os
from pathlib import Path
import sys
p=argparse.ArgumentParser()
p.add_argument('--compile-only',action='store_true',help='Compile A100 SM80 PTX without CUDA execution')
p.add_argument('--handoff-dir', type=Path)
p.add_argument('--controller-pid', type=int)
args=p.parse_args()
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/grace-token-triton-cache')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen-vl-finetune'))
import torch
import triton
from qwenvl.quantize.qat_to_awq import pack_qat_linear,verify_pack
from qwenvl.quantize.awq_linear import dequantize_gemm
from evaluation.protocols.int4_gemm import int4_matmul,_packed_gemm

if args.compile_only:
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    src=ASTSource(_packed_gemm,signature={'X':'*fp16','Q':'*i32','Z':'*i32','S':'*fp16','Y':'*fp16','M':'i32'},
        constants={'N':160,'K':256,'G':128,'BM':32,'BN':64,'BK':32})
    kernel=triton.compile(src,target=GPUTarget('cuda',80,32),options={'num_warps':4})
    print('PASS offline A100 SM80 PTX compilation:',len(kernel.asm['ptx']),'bytes; CUDA execution not tested')
    raise SystemExit(0)

if args.handoff_dir is not None:
    from evaluation.protocols.gpu_reservation import await_gpu_handoff
    await_gpu_handoff(args.handoff_dir, args.controller_pid)

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise SystemExit('CUDA required for numerical kernel validation; CPU tests do not certify GPU execution.')
device='cuda:0'
max_error=0.
for k,n in ((128,64),(256,160)):
    scale=torch.full((n*k//128,),-3.)
    codes=torch.randint(-8,8,(n,k))
    weight=(codes*scale.exp().reshape(n,-1).repeat_interleave(128,dim=1))
    packed=pack_qat_linear(weight,scale,None,4,128).to(device)
    assert verify_pack(packed.cpu(),weight,scale,4,128)==0
    packed=packed.to(device)
    dense=dequantize_gemm(packed.qweight,packed.qzeros,packed.scales,4,128).half()
    for m in (1,33,1025):
        x=torch.randn(m,k,device=device,dtype=torch.float16)
        actual=int4_matmul(x,packed)
        expected=(x.float()@dense.float()).half()
        torch.testing.assert_close(actual,expected,atol=.02,rtol=.01)
        max_error=max(max_error,float((actual-expected).abs().max()))
print('PASS', 'real CUDA packed INT4', 'max_abs_error=',max_error)
