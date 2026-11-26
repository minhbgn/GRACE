#!/usr/bin/env python3
"""Verify packed INT4 kernel against dequantized FP16 GEMM; explicit CUDA or CPU interpreter."""
import argparse
import os
from pathlib import Path
import sys
p=argparse.ArgumentParser()
p.add_argument('--interpreter',action='store_true',help='CPU Triton interpreter; not evidence of CUDA execution')
args=p.parse_args()
if args.interpreter:
    os.environ['TRITON_INTERPRET']='1'
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/grace-token-triton-cache')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen-vl-finetune'))
import torch
import triton
from qwenvl.quantize.qat_to_awq import pack_qat_linear,verify_pack
from qwenvl.quantize.awq_linear import dequantize_gemm
from evaluation.protocols.int4_gemm import int4_matmul,_packed_gemm

torch.manual_seed(42)
device='cpu' if args.interpreter else 'cuda:0'
max_error=0.
for k,n in ((128,64),(256,136)):
    scale=torch.full((n*k//128,),-3.)
    codes=torch.randint(-8,8,(n,k))
    weight=(codes*scale.exp().reshape(n,-1).repeat_interleave(128,dim=1))
    packed=pack_qat_linear(weight,scale,None,4,128).to(device)
    assert verify_pack(packed.cpu(),weight,scale,4,128)==0
    packed=packed.to(device)
    dense=dequantize_gemm(packed.qweight,packed.qzeros,packed.scales,4,128).half()
    for m in (1,33,1025):
        x=torch.randn(m,k,device=device,dtype=torch.float16)
        if args.interpreter:
            actual=torch.empty((m,n),dtype=torch.float16)
            _packed_gemm[(triton.cdiv(m,32),triton.cdiv(n,64))](x,packed.qweight,packed.qzeros,
                packed.scales,actual,m,n,k,128,32,64,32,num_warps=4)
        else:
            actual=int4_matmul(x,packed)
        expected=(x.float()@dense.float()).half()
        torch.testing.assert_close(actual,expected,atol=.02,rtol=.01)
        max_error=max(max_error,float((actual-expected).abs().max()))
print('PASS', 'CPU interpreter' if args.interpreter else 'real CUDA packed INT4', 'max_abs_error=',max_error)
