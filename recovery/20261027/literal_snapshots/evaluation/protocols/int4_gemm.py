"""Strict packed AWQ-layout W4A16 GEMM. Unpack inside a Triton tile, never a dense weight matrix.

Weight storage is INT4; tensor-core multiplication uses FP16 with FP32 accumulation.
This is a correctness-first backend, not a claim of optimal AWQ/Marlin performance.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _packed_gemm(X, Q, Z, S, Y, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
                 G:tl.constexpr, BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    # Inverse of AWQ_ORDER=[0,2,4,6,1,3,5,7].
    shifts=((cols%8)//2+(cols%2)*4)*4
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(tl.cdiv(K,BK)):
        ks=block*BK+kk
        x=tl.load(X+rows[:,None]*K+ks[None,:],(rows[:,None]<M)&(ks[None,:]<K),other=0)
        codes=tl.load(Q+ks[:,None]*(N//8)+(cols[None,:]//8),
                      (ks[:,None]<K)&(cols[None,:]<N),other=0)
        zeros=tl.load(Z+(ks[:,None]//G)*(N//8)+(cols[None,:]//8),
                      (ks[:,None]<K)&(cols[None,:]<N),other=0)
        scales=tl.load(S+(ks[:,None]//G)*N+cols[None,:],
                       (ks[:,None]<K)&(cols[None,:]<N),other=0)
        w=((((codes>>shifts[None,:])&15).to(tl.float32)-
            ((zeros>>shifts[None,:])&15).to(tl.float32))*scales).to(tl.float16)
        acc+=tl.dot(x,w)
    tl.store(Y+rows[:,None]*N+cols[None,:],acc.to(tl.float16),
             (rows[:,None]<M)&(cols[None,:]<N))


def int4_matmul(x, module):
    if not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous():
        raise ValueError('Strict INT4 GEMM requires contiguous CUDA FP16 activations')
    m,k=x.shape; n=module.out_features
    if k!=module.in_features or n%8 or k%module.group_size:
        raise ValueError('Packed shape/group mismatch')
    out=torch.empty((m,n),device=x.device,dtype=torch.float16)
    _packed_gemm[(triton.cdiv(m,32),triton.cdiv(n,64))](x,module.qweight,module.qzeros,
        module.scales,out,m,n,k,module.group_size,32,64,32,num_warps=4)
    return out
