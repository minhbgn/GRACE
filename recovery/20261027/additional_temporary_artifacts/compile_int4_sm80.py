import os,sys
os.environ['TRITON_CACHE_DIR']='/tmp/grace-token-triton-cache'
sys.path.insert(0,'/raid/lvchengtao/Minh/Distillation/reproduction/GRACE')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from evaluation.protocols.int4_gemm import _packed_gemm
src=ASTSource(_packed_gemm,signature={'X':'*fp16','Q':'*i32','Z':'*i32','S':'*fp16','Y':'*fp16'},
    constants={'M':33,'N':136,'K':256,'G':128,'BM':32,'BN':64,'BK':32})
kernel=triton.compile(src,target=GPUTarget('cuda',80,32),options={'num_warps':4})
print('PASS offline A100 SM80 compilation:',kernel.metadata.name)
print('PTX bytes:',len(kernel.asm['ptx']))
