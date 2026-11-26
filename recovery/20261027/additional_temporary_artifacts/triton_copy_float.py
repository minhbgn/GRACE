import os
os.environ['TRITON_INTERPRET']='1'
import torch,triton
import triton.language as tl
@triton.jit
def copy(X,Y):
 i=tl.arange(0,16)
 v=tl.load(X+i)
 tl.device_print('raw',v)
 tl.store(Y+i,v)
x=torch.ones(16,dtype=torch.float32)*.05
y=torch.empty_like(x)
print('x',x)
copy[(1,)](x,y)
print('y',y)
