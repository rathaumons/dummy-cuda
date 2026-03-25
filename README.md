# dummy-cuda

A minimal Python package with a native CUDA extension for testing manylinux wheel and CUDA builds.

## Features

- Single CUDA kernel (adds 1 to a CUDA tensor).
- Works with PyTorch 1 or later.
- Designed for CUDA 11 or later.

## Usage

```python
import torch
import dummy_cuda

x = torch.arange(10, dtype=torch.float32, device='cuda')
print(f"x before add_one() : {x}")  
# tensor([0., 1., 2., 3., ..., 9.], device='cuda:0')

dummy_cuda.add_one(x)
print(f"x after add_one()  : {x}")  
# Expect tensor([1., 2., 3., ..., 10.], device='cuda:0')
```
