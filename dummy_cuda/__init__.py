from .dummy_ext import add_one_cuda

def add_one(x):
    if not x.is_cuda:
        raise ValueError("Input tensor must be CUDA")
    return add_one_cuda(x)
