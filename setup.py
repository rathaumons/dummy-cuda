from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import nvidia_arch as na

if CUDA_HOME is None:
    raise RuntimeError("CUDA toolkit not found. Cannot build dummy-cuda.")

ext_modules = [
    CUDAExtension(
        name="dummy_cuda.dummy_ext",
        sources=["dummy_cuda/dummy.cpp", "dummy_cuda/dummy_kernel.cu",],
        extra_compile_args={
            "cxx": ["-O2", "-std=c++17", ],
            "nvcc": ["-O2", "-std=c++17", ] + na.make_gencode_flags("7.5;8.6", add_ptx=True)
        },
    )
]

setup(
    name="dummy-cuda",
    version="0.0.0",
    description="A minimal CUDA dummy extension for testing manylinux and PyTorch wheels.",
    author="Dummy Author",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=1.10",
        "nvidia-arch>=6.0.0",
    ],
    python_requires=">=3.9",
    zip_safe=False,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
