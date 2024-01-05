from setuptools import setup, Extension
from torch.utils import cpp_extension
import sys

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        name='quant_cuda', 
        sources=['quant_cuda.cpp', 'quant_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
