from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scatter_max',
    description="maxpooling function for dynamic voxel",
    author='zilong Yuan',
    ext_modules=[
        CUDAExtension('scatter_max', [
            'cuda/scatter_max.cpp',
            'cuda/scatter_max_cuda.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })