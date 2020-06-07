from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name = 'PointCloudVoxel',
    description="PointCloud Voxel operation, support hard voxel and dynamic voxel",
    author='zilong Yuan',
    ext_modules=[cpp_extension.CppExtension('PointCloudVoxel',
                                             ['src/PointCloudVoxel.cpp'],
                                             undef_macros=['NDEBUG'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

