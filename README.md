# PointCloudDynamicVoxel



## Why this project?

​	For 3D detection with point cloud, voxelization is a very common operation, and lots of 3D detection method use voxelization as first step to process point cloud, like VoxelNet, SECOND. But most of 3D object method use hard voxel(set max_points_in_voxel, and if the number of points in voxel exceed max_points_in_voxel, just throw the extra point).

​	In paper End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds(MVF), the author comes up with dynamic Voxel. Generally speaking, keep all the points in each voxel.Comparing to hard voxel, dynamic voxel uses less memory, and can keep all points.

​	I just implement dynamic voxel in the paper above according to my understand.
​       The reimplementation of MVF will comming soon, gives an example of how to use the project.



## How to use?

```
The project contains two parts, PointCloudVoxel and scatter, and both of them are written as a pytorch extension.

enviroment:
	Tested on ubuntu16.04, cuda10.0/cuda9.2, python3.6/python3.7,  pytorch-1.1/pytorch-1.4

install:
	cd PointCloudVoxel && python setup.py install
	cd scatter && python setup.py install

test:
	cd PointCloudVoxel/test && python test.py
	cd scatter/test && python test.py
```

