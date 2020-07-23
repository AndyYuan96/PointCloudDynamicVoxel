import numpy as np

import torch
from PointCloudVoxel import PointCloudVoxel
import time

def test():
    max_points_per_voxel = 35
    feature_size_x = (int)(50 / 0.2)
    feature_size_y = (int)(100 / 0.2)
    feature_size_z = 10
    min_x = -25.
    max_x = 25.
    min_y = 0
    max_y = 100
    min_z = -5
    max_z = 5
    max_bev_voxel_nums = 16000
    # used for front view, for bev, just keep the parameter below fixed
    rows = 1
    cols = 1
    min_theta = 0.0
    max_theta = 1.0
    min_phi = 0.0
    max_phi = 1.0

    voxelGenerator = PointCloudVoxel(max_points_per_voxel, feature_size_x, feature_size_y, feature_size_z, min_x, max_x, min_y, max_y, min_z, max_z, max_bev_voxel_nums, rows, cols,min_theta, max_theta, min_phi, max_phi)

    points = np.fromfile('pointcloud.bin',np.float32).reshape(-1,4)
    points = torch.from_numpy(points)
    print("origin point shape : ",points.shape)

    # hardVoxel TEST
    print("hardVoxel TEST")
    voxels = torch.zeros((max_bev_voxel_nums, 4 * max_points_per_voxel), dtype=torch.float32)
    voxel_coordinate = torch.zeros((max_bev_voxel_nums, 3), dtype=torch.int32)
    voxel_point_nums = torch.zeros((max_bev_voxel_nums), dtype=torch.int32)
    voxelGenerator.hardVoxelBEV(points, voxels, voxel_coordinate, voxel_point_nums)

    voxels = voxels.reshape(-1,4)
    mask = torch.zeros((1,4),dtype=torch.float32)
    voxel_mask = (voxels == mask).sum(dim=1) < 4
    voxels = voxels[voxel_mask].reshape(-1,4)  
    print("hard voxel remainding points : ", voxels.shape)
    print("hard voxel z sums : ", voxel_coordinate[:,0].sum())
    print("hard voxel x sums : ", voxel_coordinate[:,1].sum())
    print("hard voxel y sums : ", voxel_coordinate[:,2].sum())
    print()

    # dynamicVoxel TEST
    # dynamicVoxel is suitable for origin point cloud without filtering according to range

    # NOTE!!! max_point_nums must bigger than point nums of lidar, setting max_voxel_nums to max_point_nums/2 or max_point_nums/3 ..., just try
    print("dynamicVoxel TEST")
    max_point_nums = 100000
    max_voxel_nums = 50000
    bev_coordinate = torch.zeros((max_point_nums, 4), dtype=torch.float32)
    bev_local_coordinate =  torch.zeros((max_point_nums, 3), dtype=torch.float32)
    intensity = torch.zeros((max_point_nums), dtype=torch.float32)
    bev_mapping_pv = torch.zeros((max_point_nums), dtype=torch.int32)
    bev_mapping_vf = torch.zeros((max_voxel_nums, 3), dtype=torch.int32)
    start = time.time()
    voxelGenerator.dynamicVoxelBEV(points, bev_coordinate, bev_local_coordinate, intensity, bev_mapping_pv, bev_mapping_vf)
    valid_point_nums = voxelGenerator.getValidPointNums()
    valid_voxel_nums = voxelGenerator.getValidBEVVoxelNums()
    bev_coordinate = bev_coordinate[:valid_point_nums]
    intensity = intensity[:valid_point_nums]
    bev_mapping_pv = bev_mapping_pv[:valid_point_nums]
    bev_mapping_vf = bev_mapping_vf[:valid_voxel_nums]
    print("dynamicVoxelBEV consuming: ", time.time() - start)
    print("dynamic voxel remainding points : ", bev_coordinate.shape)
    print("dynamic voxel y sums : ", bev_mapping_vf[:,0].sum())
    print("dynamic voxel x sums : ", bev_mapping_vf[:,1].sum())
    print("dynamic voxel z sums : ", bev_mapping_vf[:,2].sum())
    print()

    # dynamicVoxelFaster TEST
    # dynamicVoxelFaster is suitable for already filterd point cloud according to range setting
    print("dynamicVoxelFaster TEST")
    # keep origin point cloud range unchanged, and just minus 0.01 for point filtering
    mask_x = ((points[:,0] >= min_x) + (points[:,0] < (max_x - 0.01))) == 2
    mask_y = (points[:,1] >= min_y) + (points[:,1] < (max_y - 0.01)) == 2
    mask_z = (points[:,2] >= min_z) + (points[:,2] < (max_z - 0.01)) == 2
    total_mask = (mask_x + mask_y + mask_z) == 3
    masked_points = points[total_mask]
    masked_points_local = torch.zeros_like(masked_points)

    print("point shape after masking : ", masked_points.shape)
    bev_mapping_pv_masked = torch.zeros((masked_points.shape[0]),dtype=torch.int32)
    bev_mapping_vf_masked = torch.zeros((max_voxel_nums, 3), dtype=torch.int32)
    start = time.time()
    voxelGenerator.dynamicVoxelBEVFaster(masked_points, masked_points_local, bev_mapping_pv_masked, bev_mapping_vf_masked)
    bev_mapping_vf_masked = bev_mapping_vf_masked[:voxelGenerator.getValidBEVVoxelNums(),:]
    print("dynamicVoxelBEVFaster consuming: ", time.time() - start)


if __name__ == '__main__':
    test()