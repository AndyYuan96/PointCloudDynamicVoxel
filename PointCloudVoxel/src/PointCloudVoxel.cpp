#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>

class PointCloudVoxel
{
public:
      PointCloudVoxel(int max_points_per_voxel, 
                      int feature_size_x, int feature_size_y, int feature_size_z,
                      float min_x, float max_x, float min_y, float max_y, float min_z, float max_z,
                      int max_bev_voxel_nums,
                      int rows=1, int cols=1, 
                      float min_theta=0, float max_theta=0, float min_phi=0, float max_phi=1);
      int getValidPointNums() { return valid_point_nums_; }
      int getValidBEVVoxelNums() { return valid_bev_voxel_nums_; }
      int getValidFVVoxelNums() { return valid_fv_voxel_nums_; }

      void hardVoxelBEV(torch::Tensor points_tensor,
                        torch::Tensor voxels_tensor,
                        torch::Tensor voxel_coordinate_tensor,
                        torch::Tensor voxel_point_nums_tensor);

      void dynamicVoxelBEV(torch::Tensor points_tensor,
                           torch::Tensor bev_coordinate_tensor,
                           torch::Tensor bev_local_coordinate_tensor,
                           torch::Tensor intensity_tensor,
                           torch::Tensor bev_mapping_pv_tensor,
                           torch::Tensor bev_mapping_vf_tensor);
      
      void dynamicVoxelBEVFaster(torch::Tensor points_tensor,
                                 torch::Tensor bev_mapping_pv_tensor,
                                 torch::Tensor bev_mapping_vf_tensor);
      
      
      void MVFVoxel(torch::Tensor points_tensor,
                    torch::Tensor bev_local_coordinate_tensor,
                    torch::Tensor fv_local_coordinate_tensor,
                    torch::Tensor intensity_tensor,
                    torch::Tensor bev_mapping_pv_tensor,
                    torch::Tensor bev_mapping_vf_tensor,
                    torch::Tensor fv_mapping_pv_tensor,
                    torch::Tensor fv_mapping_vf_tensor);

private:
      const int feature_size_x_;
      const int feature_size_y_;
      const int feature_size_z_;
      const int feature_size_xy_;
      const int feature_size_xyz_;
      const float min_x_;
      const float max_x_;
      const float min_y_;
      const float max_y_;
      const float min_z_;
      const float max_z_;

      const int rows_;
      const int cols_;
      const float min_theta_;
      const float max_theta_;
      const float min_phi_;
      const float max_phi_;

      const float voxel_x_step_;
      const float voxel_y_step_;
      const float voxel_z_step_;
      const float voxel_feature_step_;

      const float voxel_theta_step_;
      const float voxel_phi_step_;

      const int max_points_per_voxel_;
      const int max_bev_voxel_nums_;

      int valid_point_nums_;
      int valid_fv_voxel_nums_;
      int valid_bev_voxel_nums_;

      std::shared_ptr<int> bev_voxel_used_;
      std::shared_ptr<int> fv_voxel_used_;
};

PointCloudVoxel::PointCloudVoxel(int max_points_per_voxel,
                                int feature_size_x, int feature_size_y, int feature_size_z,
                                float min_x, float max_x, float min_y, float max_y, float min_z, float max_z,
                                int max_bev_voxel_nums, 

                                int rows, int cols, 
                                float min_theta, float max_theta, float min_phi, float max_phi):
                                max_points_per_voxel_(max_points_per_voxel), voxel_feature_step_(4 * max_points_per_voxel),
                                feature_size_x_(feature_size_x), feature_size_y_(feature_size_y), feature_size_z_(feature_size_z),
                                // must - 0.01, because in below, even points[i][0] == max_x_, the code doesn't continue, then cause error
                                min_x_(min_x), max_x_(max_x - 0.01), min_y_(min_y), max_y_(max_y - 0.01), min_z_(min_z), max_z_(max_z-0.01),
                                max_bev_voxel_nums_(max_bev_voxel_nums),

                                rows_(rows), cols_(cols), 
                                min_theta_(min_theta), max_theta_(max_theta), min_phi_(min_phi), max_phi_(max_phi),
                                feature_size_xy_(feature_size_x_ * feature_size_y_),feature_size_xyz_(feature_size_xy_ * feature_size_z),
                                voxel_x_step_((max_x_ + 0.01 - min_x_) / feature_size_x_), voxel_y_step_((max_y_ + 0.01 - min_y_) / feature_size_y_), voxel_z_step_((max_z_ + 0.01 - min_z_) / feature_size_z_),
                                voxel_theta_step_((max_theta_ - min_theta_) / rows_), voxel_phi_step_((max_phi_ - min_phi_) / cols_)
                     
  {
      bev_voxel_used_.reset(new int[feature_size_x_ * feature_size_y_ * feature_size_z_ + 1]);
      fv_voxel_used_.reset(new int[rows_ * cols_ + 1]);
      std::cout << ">>>>> Finish PointCloudVoxel Init" << std::endl;
  }

  void PointCloudVoxel::hardVoxelBEV(torch::Tensor points_tensor,
                                torch::Tensor voxels_tensor,
                                torch::Tensor voxel_coordinate_tensor,
                                torch::Tensor voxel_point_nums_tensor)
  {
    auto points = points_tensor.accessor<float,2>();
    auto voxels = voxels_tensor.accessor<float,2>();
    auto voxel_coordinate = voxel_coordinate_tensor.accessor<int,2>();
    auto voxel_point_nums = voxel_point_nums_tensor.accessor<int,1>();

    int bev_voxel_counts = 0;

    int point_nums = points.size(0);

    auto bev_voxel_used = bev_voxel_used_.get();
    std::fill(bev_voxel_used, bev_voxel_used + feature_size_x_ * feature_size_y_ * feature_size_z_ + 1, -1);

    for(int32_t i = 0; i < point_nums; ++i)
    {
        if(points[i][0] < min_x_ || points[i][0] >= max_x_ || 
            points[i][1] < min_y_ || points[i][1] >= max_y_ || 
            points[i][2] < min_z_ || points[i][2] >= max_z_)
        {
            continue;
        }

        int bev_x_index = (points[i][0] - min_x_) / voxel_x_step_;
        int bev_y_index = (points[i][1] - min_y_) / voxel_y_step_;
        int bev_z_index = (points[i][2] - min_z_) / voxel_z_step_;

        int bev_voxel_index = bev_z_index * feature_size_xy_ +  bev_y_index * feature_size_x_ + bev_x_index;

        assert(bev_voxel_index < feature_size_xyz_);
        
        // first point for the voxel
        if(bev_voxel_used[bev_voxel_index] == -1)
        {
          // voxel's count >= max_bev_voxel_nums, but don't break, just continue, points after may assign to already exist voxel
          if(bev_voxel_counts >= max_bev_voxel_nums_)
          {
              continue;
          }
          // update voxel_id
          bev_voxel_used[bev_voxel_index] = bev_voxel_counts;
          // update voxel position
          voxel_coordinate[bev_voxel_counts][0] = bev_y_index;
          voxel_coordinate[bev_voxel_counts][1] = bev_x_index;
          voxel_coordinate[bev_voxel_counts][2] = bev_z_index;
          // update voxel feature
          voxels[bev_voxel_counts][0] = points[i][0];
          voxels[bev_voxel_counts][1] = points[i][1];
          voxels[bev_voxel_counts][2] = points[i][2];
          voxels[bev_voxel_counts][3] = points[i][3];
          // update point nums in voxel
          voxel_point_nums[bev_voxel_counts] = 1;
          // update voxel's num
          bev_voxel_counts += 1;
        }
        else
        {
          int cur_voxel_point_nums = voxel_point_nums[bev_voxel_used[bev_voxel_index]];
          if(cur_voxel_point_nums < max_points_per_voxel_)
          {
            // update voxel feature
            int tmp_point_index = 4 * cur_voxel_point_nums;
            voxels[bev_voxel_used[bev_voxel_index]][tmp_point_index] = points[i][0];
            voxels[bev_voxel_used[bev_voxel_index]][tmp_point_index + 1] = points[i][1];
            voxels[bev_voxel_used[bev_voxel_index]][tmp_point_index + 2] = points[i][2];
            voxels[bev_voxel_used[bev_voxel_index]][tmp_point_index + 3] = points[i][3];
            // update point nums in voxel
            voxel_point_nums[bev_voxel_used[bev_voxel_index]] += 1;
          }
        }
    }
  }

  void PointCloudVoxel::dynamicVoxelBEV(torch::Tensor points_tensor,
                          torch::Tensor bev_coordinate_tensor,
                          torch::Tensor bev_local_coordinate_tensor,
                          torch::Tensor intensity_tensor,
                          torch::Tensor bev_mapping_pv_tensor,
                          torch::Tensor bev_mapping_vf_tensor)
  {
      auto points = points_tensor.accessor<float,2>();
      auto bev_coordinate = bev_coordinate_tensor.accessor<float,2>();
      auto bev_local_coordinate = bev_local_coordinate_tensor.accessor<float,2>();

      auto intensity = intensity_tensor.accessor<float,1>();
      auto bev_mapping_pv = bev_mapping_pv_tensor.accessor<int,1>();
      auto bev_mapping_vf = bev_mapping_vf_tensor.accessor<int,2>();

      int point_counts = 0;
      int bev_voxel_counts = 0;

      int point_nums = points.size(0);

      auto bev_voxel_used = bev_voxel_used_.get();

      std::fill(bev_voxel_used, bev_voxel_used + feature_size_x_ * feature_size_y_ * feature_size_z_ + 1, -1);

      for(int32_t i = 0; i < point_nums; ++i)
      {
          if(points[i][0] < min_x_ || points[i][0] >= max_x_ || 
             points[i][1] < min_y_ || points[i][1] >= max_y_ || 
             points[i][2] < min_z_ || points[i][2] >= max_z_)
          {
              continue;
          }

          int bev_x_index = (points[i][0] - min_x_) / voxel_x_step_;
          int bev_y_index = (points[i][1] - min_y_) / voxel_y_step_;
          int bev_z_index = (points[i][2] - min_z_) / voxel_z_step_;

          bev_coordinate[point_counts][0] = points[i][0];
          bev_coordinate[point_counts][1] = points[i][1];
          bev_coordinate[point_counts][2] = points[i][2];

          intensity[point_counts] = points[i][3];

          bev_local_coordinate[point_counts][0] = points[i][0] - min_x_ - voxel_x_step_ * bev_x_index;
          bev_local_coordinate[point_counts][1] = points[i][1] - min_y_ - voxel_y_step_ * bev_y_index;
          bev_local_coordinate[point_counts][2] = points[i][2] - min_z_ - voxel_z_step_ * bev_z_index;

          int bev_voxel_index = bev_z_index * feature_size_xy_ +  bev_y_index * feature_size_x_ + bev_x_index;

          assert((bev_voxel_index < feature_size_xyz_));

          if(bev_voxel_used[bev_voxel_index] == -1)
          {
            // change to (y,x) to satisfy dense
            bev_mapping_vf[bev_voxel_counts][0] = bev_y_index;
            bev_mapping_vf[bev_voxel_counts][1] = bev_x_index;
            bev_mapping_vf[bev_voxel_counts][2] = bev_z_index;

            bev_voxel_used[bev_voxel_index] = bev_voxel_counts;
            bev_mapping_pv[point_counts] = bev_voxel_counts;
            bev_voxel_counts += 1;
          }
          else
          {
            bev_mapping_pv[point_counts] = bev_voxel_used[bev_voxel_index];
          }
          ++point_counts;
      }
      valid_bev_voxel_nums_ = bev_voxel_counts;
      valid_point_nums_ = point_counts;
  }

  void PointCloudVoxel::dynamicVoxelBEVFaster(torch::Tensor points_tensor,
                                              torch::Tensor bev_mapping_pv_tensor,
                                              torch::Tensor bev_mapping_vf_tensor)
  {
      auto points = points_tensor.accessor<float,2>();

      auto bev_mapping_pv = bev_mapping_pv_tensor.accessor<int,1>();
      auto bev_mapping_vf = bev_mapping_vf_tensor.accessor<int,2>();

      int bev_voxel_counts = 0;

      int point_nums = points.size(0);

      auto bev_voxel_used = bev_voxel_used_.get();

      std::fill(bev_voxel_used, bev_voxel_used + feature_size_x_ * feature_size_y_ * feature_size_z_ + 1, -1);

      for(int32_t i = 0; i < point_nums; ++i)
      {
          int bev_x_index = (points[i][0] - min_x_) / voxel_x_step_;
          int bev_y_index = (points[i][1] - min_y_) / voxel_y_step_;
          int bev_z_index = (points[i][2] - min_z_) / voxel_z_step_;

          int bev_voxel_index = bev_z_index * feature_size_xy_ +  bev_y_index * feature_size_x_ + bev_x_index;

          assert((bev_voxel_index < feature_size_xyz_) || !(std::cout << ">>>>> Voxelization ERROR : Point : (x y z): " << points[i][0] << " " << points[i][1] << " " << points[i][2]
                                                            << " . Voxel index : (x y z): " << bev_x_index << " " << bev_y_index << " " << bev_z_index << ". " <<
                                                            "Solution : When using point range to filter point, minus 0.01 for max_x/max_y/max_z, don't change original point range, just minus 0.01 when filter point." << std::endl));

          if(bev_voxel_used[bev_voxel_index] == -1)
          {
            // change to (y,x) to satisfy dense
            bev_mapping_vf[bev_voxel_counts][0] = bev_y_index;
            bev_mapping_vf[bev_voxel_counts][1] = bev_x_index;
            bev_mapping_vf[bev_voxel_counts][2] = bev_z_index;

            bev_voxel_used[bev_voxel_index] = bev_voxel_counts;
            bev_mapping_pv[i] = bev_voxel_counts;
            bev_voxel_counts += 1;
          }
          else
          {
            bev_mapping_pv[i] = bev_voxel_used[bev_voxel_index];
          }
      }
      valid_bev_voxel_nums_ = bev_voxel_counts;
  }



  void PointCloudVoxel::MVFVoxel( torch::Tensor points_tensor,
                                  torch::Tensor bev_local_coordinate_tensor,
                                  torch::Tensor fv_local_coordinate_tensor,
                                  torch::Tensor intensity_tensor,
                                  torch::Tensor bev_mapping_pv_tensor,
                                  torch::Tensor bev_mapping_vf_tensor,
                                  torch::Tensor fv_mapping_pv_tensor,
                                  torch::Tensor fv_mapping_vf_tensor)
  {
      
      auto points = points_tensor.accessor<float,2>();
      auto bev_local_coordinate = bev_local_coordinate_tensor.accessor<float,2>();
      auto fv_local_coordinate = fv_local_coordinate_tensor.accessor<float,2>();
      auto intensity = intensity_tensor.accessor<float,1>();
      auto bev_mapping_pv = bev_mapping_pv_tensor.accessor<int,1>();
      auto bev_mapping_vf = bev_mapping_vf_tensor.accessor<int,2>();
      auto fv_mapping_pv = fv_mapping_pv_tensor.accessor<int,1>();
      auto fv_mapping_vf =fv_mapping_vf_tensor.accessor<int,2>();

      int point_counts = 0;
      int bev_voxel_counts = 0;
      int fv_voxel_counts = 0;

      int point_nums = points.size(0);

      auto bev_voxel_used = bev_voxel_used_.get();
      auto fv_voxel_used = fv_voxel_used_.get();

      std::fill(bev_voxel_used, bev_voxel_used + feature_size_x_ * feature_size_y_, -1);
      std::fill(fv_voxel_used, fv_voxel_used + rows_ * cols_, -1);

      for(int32_t i = 0; i < point_nums; ++i)
      {
          if(points[i][0] < min_x_ || points[i][0] >= max_x_ || 
             points[i][1] < min_y_ || points[i][1] >= max_y_ || 
             points[i][2] < min_z_ || points[i][2] >= max_z_)
          {
              continue;
          }

         
          float t1 = sqrtf(points[i][0] * points[i][0] + points[i][1] * points[i][1]);
          if(t1 == 0.0)
          {
            continue;
          }

          float dist = sqrtf(points[i][0] * points[i][0] + points[i][1] * points[i][1] + points[i][2] * points[i][2]);
          float theta = 180.0f * asinf(points[i][2] / dist) * M_1_PI;
          float phi = 180.0f *(asinf(points[i][1] / t1)) * M_1_PI;

          if(theta >= max_theta_ || theta < min_theta_ || phi >= max_phi_ || phi < min_phi_)
          {
            continue;
          }
          
          if(points[i][0] < 0.0)
          {
              if(phi > 0.0)
                  phi = 180.0 - phi;
              else
                  phi = -phi - 180.0;
              
              if(phi > 179.99)
              {
                continue;
              }
          }

          int bev_x_index = (points[i][0] - min_x_) / voxel_x_step_;
          int bev_y_index = (points[i][1] - min_y_) / voxel_y_step_;

          bev_local_coordinate[point_counts][0] = points[i][0] - min_x_ - voxel_x_step_ * bev_x_index;
          bev_local_coordinate[point_counts][1] = points[i][1] - min_y_ - voxel_y_step_ * bev_y_index;
          bev_local_coordinate[point_counts][2] = points[i][2] - min_z_;

          intensity[point_counts] = points[i][3];

          int bev_voxel_index = bev_y_index * feature_size_x_ + bev_x_index;

          if(bev_voxel_used[bev_voxel_index] == -1)
          {
            // change to (y,x) to satisfy dense
            bev_mapping_vf[bev_voxel_counts][0] = bev_y_index;
            bev_mapping_vf[bev_voxel_counts][1] = bev_x_index;

            bev_voxel_used[bev_voxel_index] = bev_voxel_counts;
            bev_mapping_pv[point_counts] = bev_voxel_counts;
            bev_voxel_counts += 1;
          }
          else
          {
            bev_mapping_pv[point_counts] = bev_voxel_used[bev_voxel_index];
          }

          int row_index = floor((theta - min_theta_) / voxel_theta_step_);
          int col_index = floor((phi - min_phi_) / voxel_phi_step_);

        
          fv_local_coordinate[point_counts][0] = theta - min_theta_ - row_index * voxel_theta_step_;
          fv_local_coordinate[point_counts][1] = phi - min_phi_ - col_index * voxel_phi_step_;
          fv_local_coordinate[point_counts][2] = dist;

          int fv_voxel_index = row_index * cols_ + col_index;
          
          if(fv_voxel_used[fv_voxel_index] == -1)
          {
            fv_mapping_vf[fv_voxel_counts][0] = row_index;
            fv_mapping_vf[fv_voxel_counts][1] = col_index;

            fv_voxel_used[fv_voxel_index] = fv_voxel_counts;
            fv_mapping_pv[point_counts] = fv_voxel_counts;

            fv_voxel_counts += 1;
          }
          else
          {
            fv_mapping_pv[point_counts] = fv_voxel_used[fv_voxel_index];
          }

          ++point_counts;
      }

      valid_bev_voxel_nums_ = bev_voxel_counts;
      valid_fv_voxel_nums_ = fv_voxel_counts;
      valid_point_nums_ = point_counts;
  }


//TORCH_EXTENSION_NAME 是在setup.py中你定义的那个名字
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<PointCloudVoxel>(m, "PointCloudVoxel")
  .def(pybind11::init<int, int, int, int, float, float, float, float, float, float, int, int, int, float, float, float, float>())
  .def("getValidPointNums", &PointCloudVoxel::getValidPointNums)
  .def("getValidBEVVoxelNums", &PointCloudVoxel::getValidBEVVoxelNums)
  .def("getValidFVVoxelNums", &PointCloudVoxel::getValidFVVoxelNums)
  .def("hardVoxelBEV", &PointCloudVoxel::hardVoxelBEV)
  .def("dynamicVoxelBEV", &PointCloudVoxel::dynamicVoxelBEV)
  .def("dynamicVoxelBEVFaster", &PointCloudVoxel::dynamicVoxelBEVFaster)
  .def("MVFVoxel", &PointCloudVoxel::MVFVoxel);
}