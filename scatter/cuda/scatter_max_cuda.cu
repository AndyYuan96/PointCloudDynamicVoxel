#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void atomicMaxFloat (float * addr, float value) {
  float old;
  if(value == 0)
  {
    value = 0.0;
    __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
  }
  else if(value > 0)
  {
    __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
  }
  else
  {
    __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
  }       
}

__global__ void scatter_max_cuda_kernel(
  const float* __restrict__ input,
  const long* __restrict__ input_index,
  float* __restrict__ output,
  int64_t cols, int64_t numel)
{

  const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < numel)
  {
    const int32_t row = index / cols;
    const int32_t col = index - row * cols;

    const int32_t row_output = input_index[row]; 
    atomicMaxFloat(&output[row_output * cols + col], input[index]);
  }
}

__global__ void get_index_cuda_kernel(
  const float* __restrict__ input,
  const long* __restrict__ input_index,
  const float* __restrict__ output,
  long* __restrict__ output_index,
  int64_t cols, int64_t numel)
{
  const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < numel)
  {
    const int32_t row = index / cols;
    const int32_t col = index - row * cols;
    
    const int32_t row_output = input_index[row];  
    if(output[row_output * cols + col] == input[index])
    {
      output_index[row_output * cols + col] = row;
    }
  }
}

// input float32 (N,C)
// input_index int64 (N) : for one line, it has C features, those C feature have same voxel_id , so just pass 1 dimension is ok
// output float32 (M,C)
// output_index int64 (M,C) : for one line, it has C features, those C feature(maxpooling's result) may come from different point, so we should add C dimension

void scatter_max_cuda(torch::Tensor input, torch::Tensor input_index,
                      torch::Tensor output, torch::Tensor output_index,
                      bool train)
{
  cudaSetDevice(input.get_device());

  int32_t threads = 1024;
  int64_t blocks = (input.numel() + threads - 1) / threads;

  scatter_max_cuda_kernel<<<blocks, threads>>>(
    input.data<float>(),
    input_index.data<long>(),
    output.data<float>(),
    input.size(1),
    input.numel());

  if(train)
  {
    get_index_cuda_kernel<<<blocks, threads>>>(
    input.data<float>(),
    input_index.data<long>(),
    output.data<float>(),
    output_index.data<long>(),
    input.size(1),
    input.numel());
  }
}


__global__ void scatter_add_cuda_kernel(
  const float* __restrict__ input,
  const long* __restrict__ input_index,
  float* __restrict__ output,
  float* __restrict__ counts,
  int64_t cols, int64_t numel)
{

  const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < numel)
  {
    const int32_t row = index / cols;
    const int32_t col = index - row * cols;

    const int32_t row_output = input_index[row]; 
    atomicAdd(&output[row_output * cols + col], input[index]);
    if(col == 0)
    {
      atomicAdd(&counts[row_output], 1.0);
    }
  }
}

__global__ void scatter_div_cuda_kernel(
  float* __restrict__ input,
  const float* __restrict__ counts,
  int64_t cols, int64_t numel)
{

  const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < numel)
  {
    const int32_t row = index / cols;    
    input[index] /= counts[row];
  }
}

__global__ void scatter_back_cuda_kernel(
  float* __restrict__ input_means,
  const float* __restrict__ output,
  const long* __restrict__ input_index,
  int64_t cols,
  int64_t numel
)
{
  const int32_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < numel)
  {
    const int32_t row = index / cols;
    const int32_t col = index - row * cols;

    input_means[index] = output[input_index[row] * cols + col];
  }
}



void scatter_mean_cuda(torch::Tensor input, torch::Tensor input_index,
                       torch::Tensor output, torch::Tensor counts, torch::Tensor input_means)
{
  cudaSetDevice(input.get_device());
  int32_t threads = 1024;
  int64_t blocks = (input.numel() + threads - 1) / threads;

  scatter_add_cuda_kernel<<<blocks, threads>>>(
    input.data<float>(),
    input_index.data<long>(),
    output.data<float>(),
    counts.data<float>(),
    input.size(1),
    input.numel());

  scatter_div_cuda_kernel<<<blocks, threads>>>(
    output.data<float>(),
    counts.data<float>(),
    output.size(1),
    output.numel());
  
  // put mean back for each point
  scatter_back_cuda_kernel<<<blocks, threads>>>(
    input_means.data<float>(),
    output.data<float>(),
    input_index.data<long>(),
    input_means.size(1),
    input_means.numel());
}







