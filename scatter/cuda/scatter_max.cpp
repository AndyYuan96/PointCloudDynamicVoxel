#include <torch/extension.h>
#include <iostream>
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be CUDA tensor")


 
void scatter_max_cuda(torch::Tensor input, torch::Tensor input_index,
                      torch::Tensor output, torch::Tensor output_index,
                      bool train);

void scatter_mean_cuda(torch::Tensor input, torch::Tensor input_index,
                      torch::Tensor output, torch::Tensor counts, torch::Tensor input_means);




void scatter_max(torch::Tensor input, torch::Tensor input_index, torch::Tensor output, torch::Tensor output_index, bool train)
{
  CHECK_CUDA(input);
  CHECK_CUDA(input_index);
  CHECK_CUDA(output);
  CHECK_CUDA(output_index);
  scatter_max_cuda(input, input_index, output, output_index, train);
}

void scatter_mean(torch::Tensor input, torch::Tensor input_index, torch::Tensor output, torch::Tensor input_means)
{
  CHECK_CUDA(input);
  CHECK_CUDA(input_index);
  CHECK_CUDA(output);

  torch::Tensor counts = torch::zeros({output.size(0)},output.options());
  scatter_mean_cuda(input, input_index, output, counts, input_means);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max", &scatter_max, "ScatterMaxCuda");
  m.def("scatter_mean", &scatter_mean, "scatterMeanCuda");
}
