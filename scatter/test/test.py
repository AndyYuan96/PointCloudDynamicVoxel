import torch
from torch.autograd import Function
from scatter_max import scatter_max

class ScatterMaxCuda(Function):
    @staticmethod
    def forward(ctx, input, input_index, output, output_index): 
        scatter_max(input, input_index, output, output_index, True)
        ctx.size = input.size()
        ctx.save_for_backward(output_index)

        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        output_index = ctx.saved_tensors[0]
        grad_input = output_grad.new_zeros(ctx.size)
        grad_input.scatter_(0, output_index, output_grad)
        return grad_input, None, None, None

def scatterMax(input, input_index, voxel_nums, train):
    '''
        only accept two dimension tensor, and do maxpooing in first dimension
    '''
    output = input.new_full((voxel_nums, input.shape[1]), torch.finfo(input.dtype).min)
    output_index = input_index.new_empty((voxel_nums, input.shape[1]))

    if train:
        output = ScatterMaxCuda.apply(input, input_index, output, output_index)
    else:
        output = scatter_max(input, input_index, output, output_index, False)
    
    return output


def scatterMean(input, input_index, voxel_nums):
    output = input.new_full((voxel_nums, input.shape[1]), 0.0)
    input_mean = input.new_empty(input.shape)

    scatter_mean(input, input_index, output,input_mean)
    return input_mean


if __name__ == '__main__':
    input_ = torch.tensor([[2, 0, 1, 4, 3], [2, 2, 1, 3, 4]],dtype=torch.float32, requires_grad=True).cuda()
    input_index = torch.tensor([[0,0,0,0,0], [0,0,0,0,0]]).long().cuda()

    tt = scatterMax(input_, input_index, 1, True)
    c = tt.sum()
    c.backward()