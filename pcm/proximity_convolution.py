import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class proximity_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, knn_kernel=5, stride=1):
        super(proximity_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.knn_kernel = knn_kernel
        # self.pconv_weight = Parameter(torch.ones(in_channels, out_channels, kernel_size, kernel_size)/9.)
        self.pconv_weight = Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.pconv_weight)

    def forward(self, input, mean=torch.Tensor([0]), std=torch.Tensor([1])):
        device = input.device
        B, C, H, W = input.shape[0], input.shape[1], input.shape[-2], input.shape[-1]
        assert C == self.in_channels, "wrong input channels"
        knn_kernel2 = self.knn_kernel**2
        pad = self.knn_kernel // 2
        input_ch1 = (input[:, 0:1, ...].clone() * std.to(device)) + mean.to(device)
        unfold_input_ch1 = F.unfold(input_ch1, self.knn_kernel, padding=pad, stride=self.stride)
        unfold_input = F.unfold(input, self.knn_kernel, padding=pad, stride=self.stride)
        center = knn_kernel2 // 2
        difference = unfold_input_ch1 - unfold_input_ch1[:, center:center + 1, ...]
        difference = torch.abs(difference)
        difference[:, center:center + 1, ...] = -1
        _, knn_idx = torch.topk(difference, self.kernel_size**2, dim=1, largest=False)
        all_knn_index = []
        for i in range(C):
            all_knn_index.append(knn_idx + i * knn_kernel2)
        all_knn_index = torch.cat(all_knn_index, dim=1)
        unfold_input = torch.gather(input=unfold_input, dim=1, index=all_knn_index)
        output = torch.matmul(input=self.pconv_weight.view(self.out_channels, -1), other=unfold_input).view(B, self.out_channels, H//self.stride, W//self.stride)

        return output
