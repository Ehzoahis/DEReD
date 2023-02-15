# MIT License

# Copyright (c) 2018 shirgur

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Modified from Single Image Depth Estimation Trained via Depth from Defocus Cues
# (https://github.com/shirgur/UnsupervisedDepthFromFocus) by Shir Gur et al.

import torch 
import torch.nn as nn
from torch.autograd import Function

import gauss_psf_cuda as gauss_psf

class GaussPSFFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, device, kernel_size=11):
        with torch.no_grad():
            x = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(kernel_size, 1).float().repeat(1, kernel_size).to(device)

            y = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(1, kernel_size).float().repeat(kernel_size, 1).to(device)

        outputs, wsum = gauss_psf.forward(input, weights, x, y)
        ctx.save_for_backward(input, outputs, weights, wsum, x, y)
        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, outputs, weights, wsum, x, y = ctx.saved_variables
        x = -x
        y = -y
        grad_input, grad_weights = gauss_psf.backward(grad.contiguous(), input, outputs, weights, wsum, x, y)
        return grad_input, grad_weights, None, None

class GaussPSF(nn.Module):
    def __init__(self, kernel_size):
        super(GaussPSF, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, image, psf):
        psf = psf.unsqueeze(1).expand_as(image).contiguous()
        return GaussPSFFunction.apply(image, psf, image.device, self.kernel_size)