import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F

class DAIFNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4, ret_bottleneck=False):
        super(DAIFNet, self).__init__()
        self.conv_down = nn.ModuleList([self.convblock(input_ch, W)] + [self.convblock(W * (2**i), W * (2**i)) for i in range(1, D)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(W * (2**D), W * (2**D), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_up = nn.ModuleList([self.upconvblock(W * (2**i), W * (2**i) // 2) for i in range(D, 0, -1)])
        self.conv_joint = nn.ModuleList([self.convblock(W * (2**i), W * (2**i // 2)) for i in range(D, 0, -1)])

        self.conv_out = nn.Sequential(
            nn.Conv2d(W, W, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(W, output_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ret_bottleneck = ret_bottleneck
        
    def convblock(self, in_ch,out_ch):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),            
            nn.ReLU(),
        )
        return block
    
    def upconvblock(self,in_ch,out_ch):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        return block

    def forward(self, x):
        B, FS, C, H, W = x.shape
        h = x.view(B*FS, C, H, W) # BxFS C H W
        h_s = []
        for i, l in enumerate(self.conv_down):
            h = self.conv_down[i](h) # BxFS C H/2**i W/2**i
            pool_h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # BxFS C H/2**(i+1) W/2**(i+1)
            w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H/2**i W/2**i
            h_s.append(torch.max(w_h, dim=1)[0]) # B C H/2**i W/2**i
            # Global Operation
            stack_pool = pool_h.view(B, FS, *pool_h.shape[-3:])
            pool_max = torch.max(stack_pool, dim=1)[0].unsqueeze(1).expand_as(stack_pool).contiguous().view(B*FS, *pool_h.shape[-3:])
            h = torch.cat([pool_h, pool_max], dim=1)

        h = self.bottleneck(h)
        w_h = h.view(B, FS, *h.shape[-3:]) # B FS C H W
        h = torch.max(w_h, dim=1)[0]
        
        for i, l in enumerate(self.conv_up):
            h = self.conv_up[i](h)
            skip_h = h_s.pop(-1)
            h = self.conv_joint[i](torch.cat([h, skip_h], dim=1))
        
        output = self.conv_out(h)
        if self.ret_bottleneck:
            return output, w_h
        else:
            return output

class ParamNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4, near=0.2, far=10):
        super(ParamNet, self).__init__()
        self.conv_flat = nn.ModuleList([self.conv_block(input_ch * (2**i), input_ch * (2**(i+1))) for i in range(D)])
        self.linear = nn.Sequential(
            nn.Linear(input_ch * W, input_ch),
            nn.ReLU(),
            nn.Linear(input_ch, input_ch // 2),
            nn.ReLU(),
            nn.Linear(input_ch // 2, output_ch)
        )
        self.near = near 
        self.far = far
        
    def conv_block(self, in_ch, out_ch):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),            
            nn.ReLU(),
        )
        return block

    def forward(self, x):
        B, FS, C, H, W = x.shape
        h = x.view(B*FS, C, H, W) # BxFS C H W
        for i, l in enumerate(self.conv_flat):
            h = self.conv_flat[i](h) # BxFS C H/2**i W/2**i
            h = F.max_pool2d(h, kernel_size=2, stride=2, padding=0) # BxFS C H/2**(i+1) W/2**(i+1)
        fds = self.linear(h.squeeze())
        fds = torch.sigmoid(fds) * (self.far - self.near) + self.far
        fds = fds.view(B, FS) # B FS 1
        return fds

class LearnableThinLens(nn.Module):
    def __init__(self, fnumber=0.5, focal_length=2.9*1e-3, sensor_size=3.1*1e-3, img_size=256, pixel_size=None):
        super(LearnableThinLens, self).__init__()
        self.focal_length = nn.Parameter(torch.tensor(focal_length))
        self.D = nn.Parameter(torch.tensor(focal_length / fnumber))
        if not pixel_size:
            self.pixel_size = nn.Parameter(torch.tensor(sensor_size / img_size))
        else:
            self.pixel_size = nn.Parameter(torch.tensor(pixel_size))
        
    def getCoC(self, dpt, focus_dist):
        # dpt : BxFS H W
        # focus_dist : BxFS H W
        sensor_dist = focus_dist * self.focal_length / (focus_dist - self.focal_length)
        CoC = self.D * sensor_dist * torch.abs(1/self.focal_length - 1/sensor_dist - 1/(dpt+1e-8))
        sigma = CoC / 2 / self.pixel_size
        return sigma.float()
