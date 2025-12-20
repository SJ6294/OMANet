import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif hasattr(m, "initialize"):
            m.initialize()


import torch
import torch.nn as nn

class BNM(nn.Module):
    def __init__(self, num_channels, imagenet_mean=[0.485, 0.456, 0.406], imagenet_std=[0.229, 0.224, 0.225]):
        super(BNM, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(1, affine=True)
        self.imagenet_mean = torch.tensor(imagenet_mean).view(1, 3, 1, 1)  # [0.485, 0.456, 0.406]
        self.imagenet_std = torch.tensor(imagenet_std).view(1, 3, 1, 1)    # [0.229, 0.224, 0.225]

        self.initialize()

    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    def hsv_to_rgb(self, hsv: torch.Tensor) -> torch.Tensor:
        hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        _c = hsv_l * hsv_s
        _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
        _m = hsv_l - _c
        _o = torch.zeros_like(_c)
        idx = (hsv_h * 6.).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        rgb = torch.empty_like(hsv)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        rgb += _m
        return rgb

    def linear_rescale_per_image(self, v_norm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    
        min_vals = v_norm.view(v_norm.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        max_vals = v_norm.view(v_norm.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        
        denom = (max_vals - min_vals).clamp(min=eps)
        
        return (v_norm - min_vals) / denom

    def forward(self, x):

        hsv = self.rgb_to_hsv(x)
        v = hsv[:, 2:, :, :] 
        v_norm = self.instance_norm(v)
       
        v_final = self.linear_rescale_per_image(v_norm)  # [0,1]

        hsv_norm = torch.cat([hsv[:, 0:1, :, :], hsv[:, 1:2, :, :], v_final], dim=1)
        
        rgb_norm = self.hsv_to_rgb(hsv_norm)

        rgb_norm = (rgb_norm - self.imagenet_mean.to(rgb_norm.device)) / self.imagenet_std.to(rgb_norm.device)

        return rgb_norm, v_final, v

    def initialize(self):
        weight_init(self)
