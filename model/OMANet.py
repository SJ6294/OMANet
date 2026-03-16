import torch
from torch import nn

from .encoder.BNM import BNM
from .encoder.pvtv2_encoder import pvt_v2_b2
from .decoder.OMANet_dec import Decoder  
from timm.models import create_model

import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class OMANet(torch.nn.Module):
    def __init__(self, load_path=None):   
        super(OMANet, self).__init__()

        self.encoder = pvt_v2_b2()
        if load_path is not None:
            pretrained_dict = torch.load(load_path)  
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)
            print('Pretrained encoder loaded.')
        self.enhancement = BNM(1)
        self.decoder = Decoder(128)
        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, epoch, shape=None):

        out = self.enhancement(x)

        features = self.encoder(out[0])
        x1 = features[0]  # 64 1st
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        if shape is None:
            shape = x.size()[2:]

        predictions = self.decoder(x1, x2, x3, x4, shape, epoch)
        return predictions, out

    def initialize(self):
        weight_init(self)

if __name__ == "__main__":
    import torch
    from thop import profile
    from ptflops import get_model_complexity_info

    model = OMANet()
    model.eval()

    input_tensor = torch.randn(1, 3, 352, 352)

    output, _ = model(input_tensor)
    print(f"Output shape: {output[-2].shape}")

    macs, params = profile(model, inputs=(input_tensor, ))
    thop_gmac = macs /  1e9 
    thop_params = params / 1e6


    input_res = 352  
    with torch.cuda.device(0): 
        macs, params_str = get_model_complexity_info(model, (3, input_res, input_res), 
                                                     as_strings=True,
                                                     print_per_layer_stat=False, 
                                                     verbose=False)
    

    from calflops import calculate_flops

    input_shape = (1, 3, 352, 352)

    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    print("\n[calflops library]")
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    print("\n[thop library]")
    print(f"Computational complexity: {thop_gmac:.2f} GMac")
    print(f"Number of parameters: {thop_params:.2f} M")
    
    print("\n[ptflops library]")
    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params_str}")
