import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
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


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)


class OPAM(nn.Module):
    def __init__(self, in_dim):
        super(OPAM, self).__init__()
        
        self.qkv_h = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1)
        self.qkv_dwconv_h = nn.Conv2d(in_dim * 3, in_dim * 3, kernel_size=3, stride=1, padding=1,
                                      groups=in_dim * 3)
        self.qkv_v = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1)
        self.qkv_dwconv_v = nn.Conv2d(in_dim * 3, in_dim * 3, kernel_size=3, stride=1, padding=1,
                                      groups=in_dim * 3)
        
        self.la = nn.Parameter(torch.zeros(1))
        self.lb = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.clam = LAM_Module_v2(2*in_dim) 
        self.conv_final = nn.Conv2d(2 * in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        
        self.object_detector = OPM(in_dim)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        qkv_h = self.qkv_dwconv_h(self.qkv_h(x))
        q_h, k_h, v_h = torch.chunk(qkv_h, 3, dim=1)
        
        axis_h = 1 * H
        view = (B, -1, axis_h)
        projected_query_h = q_h.view(*view).permute(0, 2, 1)
        projected_key_h = k_h.view(*view)
        attention_map_h = torch.bmm(projected_query_h, projected_key_h)
        attention_h = self.softmax(attention_map_h)
        projected_value_h = v_h.view(*view)
        out_h = torch.bmm(projected_value_h, attention_h.permute(0, 2, 1))
        out_h = out_h.view(B, C, H, W)
        out_h = self.la * out_h
        
        # Position-W
        qkv_v = self.qkv_dwconv_v(self.qkv_v(x))
        q_v, k_v, v_v = torch.chunk(qkv_v, 3, dim=1)
        axis_w = 1 * W
        view = (B, -1, axis_w)
        projected_query_w = q_v.view(*view).permute(0, 2, 1)
        projected_key_w = k_v.view(*view)
        attention_map_w = torch.bmm(projected_query_w, projected_key_w)
        attention_w = self.softmax(attention_map_w)
        projected_value_w = v_v.view(*view)
        out_w = torch.bmm(projected_value_w, attention_w.permute(0, 2, 1))
        out_w = out_w.view(B, C, H, W)
        out_w = self.lb * out_w
        
        out_combined = self.clam(torch.cat([out_h.unsqueeze(1), out_w.unsqueeze(1)], 1))
        
        out_final = self.conv_final(out_combined) + x
        
        object_prob = self.object_detector(attention_h, attention_w, out_final)
        # self.object_detector(attention_h, attention_w, out_final)
        # print(object_prob)
        return out_final, object_prob

class OPM(nn.Module):
    """
    prob = λe * (1 - min(normEnt_row, normEnt_col))
         + λs * sparsity_score
         + λg * global_feature_score
    """
    def __init__(self, in_dim, k=0.05):
        super().__init__()
        self.k = k          
        self.w_e = nn.Parameter(torch.tensor(0.0))
        self.w_s = nn.Parameter(torch.tensor(0.0))
        self.w_g = nn.Parameter(torch.tensor(0.0))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def _entropy(att, eps=1e-8):
        return -(att * (att + eps).log()).sum(-1).mean(-1)   # (B,)

    def _sparsity(self, att):
        B, L, _ = att.shape
        k = max(1, int(self.k * L * L))
        topk = torch.topk(att.view(B, -1), k, dim=-1)[0]     # (B,k)
        return (topk.mean(-1) - att.mean()).clamp(min=0)     # (B,)

    def forward(self, att_r, att_c, feat):
        B, C, H, W = feat.shape

        e_r = self._entropy(att_r) / math.log(att_r.size(-1))
        e_c = self._entropy(att_c) / math.log(att_c.size(-1))
        ent_score = (1.0 - torch.min(e_r, e_c)).clamp(0.0, 1.0)

        sparsity_raw  = self._sparsity(att_r) * 0.5 + self._sparsity(att_c) * 0.5
        sparsity_score = torch.sigmoid(sparsity_raw)

        g = self.pool(feat).view(B, -1)
        g_score = self.fc(g).squeeze(1)                      # (B,)

        w = torch.stack([self.w_e, self.w_s, self.w_g], dim=0)
        a = torch.softmax(w, dim=0)

        prob = a[0]*ent_score + a[1]*sparsity_score + a[2]*g_score
        prob = prob.clamp(0., 1.) 

        return prob

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x,mask=None):
        b,c,h,w = x.shape
        q=self.qkv1conv(self.qkv_0(x))
        k=self.qkv2conv(self.qkv_1(x))
        v=self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q=q*mask
            k=k*mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

class MSA_head(nn.Module):
    def __init__(self, mode='dilation',dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x,mask=None):
        x = x + self.attn(self.norm1(x),mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)

class MSA_module(nn.Module):
    def __init__(self, dim=128):
        super(MSA_module, self).__init__()
        self.B_TA = MSA_head()
        self.F_TA = MSA_head()
        self.TA = MSA_head()
        self.Fuse = nn.Conv2d(3*dim,dim,kernel_size=3,padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
    
    def forward(self,x,side_x,mask):
        N,C,H,W = x.shape
        mask = F.interpolate(mask,size=x.size()[2:],mode='bilinear')
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)
        xf = self.F_TA(x,mask_d)
        xb = self.B_TA(x,1-mask_d)
        x = self.TA(x)
        x = torch.cat((xb,xf,x),1)
        x = x.view(N,3*C,H,W)
        x = self.Fuse(x)
        D = self.Fuse2(side_x+side_x*x)
        return D
    
    def initialize(self):
        weight_init(self)


# Background Guide Module
class G_Spade(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(G_Spade, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(out_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh() 
        )
        self.mlp_beta = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, x, edge,  obj_prob=None):
        normalized = self.param_free_norm(x)

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Object-aware gating (α in [0,1]): 객체 부재 시 변조 억제
        if obj_prob is not None:
            if obj_prob.dim() == 1:
                alpha = obj_prob.view(-1, 1, 1, 1)
            else:
                alpha = obj_prob
            alpha = alpha.clamp(0., 1.).detach()   
            gamma = gamma * alpha
            beta  = beta  * alpha

        out = normalized * (1 + gamma) + beta
        return out

    def initialize(self):
        weight_init(self)



class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

    def initialize(self):
        weight_init(self)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
    def initialize(self):
        weight_init(self)


class ADM(nn.Module):
    def __init__(self):
        super(ADM, self).__init__()
        self.cbr = BasicConv(128, 64)
        
        self.edge_linear = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.initialize()
        
    def forward(self, x1, x2, x3, x4):

        layer2_edge = self.upsample2(x2)
        layer3_edge = self.upsample4(x3)
        layer4_edge = self.upsample8(x4)

        x_sum = x1+layer2_edge+layer3_edge
        x_att = x_sum * torch.sigmoid(layer4_edge) + x_sum
        x_cbr = self.cbr(x_att)

        edge_map = self.edge_linear(x_cbr)

        return torch.sigmoid(edge_map)
    
    def initialize(self):
        weight_init(self)


#### Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):  
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out
    

class AGM(nn.Module):
    def __init__(self, dim=128):
        super(AGM, self).__init__()

        self.spade = G_Spade(dim, dim)

        self.Fuse = BasicConv(2*dim,dim)

    def forward(self,x,edge, obj_prob=None):

        edge = F.interpolate(edge,size=x.size()[2:],mode='bilinear')
        spade_f = self.spade(x, edge, obj_prob)   
        fuse = self.Fuse(torch.cat([spade_f, x],1))
             
        return fuse  
    
    def initialize(self):
        weight_init(self)




class OMGM(nn.Module):
    def __init__(self, channels, switch_epoch=100):
        super(OMGM, self).__init__()

        self.switch_epoch = switch_epoch

        self.B_TA = MSA_head()
        self.F_TA = MSA_head()
        self.TA = MSA_head()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec = BasicConv(channels*2, channels)
        self.fuse = BasicConv(channels*3, channels)
        self.initialize()
    
    @staticmethod
    def _cosine_ramp(epoch: int, T: int, device):
        # 0 -> 1 : cosine schedule
        # epoch=0 => 0, epoch>=T => 1
        e = min(epoch, T)
        ramp = 0.5 * (1.0 - math.cos(math.pi * (e / T)))
        return torch.tensor(ramp, device=device, dtype=torch.float32)

        
    def forward(self, x_h, mask, x_l, obj_prob, epoch):
        x_h = self.up(x_h)

        x_cat = torch.cat([x_h, x_l], dim=1)
        out = self.dec(x_cat)

        N,C,H,W = out.shape
        xo = self.TA(out)

        mask = F.interpolate(mask,size=out.size()[2:],mode='bilinear')
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)
        
        xb = self.B_TA(out, 1 - mask_d)        
        xf = self.F_TA(out, mask_d)

        po = obj_prob.detach().view(N, 1, 1, 1).clamp(0.0, 1.0)

        ramp = self._cosine_ramp(epoch, self.switch_epoch, out.device)  
        s = (ramp * po).clamp(0.0, 1.0)                                  # (N,1,1,1)

        xf = xf * s
        xb = xb * (1.0 - s)

        out = torch.cat([xo, xb, xf], dim=1)
        out = self.fuse(out)
        out = torch.cat([xo, xb, xf], 1)

        out = self.fuse(out)
              
        return out   
    
    def initialize(self):
        weight_init(self)
    

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        # input : [64, 128, 320, 512]
        
        # camoformer
        self.side_conv1 = nn.Conv2d(512, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        self.aux_det = ADM()
        self.pa = OPAM(channels)

        self.agm2=AGM(channels)
        self.agm3=AGM(channels)
        self.agm4=AGM(channels)

        self.omgm2 = OMGM(channels)
        self.omgm3 = OMGM(channels)
        self.omgm4 = OMGM(channels)


        self.predtrans1  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans2  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans3  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans4  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.predtransd1  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtransd2  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtransd3  = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.initialize()
        


    def forward(self, E4, E3, E2, E1,shape, epoch):
        # E4, E3, E2, E1  : [512, 320, 128, 64]
        E4, E3, E2, E1= self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)

        E4, object_prob = self.pa(E4)

        aux_map = self.aux_det(E1, E2, E3, E4)  # input : [64, 128, 320, 512]

        P4 = self.predtrans4(E4)
        E3 = self.agm2(E3,aux_map, object_prob)

        P3 = self.predtrans3(E3)
        E2 = self.agm3(E2,aux_map, object_prob)

        P2 = self.predtrans2(E2)
        E1 = self.agm4(E1,aux_map, object_prob)

        P1 = self.predtrans1(E1)

        DF3 = self.omgm2(E4, P4, E3, object_prob, epoch)
        D3 = self.predtransd3(DF3)
        DF2 = self.omgm3(DF3, D3, E2, object_prob, epoch)
        D2 = self.predtransd2(DF2)
        DF1 = self.omgm4(DF2, D2, E1, object_prob, epoch)
        D1 = self.predtransd1(DF1)

        # coarse map
        P1 = F.interpolate(P1, size=shape, mode='bilinear')
        P2 = F.interpolate(P2, size=shape, mode='bilinear')
        P3 = F.interpolate(P3, size=shape, mode='bilinear')
        P4 = F.interpolate(P4, size=shape, mode='bilinear')       
        
        # detail map
        D1 = F.interpolate(D1, size=shape, mode='bilinear')
        D2 = F.interpolate(D2, size=shape, mode='bilinear')
        D3 = F.interpolate(D3, size=shape, mode='bilinear') 

        # shape : input image size
        aux_out = F.interpolate(aux_map, size=shape, mode='bilinear')
        
        predictions = object_prob, P4, P3, P2, P1, D3, D2, D1, aux_out
        
        return predictions

    def initialize(self):
        weight_init(self)




