

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def bce_iou_loss(pred: torch.Tensor,
                 mask: torch.Tensor,
                 smooth: float = 1.0) -> torch.Tensor:
    """
    pred  : (B, 1, H, W)  ─ raw logits
    mask  : (B, 1, H, W)  ─ binary GT (0/1)
    return: scalar loss = BCE + IoU
    """
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred_prob = torch.sigmoid(pred)
    intersection = (pred_prob * mask).sum(dim=(2, 3))
    union        = (pred_prob + mask).sum(dim=(2, 3)) - intersection
    iou_loss     = 1.0 - (intersection + smooth) / (union + smooth)
    iou_loss     = iou_loss.mean()

    return bce + iou_loss


class AdaptiveSegLoss(nn.Module):
    """Noise‑aware loss: focal + λ·Dice + (1-λ)·FalseDice, SNR‑weighted."""
    def __init__(self, alpha=0.8, gamma=2.0, beta=0.1, cosine_T=120, kappa=1.0):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.beta=beta; self.T=cosine_T; self.kappa=kappa

    def _weighted_dice(self,p,t,w,eps=1e-6):
        b=p.size(0); p=p.view(b,-1); t=t.view(b,-1); w=w.view(b,-1)
        inter = 2*(w*p*t).sum(1)+eps; denom=(w*(p+t)).sum(1)+eps
        return 1-inter/denom
    
    def _weighted_false_dice(self,p,t,w,eps=1e-6):
        b=p.size(0); p=p.view(b,-1); t=t.view(b,-1); w=w.view(b,-1)
        inter = 2*(w*(1-p)*(1-t)).sum(1)+eps; denom=(w*((1-p)+(1-t))).sum(1)+eps
        return 1-inter/denom
    
    def _weighted_focal(self,p,t,w,eps=1e-12):
        pt=torch.where(t==1,p,1-p); alpha_t=torch.where(t==1,self.alpha,1-self.alpha)
        fl = -alpha_t*((1-pt)**self.gamma)*torch.log(pt.clamp(min=eps))
        return (w*fl).view(p.size(0),-1).mean(1)
    
    def _dice(self,p,t,eps=1e-6):
        b=p.size(0); p=p.view(b,-1); t=t.view(b,-1)
        inter = 2*(p*t).sum(1)+eps; denom=((p+t)).sum(1)+eps
        return 1-inter/denom
    
    def _false_dice(self,p,t,eps=1e-6):
        b=p.size(0); p=p.view(b,-1); t=t.view(b,-1)
        inter = 2*((1-p)*(1-t)).sum(1)+eps; denom=(((1-p)+(1-t))).sum(1)+eps
        return 1-inter/denom

    def forward(self, seg_prob:torch.Tensor, target:torch.Tensor, snr_map:torch.Tensor, obj_prob:torch.Tensor,
                epoch:int, mode:str='tr'):
        
        B=seg_prob.size(0)
        w = 1 - torch.clamp(snr_map, min=1e-3, max=1.0).pow(self.kappa)  # (B,1,H,W)

        gt_obj=(target.view(B,-1).sum(1)>0).float()
        if mode == 'tr':
            lam_c = 0.5*(1+math.cos(math.pi*min(epoch,self.T)/self.T))  # 1→0
            lam = (1-lam_c)*obj_prob.detach() + lam_c*gt_obj  # (B,)
        else:
            lam = obj_prob.detach()


        l_focal = self._weighted_focal(seg_prob, target, w)
        l_dice  = self._weighted_dice(seg_prob, target, w)
        l_fdice = self._weighted_false_dice(seg_prob, target, w)


        seg_loss = l_focal + lam*l_dice + (1-lam)*l_fdice

        L_obj = F.binary_cross_entropy(obj_prob, gt_obj, reduction='mean')
        return seg_loss.mean()+self.beta*L_obj, seg_loss.mean(), L_obj
