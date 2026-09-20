import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):
    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale, mask_hist=None, sat_hist=None):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  


def pmc_loss(logits_gr, logits_rg, target):
    loss_gr = F.cross_entropy(logits_gr, target)
    loss_rg = F.cross_entropy(logits_rg, target)
    return 0.5 * (loss_gr + loss_rg)

class InfoSem(nn.Module):
    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu', nce_weight=1.0, kl_weight=0, mmd_weight=0):
        super().__init__()
        self.kl_weight = kl_weight
        self.nce_weight = nce_weight
        self.mmd_weight = mmd_weight

        self.nce = InfoNCE(loss_function=loss_function, device=device)

    def forward(self, image_features1, image_features2, logit_scale, out_loss):
        loss = self.nce_weight * self.nce(image_features1, image_features2, logit_scale)
        pmc = pmc_loss(out_loss[0],out_loss[1],out_loss[2])
        loss = loss + 0.025*pmc

        return loss