import torch
import torch.nn as nn
from torch import Tensor



class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, pred_gen_mpd, pred_gen_msd, pred_real_mpd, pres_real_msd, **batch):
        mpd_loss = 0
        for gen_k, real_k in zip(pred_gen_mpd, pred_gen_mpd):
            mpd_loss += ((real_k - 1)**2).mean() + (gen_k**2).mean()

        msd_loss = 0
        for gen_k, real_k in zip(pred_gen_msd, pred_gen_msd):
            msd_loss += ((real_k - 1)**2).mean() + (gen_k**2).mean()
        
        return {"loss_D": mpd_loss + msd_loss}