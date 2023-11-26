import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.lambda_mel = 45.0
        self.lamda_fm = 2.


    def forward(self, pred_gen_mpd, pred_gen_msd, fm_real_mpd, fm_real_msd, 
    fm_gen_mpd, fm_gen_msd, generated_spectrogram, spectrogram, **batch):

    mel_loss = F.l1_loss(spectrogram, generated_spectrogram)

    adv_loss_mpd = 0.
    for gen_k in pred_gen_mpd:
        adv_loss_mpd += ((gen_k - 1)**2).mean()
    
    adv_loss_msd = 0.
    for gen_k in pred_gen_msd:
        adv_loss_msd += ((gen_k - 1)**2).mean()

    adv_loss = adv_loss_mpd + adv_loss_msd

    fm_loss_mpd = 0.
    for f_real, f_gen in zip(fm_real_mpd, fm_gen_mpd):
        fm_loss_mpd += F.l1_loss(f_real, f_gen)

    fm_loss_msd = 0.
    for f_real, f_gen in zip(fm_real_msd, fm_gen_msd):
        fm_loss_msd += F.l1_loss(f_real, f_gen)

    fm_loss = fm_loss_mpd + fm_loss_msd

    loss_G = adv_loss + self.lambda_mel * mel_loss + self.lambda_fm * fm_loss

    return {
        "loss_G": loss_G,
        "mel_loss": mel_loss,
        "fm_loss": fm_loss,
        "adv_loss_G": adv_loss
    }