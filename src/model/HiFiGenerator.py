import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from src.base import BaseModel

class ResBlock(nn.Module):
    def __init__(self, k, D, channels):
        super().__init__()
        self.k = k
        self.D = D
        
        self.res_layers = nn.ModuleList()
        for Dm in D:
            Dm_block = []
            for Dnm in Dm:
                Dm_block.append(nn.LeakyReLU())
                Dm_block.append(weight_norm(nn.Conv1d(channels, channels, kernel_size=k, dilation=Dnm, 
                                              padding='same')))
            self.res_layers.append(nn.Sequential(*Dm_block))
    
    def forward(self, x):
        for res_layer in self.res_layers:
            x = x + res_layer(x)
        return x


class MultiReceptieveFieldFusion(nn.Module):
    def __init__(self, kr, Dr, channels):
        super().__init__()

        self.ResBlocks = nn.ModuleList()
        for k, D in zip(kr, Dr):
            self.ResBlocks.append(ResBlock(k, D, channels))
        
    def forward(self, x):
        out = 0.
        for res_block in self.ResBlocks:
            out = out + res_block(x)
        return out


class HiFiGenerator(BaseModel):
    def __init__(self, ku, kr, Dr, hu=512, num_mels=80):
        super().__init__()

        self.start_conv = weight_norm(nn.Conv1d(num_mels, hu, kernel_size=7, padding='same'))
        self.body = []
        channels = hu
        for ku_l in ku:
            self.body.append(nn.LeakyReLU())
            self.body.append(weight_norm(nn.ConvTranspose1d(channels, channels // 2,
                                                            kernel_size=ku_l, stride=ku_l // 2,
                                                            padding = ku_l // 4)))
            self.body.append(MultiReceptieveFieldFusion(kr, Dr, channels // 2))
            channels = channels // 2

        self.body = nn.Sequential(*self.body)
        self.end_conv = weight_norm(nn.Conv1d(channels, 1, kernel_size=7, padding='same'))


    def forward(self, spectrogram, **batch):
        # spectrogram: (B, H, T)
        x = self.start_conv(spectrogram) 
        x = self.body(x)
        x = self.end_conv(F.leaky_relu(x)) # (B, 1, T')
        x = F.tanh(x).squeeze(1) # (B, T')

        return {"gen_audio": x}
