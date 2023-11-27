from torch.nn as nn
import torch.nn.functional as F

from src.base import BaseModel

class ResBlock(nn.Module):
    def __init__(self, k, D, channels):
        super().__init__()
        self.k = k
        self.D = D
        
        self.res_layres = nn.ModuleList()
        for Dm in D:
            Dm_block = nn.Sequential()
            for Dnm in Dm:
                Dm_block.add_module(nn.LeakyReLU())
                Dm_block.add_module(nn.Conv1d(channels, channels, kernel=k, dialation=Dnm, 
                                              padding='same'))

            self.res_layers.append(Dm_block)
    
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


class HiFiGenerator(nn.Module):
    def __init__(self, ku, kr, Dr, hu=512, num_mels=80):
        super().__init__():

        self.start_conv = nn.Conv1d(num_mels, hu, kernel_size=7, padding='same')
        self.body = nn.Sequential()
        channels = hu
        for ku_l, l in enumerater(ku):
            self.body.add_module(nn.LeakyReLU())
            self.body.add_module(nn.ConvTranspose1d(channels, channels // 2,
                                                    kernel_size=ku_l, stride=ku_l // 2,
                                                    padding = ku_l // 4))
            self.body.add_module(MultiReceptieveFieldFusion(kr, Dr, channels // 2))
            channels = channels // 2

        self.end_conv = nn.Conv1d(channels, 1, kernel_size=7, padding='same')


    def forward(self, spectrogram):
        # spectrogram: (B, H, T)
        x = self.start_conv(spectrogram) 
        x = self.body(x)
        x = self.end_conv(F.leaky_relu(x)) # (B, 1, T')
        x = F.tanh(x).squeeze(1) # (B, T')

        return {"gen_audio": x}
