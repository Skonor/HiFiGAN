import torch.nn as nn 
import torch.nn.functional as F
from src.base import BaseModel
from torch.nn.utils import weight_norm



class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding='same')),
            weight_norm(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding='same'))
            ])

        self.last_conv = nn.Conv1d(1024, 1, kernel_size=3, padding='same')


    def forward(self, x):
        # x : (B, T)
        feature_map = []
        x = x.unsqueeze(1) # (B, 1, T)
        for conv in self.body:
            x = F.leaky_relu(conv(x))
            feature_map.append(x)
        
        x = self.last_conv(x) # (B, 1, T')
        feature_map.append(x)
        return x.flatten(1, -1), feature_map



class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList()
        for _ in range(3):
            self.discriminators.append(ScaleDiscriminator())



    def forward(self, x):
        # x: (B, T)
        feature_maps = []
        outputs = []
        for i, Dk in enumerate(self.discriminators):
            out, feature_map = Dk(x)
            feature_maps.append(feature_map)
            outputs.append(out)
            if i < len(self.discriminators) - 1:
                x = F.avg_pool1d(x, kernel_size=4, padding=2)
        
        return outputs, feature_maps


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()

        self.p = period
        self.body = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), padding=(2, 0)))
            ])
                                  
        
        self.last_conv = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))

        
    
    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        reshape_dim = (T - 1) // self.p + 1
        pad_shape = reshape_dim * self.p - T
        x = F.pad(x, (0, pad_shape), "constant", 0)
        x = x.reshape((B, reshape_dim, self.p)).unsqueeze(1) # (B, 1, T / p, p)
        feature_map = []
        for conv in self.body:
            x = F.leaky_relu(conv(x))
            feature_map.append(x)
        
        x = self.last_conv(x)
        feature_map.append(x)
        x = x.flatten(1, -1)
        
        return x, feature_map


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11)
        ])

    def forward(self, x):
        # x: (B, T)
        outputs = []
        feature_maps = []
        for D in self.discriminators:
            out, fmap = D(x)
            outputs.append(out)
            feature_maps.append(fmap)

        return outputs, feature_maps


class HiFiDiscriminatorB(BaseModel):
    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()


    def forward(self, audio, gen_audio, **batch):

        pred_real_msd, fm_real_msd = self.msd(audio)
        pred_gen_msd, fm_gen_msd = self.msd(gen_audio)
        pred_real_mpd, fm_real_mpd = self.mpd(audio)
        pred_gen_mpd, fm_gen_mpd = self.mpd(gen_audio)

        return {"pred_real_msd": pred_real_msd,
                "pred_gen_msd": pred_gen_msd,
                "pred_real_mpd": pred_real_mpd,
                "pred_gen_mpd": pred_gen_mpd,
                "fm_real_msd": fm_real_msd,
                "fm_gen_msd": fm_gen_msd,
                "fm_real_mpd": fm_real_mpd,
                "fm_gen_mpd": fm_gen_mpd}
