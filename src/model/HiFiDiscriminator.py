import torch.nn as nn 
import torch.nn.functional as F



class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding='same'),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20),
            nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20),
            nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20),
            nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding='same')
            ])

        self.last_conv = nn.Conv1d(1024, 1, kernel_size=3, padding='same')


    def forward(self, x):
        # x : (B, T)
        feature_map = []
        x.unsqueeze(1) # (B, 1, T)
        for conv in self.body:
            x = F.leaky_relu(conv(x))
            feature_map.append(x)
        
        x = self.last_conv(x) # (B, 1, T')
        return x.flatten(1, -1), feature_map



class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList()
        for _ in range(3):
            self.discriminators.append(PeriodDiscriminator())



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
    def __init__(self, period)
        super().__init__()

        self.p = period
        self.body = nn.ModuleList([
            nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, kernel_size=(5, 1), padding=(2, 0))
            ])
                                  
        
        self.last_conv = nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))

        
    
    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        reshape_dim = (T - 1) // self.p + 1
        pad_shape = reshape_dim * self.p - T
        x = F.pad(x, (0, pad_shape), "constant", 0)
        x = x.reshape((B, reshape_dim, self.p)).unsqueeze(1) # (B, 1, T / p, p)
        feature_map = []
        for conv in self.body:
            x = conv(x)
            feature_map.append(x)

        x = x.flatten(1, -1)
        
        return x, feature_map









class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PeriodSicriminator(2),
            PeriodSicriminator(3),
            PeriodSicriminator(5),
            PeriodSicriminator(7),
            PeriodSicriminator(11)
        ])

    def forward(self, x):
        # x: (B, T)
        outputs = []
        feature_maps = []
        for D in self.discriminators:
            out, fmap = D(x)
            outputs.append(out)
            feature_maps.append(fmap)


class HiFiDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()


    def forward(self, audio_real, audio_gen):

        pred_real_msd, fm_real_msd = self.msd(audio_real)
        pred_gen_msd, fm_gen_msd = self.msd(audio_gen)
        pred_real_mpd, fm_real_mpd = self.mpd(audio_real)
        pred_gen_mpd, fm_gen_mpd = self.mpd(audio_gen)

        return {"pred_real_msd": pred_real_msd,
                "pred_gen_msd": pred_gen_msd,
                "pred_real_mpd": pred_real_mpd,
                "pred_gen_mpd": pred_gen_mpd,
                "fm_real_msd": fm_real_msd,
                "fm_gen_msd": fm_gen_msd,
                "fm_real_mpd": fm_real_mpd,
                "fm_gen_mpd": fm_gen_mpd}