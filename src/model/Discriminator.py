from torch import nn
from torch.nn import Sequential



class HiFiDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = ...
        self.mpd = ...


    def forward(self, audio_real, audio_fake):

        return {"pred_real_msd": 0,
                "pred_fake_msd": 0,
                "pred_real_mpd": 0,
                "pred_fake_mpd": 0,
                "feat_real_msd": 0,
                "feat_fake_msd": 0,
                "feat_real_mpd": 0,
                "feat_fake_mpd": 0}
