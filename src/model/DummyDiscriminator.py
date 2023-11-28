from torch import nn
from torch.nn import Sequential



class DummyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.msd = None
        self.mpd = None


    def forward(self, audio_real, audio_fake):

        return {"pred_real_msd": 0,
                "pred_gen_msd": 0,
                "pred_real_mpd": 0,
                "pred_gen_mpd": 0,
                "fm_real_msd": 0,
                "fm_gen_msd": 0,
                "fm_real_mpd": 0,
                "fm_gen_mpd": 0}
