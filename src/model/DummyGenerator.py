from torch import nn
from torch.nn import Sequential

from src.base import BaseModel


class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, spectrogram):
        return {"gen_audio": 0.}
