import torch
from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply

from torch import Tensor
from torchaudio import transforms as T


class TimeStretch(AugmentationBase):
    def __init__(self, stretch, *args, **kwargs):
        self._aug = T.TimeStretch(*args, **kwargs)
        self.stretch = stretch

    def __call__(self, data: Tensor):
            x = data.unsqueeze(1)
            x = self._aug(x, self.stretch).squeeze(1)
            x = torch.absolute(x)
            return x

class RandomTimeStretch(RandomApply):
    def __init__(self, *args, **kwargs):
        if 'p' not in kwargs:
            p = 0.3
        else:
            p = kwargs['p']
            del kwargs['p']
        super.__init__(TimeStretch(*args, **kwargs), p = p)