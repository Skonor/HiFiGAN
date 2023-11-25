import torch_audiomentations
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply

class Noise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

 
class RandomNoise(RandomApply):
    def __init__(self, *args, **kwargs):
        if 'p' not in kwargs:
            p = 0.3
        else:
            p = kwargs['p']
            del kwargs['p']
        super.__init__(Noise(*args, **kwargs), p = p)