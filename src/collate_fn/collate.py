import logging
import torch
from typing import List
from src.utils import MelSpectrogram

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    melspec_gen = MelSpectrogram()
    
    audio_lengths = []
    audio_path = []
    for ds in dataset_items:
        audio_lengths.append(ds["audio"].shape[-1])
        audio_path.append(ds["audio_path"])

    batch_audio = torch.zeros(len(audio_lengths), max(audio_lengths))

    for i, ds in enumerate(dataset_items):
        batch_audio[i, :audio_lengths[i]] = ds["audio"]

    batch_spectrogram = melspec_gen(batch_audio)

    return {
        "spectrogram": batch_spectrogram,
        "audio": batch_audio,
        "audio_path": audio_path
    }
