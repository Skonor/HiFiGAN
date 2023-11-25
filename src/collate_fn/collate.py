import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    
    spec_lengths = []
    text_encoded_length = []
    for ds in dataset_items:
        spec_lengths.append(ds['spectrogram'].shape[-1])
        text_encoded_length.append(ds['text_encoded'].shape[-1])

    spec_dim = dataset_items[0]['spectrogram'].shape[1]
    batch_spectrogram = torch.zeros(len(spec_lengths), spec_dim, max(spec_lengths))
    batch_encoded_text = torch.zeros(len(text_encoded_length), max(text_encoded_length))

    texts = []
    audio_path = []
    audio = []
    for i, ds in enumerate(dataset_items):
        batch_spectrogram[i, :, :spec_lengths[i]] = ds['spectrogram']
        batch_encoded_text[i, :text_encoded_length[i]] = ds['text_encoded']
        texts.append(ds['text'])
        audio_path.append(ds['audio_path'])
        audio.append(ds['audio'])

    text_encoded_length = torch.tensor(text_encoded_length).long()
    spec_lengths = torch.tensor(spec_lengths).long()

    return {
        'spectrogram': batch_spectrogram,
        'spectrogram_length': spec_lengths,
        'text_encoded': batch_encoded_text,
        'text_encoded_length': text_encoded_length,
        'text': texts,
        'audio_path': audio_path,
        'audio': audio
    }
