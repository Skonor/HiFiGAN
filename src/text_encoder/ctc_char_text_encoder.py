from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder
import multiprocessing

import torch

from .char_text_encoder import CharTextEncoder
from src.utils import ROOT_PATH




class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.lm_path = ROOT_PATH / 'data' / 'LMs' / '3-gram.pruned.1e-7.arpa'
        self.lm_loaded = False
        self.lm_decoder = None

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            if self.ind2char[ind] == self.EMPTY_TOK:
                last_char = self.EMPTY_TOK
                continue
            if self.ind2char[ind] != last_char:
                result.append(self.ind2char[ind])
                last_char = self.ind2char[ind]

        return ''.join(result)

    def extend_and_merge(self, frame: torch.tensor, hypos_dict: defaultdict(float)) -> defaultdict(float):

        new_hypos = defaultdict(float)
        
        for next_char_ind, next_char_proba in enumerate(frame):
            for (pref, last_char), prob in hypos_dict.items():
                next_char = self.ind2char[next_char_ind]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char == self.EMPTY_TOK:
                        new_pref = pref
                    else:
                        new_pref = pref + next_char
                new_hypos[(new_pref, next_char)] += next_char_proba.item() * prob
        
        return new_hypos

    def truncate(self, hypos_dict: defaultdict(float) , beam_size: int):
        hypos_list = list(hypos_dict.items())
        hypos_list.sort(key=lambda x: -x[1])
        return dict(hypos_list[:beam_size])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 5) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        hypos_dict = {('', self.EMPTY_TOK): 1.0}

        for i in range(probs_length):
            hypos_dict = self.extend_and_merge(probs[i, :], hypos_dict)
            hypos_dict = self.truncate(hypos_dict, beam_size)

        final_hypos_dict = defaultdict(float)
        for (text, last_char), prob in hypos_dict.items():
            final_hypos_dict[text] += prob

        for text, prob in final_hypos_dict.items():
            hypos.append(Hypothesis(text, prob))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    
    
    def ctc_lm_beam_search(self, log_probs: torch.tensor, probs_length: torch.tensor, beam_size = 10) -> List[str]:

        if not self.lm_loaded:
            if not self.lm_path.exists():
                raise RuntimeError("LM not loaded. Load it by running scripts/load_lm.py")
            labels = [""] + list(self.alphabet)
            labels =  list(map(lambda x: x.upper(), labels))
            self.lm_decoder = build_ctcdecoder(labels=labels, kenlm_model_path=str(self.lm_path))
            self.lm_loaded = True

        logits_list = [log_probs[i][:probs_length[i]].numpy() for i in range(len(probs_length))]
        
        with multiprocessing.get_context("fork").Pool() as pool:
            pred_list = self.lm_decoder.decode_batch(pool, logits_list, beam_width=beam_size)
        
        pred_list = list(map(lambda x: x.lower(), pred_list))

        return pred_list
