import argparse
import json
from src.metric.utils import calc_wer, calc_cer

from pathlib import Path


def main(output_path):
    with output_path.open() as f:
        output_js = json.load(f)
    
    argmax_wers= []
    argmax_cers = []
    beamsearch_wers = []
    beamsearch_cers = []
    lm_wers = []
    lm_cers = []

    for pred in output_js:
        target_text = pred['ground_truth']
        argmax_pred = pred['pred_text_argmax']
        beamsearch_pred = pred['pred_text_beam_search'][0][0]
        lm_pred = pred['pred_text_lm']

        argmax_wers.append(calc_wer(argmax_pred, target_text))
        argmax_cers.append(calc_cer(argmax_pred, target_text))
        beamsearch_wers.append(calc_wer(beamsearch_pred, target_text))
        beamsearch_cers.append(calc_cer(beamsearch_pred, target_text))
        lm_wers.append(calc_wer(lm_pred, target_text))
        lm_cers.append(calc_cer(lm_pred, target_text))

    argmax_cer = sum(argmax_cers) / len(argmax_cers)
    argmax_wer = sum(argmax_wers) / len(argmax_wers)
    beamsearch_cer = sum(beamsearch_cers) / len(beamsearch_cers)
    beamsearch_wer = sum(beamsearch_wers) / len(beamsearch_wers)
    lm_cer = sum(lm_cers) / len(lm_cers)
    lm_wer = sum(lm_wers) / len(lm_wers)

    print(f"Argmax WER: {argmax_wer}; Argmax CER: {argmax_cer}")
    print(f"Bemasearch WER: {beamsearch_wer}; Beamsearch CER: {beamsearch_cer}")
    print(f"LM Beamsearch WER: {lm_wer}; LM Beamsearch CER: {lm_cer}")




if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="output json file path generated from test.py",
    )

    args = args.parse_args()

    output_path = Path(args.output)

    main(output_path)