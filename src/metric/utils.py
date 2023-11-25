# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1.

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_text_words = target_text.split(' ')
    predicted_text_words = predicted_text.split(' ')
    if len(target_text_words) == 0:
        return 1.

    return editdistance.eval(target_text_words, predicted_text_words) / len(target_text_words)
