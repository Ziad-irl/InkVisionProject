import numpy as np
import editdistance
from difflib import SequenceMatcher

def wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / float(len(r))

def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(1, len(ref))

def f1_char_overlap(ref, hyp):
    matcher = SequenceMatcher(None, ref, hyp)
    match = sum(triple.size for triple in matcher.get_matching_blocks())
    precision = match / max(len(hyp), 1)
    recall = match / max(len(ref), 1)
    f1 = 2 * precision * recall / max((precision + recall), 1e-8)
    return f1

def evaluate_prediction(prediction, ground_truth):
    """Compute WER, CER, and F1 Score for a single prediction."""
    # If inputs are strings, convert to lists of chars/words as needed
    if isinstance(prediction, str) and isinstance(ground_truth, str):
        prediction = prediction.strip().lower()
        ground_truth = ground_truth.strip().lower()
        return {
            "WER": round(wer(ground_truth, prediction) * 100, 2),
            "CER": round(cer(ground_truth, prediction) * 100, 2),
            "F1": round(f1_char_overlap(ground_truth, prediction) * 100, 2)
        }
    elif isinstance(prediction, list) and isinstance(ground_truth, list):
        if len(prediction) != len(ground_truth):
            raise ValueError("Found input variables with inconsistent numbers of samples")
        # ... list-based metrics (e.g., precision_score, etc.) ...
        pass
    else:
        raise ValueError("Inputs must be both strings or both lists")
