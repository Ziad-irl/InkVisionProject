import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from evaluate import evaluate_prediction

def test_evaluate_prediction_inconsistent_lengths():
    y_true = [1, 0, 1, 1, 0] * 12  # 60 elements
    y_pred = [1, 0, 1] * 11        # 33 elements
    with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples"):
        evaluate_prediction(y_true, y_pred)