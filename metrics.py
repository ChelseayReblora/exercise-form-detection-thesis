"""Statistical treatment metric from the thesis (Chapter 3, Eq. 3.1).

Percentage Error = (VA - VE) / VE * 100

Where, per the thesis text:
    VA = Actual Value  -> number of correctly classified exercises
    VE = Expected Value -> total number of tested samples
"""
import numpy as np


def percentage_error(va, ve):
    """Percentage Error per thesis Eq. 3.1.

    Returns a signed value. Since VA (correct count) <= VE (total tested),
    the result is always <= 0; its magnitude is the misclassification rate.
    """
    if ve == 0:
        raise ValueError("VE (total tested samples) cannot be zero.")
    return (va - ve) / ve * 100.0


def misclassification_rate(y_true, y_pred):
    """Straightforward incorrect/total * 100, as actually computed in train_model.py."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute error rate over zero samples.")
    incorrect = np.sum(y_true != y_pred)
    return (incorrect / len(y_true)) * 100.0
