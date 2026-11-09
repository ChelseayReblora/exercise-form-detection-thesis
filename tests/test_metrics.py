"""
Covers thesis objective #3: "evaluate the accuracy of the model's performance
by using percentage error" (Chapter 3, Statistical Treatment, Eq. 3.1 / Table 3.2).
"""
import numpy as np
import pytest

from metrics import percentage_error, misclassification_rate


class TestPercentageErrorFormula:
    def test_all_correct_gives_zero_error(self):
        # VA (correct count) == VE (total tested) -> perfect run
        assert percentage_error(va=100, ve=100) == 0.0

    def test_all_incorrect_gives_negative_100(self):
        # VA = 0 correct out of VE tested
        assert percentage_error(va=0, ve=100) == -100.0

    def test_matches_thesis_example_values(self):
        # Sanity-check against the magnitude style reported in Chapter 2
        # (e.g. "2.62% for systolic blood pressure") -- same formula shape.
        result = percentage_error(va=95, ve=100)
        assert result == pytest.approx(-5.0)

    def test_raises_on_zero_expected_samples(self):
        with pytest.raises(ValueError):
            percentage_error(va=0, ve=0)


class TestMisclassificationRate:
    def test_matches_train_model_py_computation(self):
        """train_model.py computes:
            incorrect_predictions = sum(pred != true)
            percentage_error = incorrect / total * 100
        This must stay in sync with metrics.misclassification_rate so the
        reported thesis metric doesn't silently drift if train_model.py changes."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])  # 2 wrong out of 5
        assert misclassification_rate(y_true, y_pred) == pytest.approx(40.0)

    def test_perfect_predictions_zero_error(self):
        y_true = np.array([0, 1, 2, 1, 0])
        assert misclassification_rate(y_true, y_true) == 0.0

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            misclassification_rate(np.array([]), np.array([]))


class TestFormulaSignConventionDiscrepancy:
    def test_thesis_formula_and_train_script_disagree_on_sign(self):
        """Flag for the researchers: Eq. 3.1 in the thesis ((VA-VE)/VE*100,
        VA=correct count) always yields a NEGATIVE or zero percentage, while
        train_model.py's actual implementation reports a POSITIVE misclassification
        rate (incorrect/total*100). Both encode the same magnitude of error but
        with opposite signs -- worth reconciling before writing up Table 3.2 in
        the final paper so the reported numbers match the stated formula."""
        va, ve = 90, 100
        thesis_formula_result = percentage_error(va=va, ve=ve)
        y_true = np.array([1] * va + [0] * (ve - va))
        y_pred = np.array([1] * va + [1] * (ve - va))  # all "incorrect" entries misclassified
        train_script_result = misclassification_rate(y_true, y_pred)

        assert thesis_formula_result == pytest.approx(-10.0)
        assert train_script_result == pytest.approx(10.0)
        assert thesis_formula_result == pytest.approx(-train_script_result)
