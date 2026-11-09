"""
Covers thesis objective #2 (form classification via GRU) at the model-artifact
level: given the actual gru_model.keras/scaler.pkl/encoders shipped in the
repo, does the inference path in realtime_detector.py produce well-formed,
consistent predictions? These tests don't need a camera at all.
"""
import numpy as np
import pytest

from conftest import requires_model, requires_tensorflow, MODEL_DIR, SEQ_LEN, FEATURE_DIM


@requires_tensorflow
@requires_model
class TestModelIO:
    @staticmethod
    @pytest.fixture(scope="class")
    def loaded_assets():
        import joblib
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_DIR / "gru_model.keras")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        exercise_encoder = joblib.load(MODEL_DIR / "exercise_encoder.pkl")
        form_encoder = joblib.load(MODEL_DIR / "form_encoder.pkl")
        return model, scaler, exercise_encoder, form_encoder

    def test_model_accepts_expected_input_shape(self, loaded_assets):
        model, scaler, _, _ = loaded_assets
        dummy = np.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        preds = model(dummy, training=False)
        assert len(preds) == 2, "Expected two output heads: exercise_output, form_output"

    def test_output_heads_are_valid_probability_distributions(self, loaded_assets):
        model, scaler, exercise_encoder, form_encoder = loaded_assets
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(1, SEQ_LEN, FEATURE_DIM)).astype(np.float32)
        scaled = scaler.transform(raw.reshape(-1, FEATURE_DIM)).reshape(1, SEQ_LEN, FEATURE_DIM).astype(np.float32)

        exercise_probs, form_probs = model(scaled, training=False)
        exercise_probs, form_probs = exercise_probs.numpy(), form_probs.numpy()

        assert exercise_probs.shape == (1, len(exercise_encoder.classes_))
        assert form_probs.shape == (1, len(form_encoder.classes_))
        assert np.isclose(exercise_probs.sum(), 1.0, atol=1e-4)
        assert np.isclose(form_probs.sum(), 1.0, atol=1e-4)

    def test_form_classes_are_good_and_bad(self, loaded_assets):
        """Sanity check against Table 3.1 in the thesis: form labels should be
        exactly 'good' / 'bad'."""
        _, _, _, form_encoder = loaded_assets
        assert set(form_encoder.classes_) == {"good", "bad"}

    def test_exercise_classes_match_three_target_exercises(self, loaded_assets):
        """Sanity check against thesis scope: only lat pulldown, seated cable
        row, and cable pullover are in scope -- no other exercise should ever
        be a possible prediction."""
        _, _, exercise_encoder, _ = loaded_assets
        expected = {"lat_pulldown", "seated_cable_row", "cable_pullover"}
        assert set(exercise_encoder.classes_) == expected

    def test_zero_input_is_deterministic(self, loaded_assets):
        """Warmup call + a second identical call should produce identical
        predictions -- guards against hidden statefulness (e.g. dropout not
        disabled at inference) that would make live classification unstable."""
        model, scaler, _, _ = loaded_assets
        dummy = np.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        preds1 = [p.numpy() for p in model(dummy, training=False)]
        preds2 = [p.numpy() for p in model(dummy, training=False)]
        for p1, p2 in zip(preds1, preds2):
            np.testing.assert_allclose(p1, p2)
