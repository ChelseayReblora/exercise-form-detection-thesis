import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

MODEL_DIR = Path(__file__).resolve().parent.parent / "back_exercise_form_detector" / "models"

SEQ_LEN = 60
FEATURE_DIM = 8
FRAME_WIDTH = 320
FRAME_HEIGHT = 240


def make_translating_frame(width, height, cx, cy, size=40, value=255):
    """A single synthetic grayscale frame: a checkerboard patch on a black
    background, centered at (cx, cy). A checkerboard (vs. a plain square) gives
    goodFeaturesToTrack enough distinct corners (>= 10) to detect and track --
    a plain solid square only has 4 corners, which is below the extractor's
    refresh threshold and never actually exercises steady-state tracking.
    """
    frame = np.zeros((height, width), dtype=np.uint8)
    x0, x1 = max(cx - size, 0), min(cx + size, width)
    y0, y1 = max(cy - size, 0), min(cy + size, height)
    patch = frame[y0:y1, x0:x1]
    cell = 8
    ys, xs = np.indices(patch.shape)
    checker = (((xs // cell) + (ys // cell)) % 2) * value
    frame[y0:y1, x0:x1] = checker.astype(np.uint8)
    return frame


@pytest.fixture
def translating_square_sequence():
    """20 frames of a square moving diagonally at a constant velocity (dx=2, dy=1
    per frame). Positioned to stay within the default static ROI (roi.py's
    STATIC_ROI_X_FRAC/Y_FRAC) for the full trajectory, so these tests exercise
    steady-state tracking rather than incidentally clipping against the ROI
    boundary."""
    n_frames = 20
    frames = []
    for i in range(n_frames):
        cx = 140 + i * 2
        cy = 100 + i * 1
        frames.append(make_translating_frame(FRAME_WIDTH, FRAME_HEIGHT, cx, cy))
    return frames


@pytest.fixture
def static_noise_sequence():
    """20 frames of pure random noise -- no coherent motion, texture present so
    corners *can* be found, but nothing should track consistently frame to frame."""
    rng = np.random.default_rng(42)
    return [
        rng.integers(0, 255, size=(FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        for _ in range(20)
    ]


@pytest.fixture
def blank_sequence():
    """20 completely flat frames -- no texture at all, goodFeaturesToTrack should
    find zero corners. Used to test the 'no points detected' failure path."""
    return [np.full((FRAME_HEIGHT, FRAME_WIDTH), 128, dtype=np.uint8) for _ in range(20)]


def model_assets_available():
    required = ["gru_model.keras", "scaler.pkl", "exercise_encoder.pkl", "form_encoder.pkl"]
    return all((MODEL_DIR / f).exists() for f in required)


requires_model = pytest.mark.skipif(
    not model_assets_available(),
    reason="Trained model assets not found in back_exercise_form_detector/models/",
)

def _tensorflow_available():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


requires_tensorflow = pytest.mark.skipif(
    not _tensorflow_available(), reason="tensorflow not installed"
)
