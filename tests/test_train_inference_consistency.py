"""
extract_features.py (offline training data generation) and
realtime_detector.py (live inference) both now build their tracking on the
shared OpticalFlowTracker (src/optical_flow_core.py) instead of each
maintaining an independent copy of the same logic. That independent-copies
setup previously caused a real divergence bug (a refresh-frame mismatch,
originally caught here by an approximate statistical comparison).

With a single shared implementation, the comparison can now be an exact
equivalence check instead of an approximation: given identical frame
sequences and identical parameters, the two pipelines' tracker instances
must produce bit-identical output. If someone reintroduces a
pipeline-specific override in either file, this is what catches it.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow", reason="realtime_detector.py imports tensorflow at module scope")

import extract_features as ef
import realtime_detector as rd
from realtime_detector import OpticalFlowExtractor
from optical_flow_core import OpticalFlowTracker


class TestSharedTrackerUsage:
    def test_realtime_extractor_subclasses_the_shared_tracker(self):
        assert issubclass(OpticalFlowExtractor, OpticalFlowTracker), (
            "realtime_detector.OpticalFlowExtractor must subclass the shared "
            "OpticalFlowTracker, not reimplement tracking logic independently."
        )

    def test_training_and_realtime_params_match(self):
        """extract_features.py and realtime_detector.py declare their own
        feature_params/lk_params constants (kept separate so each file stays
        readable on its own), but the values must be equal, or the two
        pipelines will silently diverge again despite sharing the tracker
        class itself."""
        assert ef.feature_params == rd.FEATURE_PARAMS
        assert ef.lk_params == rd.LK_PARAMS


class TestFeatureOutputEquivalence:
    def test_identical_frame_sequence_produces_identical_features(self, translating_square_sequence):
        train_tracker = OpticalFlowTracker(feature_params=ef.feature_params, lk_params=ef.lk_params)
        live_extractor = OpticalFlowExtractor()

        for frame in translating_square_sequence:
            train_features, _, _ = train_tracker.process_frame(frame)
            live_features, _, _ = live_extractor.process_frame(frame)
            np.testing.assert_array_equal(
                train_features, live_features,
                err_msg="Training-side and live-inference tracking diverged on an identical frame sequence.",
            )
