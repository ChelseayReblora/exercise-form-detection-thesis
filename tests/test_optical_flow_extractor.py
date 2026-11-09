"""
Covers thesis objective #2: "detect the proper form ... using the Lucas-Kanade
optical flow method". These tests validate the tracking pipeline in isolation,
independent of the GRU model, using synthetic frames with known motion so the
expected optical flow output is known ahead of time.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow", reason="realtime_detector.py imports tensorflow at module scope")

from realtime_detector import OpticalFlowExtractor, FEATURE_DIM  # noqa: E402


class TestFirstFrame:
    def test_first_frame_returns_zero_features_and_no_points(self, translating_square_sequence):
        extractor = OpticalFlowExtractor()
        features, good_new, good_old = extractor.process_frame(translating_square_sequence[0])

        assert good_new is None
        assert good_old is None
        np.testing.assert_array_equal(features, np.zeros(FEATURE_DIM, dtype=np.float32))


class TestTrackingOnKnownMotion:
    def test_points_are_tracked_across_frames(self, translating_square_sequence):
        """Basic regression test for the 'walang nattrack na points' bug:
        after warming up on frame 1, every subsequent frame on a textured,
        moving scene should produce a non-empty set of tracked points."""
        extractor = OpticalFlowExtractor()

        for frame in translating_square_sequence:
            features, good_new, good_old = extractor.process_frame(frame)

        # Only check the tail of the sequence -- give the tracker a couple frames to warm up.
        extractor = OpticalFlowExtractor()
        tracked_counts = []
        for frame in translating_square_sequence:
            _, good_new, _ = extractor.process_frame(frame)
            tracked_counts.append(0 if good_new is None else len(good_new))

        # First frame has no prior frame to compare against -- expected to be 0.
        assert tracked_counts[0] == 0
        # From frame 2 onward, points should consistently be tracked.
        assert all(c > 0 for c in tracked_counts[1:]), (
            f"Expected nonzero tracked points from frame 2 onward, got {tracked_counts}"
        )

    def test_motion_direction_matches_known_translation(self, translating_square_sequence):
        """The square moves +2px in x and +1px in y every frame. Mean dx/dy in the
        extracted features should reflect that direction (allowing for LK noise)."""
        extractor = OpticalFlowExtractor()
        last_features = None
        for frame in translating_square_sequence:
            last_features, _, _ = extractor.process_frame(frame)

        mean_dx, mean_dy = last_features[0], last_features[1]
        assert mean_dx > 0, "Expected positive mean x-displacement for rightward motion"
        assert mean_dy > 0, "Expected positive mean y-displacement for downward motion"

    def test_refresh_baseline_is_never_stale(self, translating_square_sequence, monkeypatch):
        """Regression test for the exact bug found in code review: when points are
        refreshed via goodFeaturesToTrack, they must be baselined against the SAME
        frame that produced them, not matched against a stale self.old_gray from
        the previous iteration. This test forces a refresh mid-sequence (by
        clearing old_points) and asserts self.old_gray is updated to the frame
        the new points were detected on before any subsequent LK call uses it."""
        extractor = OpticalFlowExtractor()

        # warm up for a couple frames
        extractor.process_frame(translating_square_sequence[0])
        extractor.process_frame(translating_square_sequence[1])

        # Force an artificial refresh condition (simulates points being lost,
        # e.g. after occlusion or fast motion in a real gym video).
        extractor.old_points = None
        frame_that_triggers_refresh = translating_square_sequence[2]
        extractor.process_frame(frame_that_triggers_refresh)

        # After the refresh, old_gray must be the SAME frame the new keypoints
        # were detected on -- otherwise the next LK call compares points against
        # the wrong image and tracking silently collapses.
        np.testing.assert_array_equal(extractor.old_gray, frame_that_triggers_refresh)


class TestNoMotionOrNoTexture:
    def test_blank_frames_yield_no_points(self, blank_sequence):
        """goodFeaturesToTrack should find zero corners on a textureless frame;
        the extractor must degrade gracefully (zero features, no crash)."""
        extractor = OpticalFlowExtractor()
        for frame in blank_sequence:
            features, good_new, good_old = extractor.process_frame(frame)

        assert good_new is None or len(good_new) == 0
        np.testing.assert_array_equal(features, np.zeros(FEATURE_DIM, dtype=np.float32))

    def test_recovers_once_texture_appears(self, blank_sequence, translating_square_sequence):
        """If the scene starts blank (e.g. camera still focusing) and then a
        textured moving subject enters frame, tracking should pick up without
        needing a restart."""
        extractor = OpticalFlowExtractor()
        for frame in blank_sequence[:5]:
            extractor.process_frame(frame)

        tracked_counts = []
        for frame in translating_square_sequence:
            _, good_new, _ = extractor.process_frame(frame)
            tracked_counts.append(0 if good_new is None else len(good_new))

        assert any(c > 0 for c in tracked_counts), "Tracker never recovered once texture appeared"


class TestFeatureVectorShape:
    def test_feature_vector_always_length_8(self, translating_square_sequence, blank_sequence):
        extractor = OpticalFlowExtractor()
        for frame in translating_square_sequence + blank_sequence:
            features, _, _ = extractor.process_frame(frame)
            assert features.shape == (FEATURE_DIM,)
            assert features.dtype == np.float32


class TestRoiIsolation:
    def test_background_clutter_outside_roi_is_never_tracked(self):
        """Simulates gym equipment / background clutter sitting near the frame
        edge (outside the default static ROI, x <= 48px at FRAME_WIDTH=320)
        alongside a moving, textured subject nearer frame center. Only the
        subject's points should ever be tracked -- this is the direct
        end-to-end check that ROI isolation actually keeps stray points out."""
        extractor = OpticalFlowExtractor()

        def make_frame(cx, cy):
            frame = np.zeros((240, 320), dtype=np.uint8)
            # "exerciser" -- textured, moving, inside the default ROI
            y0, y1, x0, x1 = cy - 40, cy + 40, cx - 40, cx + 40
            patch = frame[y0:y1, x0:x1]
            ys, xs = np.indices(patch.shape)
            frame[y0:y1, x0:x1] = ((((xs // 8) + (ys // 8)) % 2) * 255).astype(np.uint8)
            # "background clutter" -- static, high-contrast, at the frame
            # edge (outside the default ROI's x_frac=(0.15, 0.85))
            frame[0:40, 0:40] = 255
            return frame

        last_good_new = None
        for i in range(20):
            frame = make_frame(140 + i * 2, 100 + i * 1)
            _, good_new, _ = extractor.process_frame(frame)
            if good_new is not None and len(good_new) > 0:
                last_good_new = good_new

        assert last_good_new is not None, "Tracker never picked up the subject"
        xs = last_good_new[:, 0]
        assert np.all(xs > 48), (
            f"Tracked points found in the excluded clutter region (x<=48): {xs}"
        )
