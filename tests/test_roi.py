"""
Covers the ROI (Region of Interest) isolation layer that constrains
optical-flow keypoint search to where the exerciser is expected to be,
instead of the full frame -- this is what addresses stray points picked up
from background gym equipment / clutter. Does not touch the underlying
Shi-Tomasi / Lucas-Kanade algorithms, only where they're allowed to search
and which resulting points are kept.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from roi import build_static_roi, roi_from_tracked_points, combined_roi, points_inside_mask

FRAME_SHAPE = (240, 320)


class TestBuildStaticRoi:
    def test_returns_binary_mask_of_frame_size(self):
        mask = build_static_roi(FRAME_SHAPE)
        assert mask.shape == FRAME_SHAPE
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_excludes_frame_corners(self):
        """Gym equipment / background clutter tends to sit at frame edges
        given the thesis's fixed camera framing -- the static ROI should
        exclude the extreme corners."""
        mask = build_static_roi(FRAME_SHAPE)
        assert mask[0, 0] == 0
        assert mask[0, -1] == 0
        assert mask[-1, 0] == 0
        assert mask[-1, -1] == 0

    def test_includes_frame_center(self):
        mask = build_static_roi(FRAME_SHAPE)
        h, w = FRAME_SHAPE
        assert mask[h // 2, w // 2] == 255


class TestRoiFromTrackedPoints:
    def test_none_when_no_points(self):
        assert roi_from_tracked_points(FRAME_SHAPE, None) is None
        assert roi_from_tracked_points(FRAME_SHAPE, np.zeros((0, 2))) is None

    def test_bounds_tightly_around_points_with_padding(self):
        points = np.array([[100, 100], [150, 120], [120, 140]], dtype=np.float32)
        mask = roi_from_tracked_points(FRAME_SHAPE, points, pad_px=10)
        assert mask[120, 125] == 255  # inside the point cluster
        assert mask[0, 0] == 0        # far outside


class TestCombinedRoi:
    def test_falls_back_to_static_when_no_tracked_points(self):
        static_mask = build_static_roi(FRAME_SHAPE)
        combined = combined_roi(FRAME_SHAPE, tracked_points=None, static_mask=static_mask)
        np.testing.assert_array_equal(combined, static_mask)

    def test_tightens_search_area_when_points_available(self):
        static_mask = build_static_roi(FRAME_SHAPE)
        points = np.array([[150, 100], [160, 110]], dtype=np.float32)
        combined = combined_roi(FRAME_SHAPE, tracked_points=points, static_mask=static_mask)
        assert combined.sum() <= static_mask.sum()

    def test_never_exceeds_the_static_outer_bound(self):
        """Even if the tracked-point bounding box is huge (e.g. a tracking
        glitch flings a point to the frame edge), the combined ROI must stay
        within the static ROI's outer bound."""
        static_mask = build_static_roi(FRAME_SHAPE)
        points = np.array([[0, 0], [319, 239]], dtype=np.float32)
        combined = combined_roi(FRAME_SHAPE, tracked_points=points, static_mask=static_mask)
        assert np.all(combined[static_mask == 0] == 0)


class TestPointsInsideMask:
    def test_flags_points_correctly(self):
        mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        mask[50:100, 50:100] = 255
        points = np.array([[60, 60], [10, 10], [70, 90]], dtype=np.float32)
        inside = points_inside_mask(points, mask)
        np.testing.assert_array_equal(inside, [True, False, True])
