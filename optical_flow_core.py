"""Shared optical-flow point tracking + motion feature extraction.

Used identically by both extract_features.py (offline training data
generation) and realtime_detector.py (live inference), so the two pipelines
can no longer silently diverge -- previously each maintained its own copy of
this logic, and the two copies drifted apart (see git history / the
train/inference consistency test).

The underlying algorithms are unchanged: Shi-Tomasi corner detection
(cv2.goodFeaturesToTrack) seeds points, Lucas-Kanade pyramidal optical flow
(cv2.calcOpticalFlowPyrLK) tracks them frame to frame. This module adds an
ROI layer (see roi.py) that constrains where those calls are allowed to
search and which resulting points are kept, to avoid tracking stray points
on background clutter.
"""
import cv2
import numpy as np

from roi import build_static_roi, combined_roi, points_inside_mask

FEATURE_DIM = 16  # 8 global stats + 4 upper-region stats + 4 lower-region stats

FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Below this, tracking is too sparse to trust at all -- do a full reset and
# skip this frame (same as the old behavior). Above this but below
# REPLENISH_POINTS, keep tracking with what we have (real motion data) while
# topping up new corners in the background, instead of throwing the frame
# away. This is what stops the feature sequence from filling up with
# fake "no motion" frames every time the point count dips.
MIN_POINTS_HARD_FLOOR = 4
REPLENISH_BELOW = 20

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def _filter_by_status(old_points, new_points, status):
    """Applies the LK status mask, dropping OpenCV's redundant middle
    dimension so callers get plain (K, 2) arrays instead of (K, 1, 2)."""
    if old_points is None or new_points is None or status is None:
        return None, None

    status = status.reshape(-1)
    good_new = new_points[status == 1].reshape(-1, 2)
    good_old = old_points[status == 1].reshape(-1, 2)

    if len(good_new) == 0:
        return None, None

    return good_new, good_old


def _region_stats(good_old, good_new, region_mask):
    """4 stats (mean dx, mean dy, mean magnitude, point count) for the
    subset of points whose OLD position falls inside region_mask. Returns
    zeros if no points fall in this region this frame -- a region having no
    tracked points is itself informative (that body part isn't moving /
    isn't visible), not an error."""
    if good_new is None or region_mask is None or not np.any(region_mask):
        return np.zeros(4, dtype=np.float32)

    new_r = good_new[region_mask]
    old_r = good_old[region_mask]
    displacement = new_r - old_r
    dx = displacement[:, 0]
    dy = displacement[:, 1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    return np.array([
        np.mean(dx),
        np.mean(dy),
        np.mean(magnitude),
        len(new_r)
    ], dtype=np.float32)


def _feature_stats(good_old, good_new, frame_height=None):
    """Extracts the 16 motion features required by the GRU model: 8 global
    pooled stats (as before) plus 4+4 stats split by whether each point's
    OLD position was in the upper or lower half of the frame.

    The global-only version of this function pooled every tracked point
    together, which meant a subtle, localized form deviation (e.g. only the
    lower body swaying) got averaged away by all the other, unrelated
    motion in the same frame. Splitting by region lets the model actually
    see *where* the deviation is, not just an aggregate "how much motion
    overall" signal.

    frame_height is required to compute the region split; if not provided
    (e.g. a caller that hasn't been updated), the region stats are left as
    zeros and only the global 8 stats carry real information -- shape stays
    correct either way so this fails safe rather than crashing.
    """
    if good_new is None or len(good_new) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    displacement = good_new - good_old
    dx = displacement[:, 0]
    dy = displacement[:, 1]

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    direction = np.arctan2(dy, dx)

    global_stats = np.array([
        np.mean(dx),
        np.mean(dy),
        np.mean(magnitude),
        np.mean(direction),
        len(good_new),
        np.std(dx),
        np.std(dy),
        np.std(magnitude)
    ], dtype=np.float32)

    if frame_height is None:
        return np.concatenate([global_stats, np.zeros(8, dtype=np.float32)])

    midline = frame_height / 2.0
    upper_mask = good_old[:, 1] < midline
    lower_mask = ~upper_mask

    upper_stats = _region_stats(good_old, good_new, upper_mask)
    lower_stats = _region_stats(good_old, good_new, lower_mask)

    return np.concatenate([global_stats, upper_stats, lower_stats])


class OpticalFlowTracker:
    """Stateful Lucas-Kanade point tracker with ROI-constrained keypoint
    search. Both the training feature extractor and the live detector
    instantiate this class rather than reimplementing tracking themselves."""

    def __init__(self, feature_params=None, lk_params=None):
        self.feature_params = feature_params or FEATURE_PARAMS
        self.lk_params = lk_params or LK_PARAMS
        self.old_gray = None
        self.old_points = None
        self.static_roi = None

    def _search_mask(self, frame_shape, tracked_points=None):
        if self.static_roi is None:
            self.static_roi = build_static_roi(frame_shape)
        if tracked_points is None:
            tracked_points = self.old_points
        return combined_roi(frame_shape, tracked_points=tracked_points, static_mask=self.static_roi)

    def _replenish_points(self, frame_gray, existing_points):
        """Tops up existing tracked points with freshly detected corners,
        instead of discarding them. Suppresses the search mask around points
        we're already tracking so goodFeaturesToTrack contributes new ones
        rather than re-finding the same corners."""
        needed = self.feature_params.get('maxCorners', 100) - len(existing_points)
        if needed <= 0:
            return existing_points

        mask = self._search_mask(frame_gray.shape, tracked_points=existing_points)
        exclusion_radius = self.feature_params.get('minDistance', 7)
        for (x, y) in existing_points.reshape(-1, 2):
            cv2.circle(mask, (int(x), int(y)), exclusion_radius, 0, -1)

        topup_params = dict(self.feature_params)
        topup_params['maxCorners'] = needed
        new_points = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **topup_params)

        if new_points is None:
            return existing_points

        return np.vstack([existing_points.reshape(-1, 1, 2), new_points.reshape(-1, 1, 2)])

    def process_frame(self, frame_gray):
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        good_new = None
        good_old = None

        # Only a full reset + skipped frame when tracking is critically sparse
        # (or this is the very first frame). Between the hard floor and
        # REPLENISH_BELOW, we still compute real optical flow below and just
        # top up afterward -- see the replenish call further down.
        if self.old_points is None or len(self.old_points) < MIN_POINTS_HARD_FLOOR:
            mask = self._search_mask(frame_gray.shape)
            self.old_points = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=mask,
                **self.feature_params
            )
            self.old_gray = frame_gray.copy()
            return features, good_new, good_old

        if self.old_gray is not None:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.old_gray,
                frame_gray,
                self.old_points,
                None,
                **self.lk_params
            )

            try:
                good_new, good_old = _filter_by_status(self.old_points, new_points, status)

                if good_new is not None and len(good_new) > 0:
                    mask = self._search_mask(frame_gray.shape, tracked_points=good_new)
                    inside = points_inside_mask(good_new, mask)
                    good_new, good_old = good_new[inside], good_old[inside]
                    if len(good_new) == 0:
                        good_new, good_old = None, None

                features = _feature_stats(good_old, good_new, frame_height=frame_gray.shape[0])

            except Exception as e:
                print(f"Optical Flow Error: {e}")
                good_new, good_old = None, None

            if good_new is not None and len(good_new) > 0:
                self.old_points = good_new.reshape(-1, 1, 2)
                if len(self.old_points) < REPLENISH_BELOW:
                    self.old_points = self._replenish_points(frame_gray, self.old_points)
            else:
                self.old_points = None
                good_new, good_old = None, None

        self.old_gray = frame_gray.copy()

        return features, good_new, good_old


def pad_or_trim(sequence, seq_len=60):
    """Makes a variable-length sequence of per-frame feature vectors exactly
    seq_len frames long. Shared by extract_features.py (whole pre-recorded
    rep videos) and realtime_detector.py (live, rep-segmented sequences) so
    the two preprocessing paths cannot silently diverge on this either.

    Shorter sequences are padded by repeating the last real frame's
    features, rather than zeros. Zero-padding tells the model "everything
    suddenly stopped existing", which is a fabricated signal that was never
    part of the actual rep. Repeating the last frame reads as "held the end
    position", which is at least a plausible real state and doesn't
    introduce fake stillness into the sequence.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) == 0:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)

    if len(sequence) < seq_len:
        last_frame = sequence[-1:]
        padding = np.repeat(last_frame, seq_len - len(sequence), axis=0)
        sequence = np.vstack([sequence, padding])
    elif len(sequence) > seq_len:
        indices = np.linspace(0, len(sequence) - 1, seq_len).astype(int)
        sequence = sequence[indices]

    return sequence
