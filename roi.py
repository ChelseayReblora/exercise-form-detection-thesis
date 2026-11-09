"""ROI (Region of Interest) construction utilities.

Constrains optical-flow keypoint search to where the exerciser is expected
to be, instead of the full frame. This is what eliminates stray points
picked up from gym equipment, other people, or background clutter -- it does
not change the underlying Shi-Tomasi corner detection or Lucas-Kanade optical
flow algorithms, only where they are allowed to look.

Two masks are layered:
  1. A static ROI, derived from the thesis's fixed camera geometry (200cm
     distance, 100cm height, back-angled view) -- used as an outer bound and
     as the fallback when no tracked-point history exists yet.
  2. A tracked-point-history ROI, built from the bounding box of whatever is
     currently being tracked -- tightens the search area once tracking is
     established.
"""
import cv2
import numpy as np

# Fraction of frame width/height the exerciser is expected to occupy, given
# the thesis's fixed camera placement. Tuned to exclude frame edges (where
# gym equipment / background clutter tends to sit) while comfortably
# containing a centered subject.
STATIC_ROI_X_FRAC = (0.15, 0.85)
STATIC_ROI_Y_FRAC = (0.05, 1.0)

# Padding applied around a tracked-point bounding box before using it as the
# next frame's search region.
TRACKED_ROI_PAD_PX = 40


def build_static_roi(frame_shape, x_frac=STATIC_ROI_X_FRAC, y_frac=STATIC_ROI_Y_FRAC):
    """Fixed elliptical mask covering the region the exerciser is expected to
    occupy. Cheap and deterministic; used as a fallback and as the outer
    bound for the tracked-point ROI below."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, x1 = int(w * x_frac[0]), int(w * x_frac[1])
    y0, y1 = int(h * y_frac[0]), int(h * y_frac[1])
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    axes = (max((x1 - x0) // 2, 1), max((y1 - y0) // 2, 1))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
    return mask


def roi_from_tracked_points(frame_shape, points, pad_px=TRACKED_ROI_PAD_PX):
    """Bounding-box mask around currently tracked points, inflated by a
    margin. Returns None if there are no points to build a region from --
    callers should fall back to the static ROI in that case."""
    if points is None or len(points) == 0:
        return None

    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = cv2.boundingRect(points.reshape(-1, 1, 2).astype(np.float32))
    x0, y0 = max(x - pad_px, 0), max(y - pad_px, 0)
    x1, y1 = min(x + bw + pad_px, w), min(y + bh + pad_px, h)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    return mask


def combined_roi(frame_shape, tracked_points=None, static_mask=None):
    """Layers the tracked-point-history ROI inside the static ROI's outer
    bound. Falls back to the static ROI alone when no tracked history is
    available (first frame, or right after a full tracking loss)."""
    if static_mask is None:
        static_mask = build_static_roi(frame_shape)

    tracked_mask = roi_from_tracked_points(frame_shape, tracked_points)
    if tracked_mask is None:
        return static_mask

    return cv2.bitwise_and(static_mask, tracked_mask)


def points_inside_mask(points, mask):
    """Boolean array: which of `points` (shape (N,2) or (N,1,2)) fall inside
    the non-zero region of `mask`. Used to drop tracked points that drifted
    outside the current ROI instead of silently keeping them."""
    pts = points.reshape(-1, 2)
    h, w = mask.shape[:2]
    xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
    return mask[ys, xs] > 0
