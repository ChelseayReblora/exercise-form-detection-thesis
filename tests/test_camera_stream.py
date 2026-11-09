"""
Covers thesis objective #1: "capture real-time back muscle exercise videos
using the MakerFocus Camera Module". Since CI/dev machines won't have a
Raspberry Pi or that camera attached, these tests fake the picamera2 module
to validate the capture + color-conversion logic without real hardware, and
validate the cv2.VideoCapture fallback path used for Mac/dev testing.
"""
import sys
import types

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow", reason="realtime_detector.py imports tensorflow at module scope")


def _install_fake_picamera2(monkeypatch, frame_bgrx):
    """Installs a fake `picamera2` module in sys.modules so CameraStream takes
    the Picamera2 branch without needing real hardware."""

    class FakePicamera2:
        def __init__(self):
            self.started = False

        def create_preview_configuration(self, main):
            return {"main": main}

        def configure(self, config):
            self.config = config

        def start(self):
            self.started = True

        def set_controls(self, controls):
            self.controls = controls

        def capture_array(self, name):
            return frame_bgrx

        def stop(self):
            self.started = False

    fake_module = types.ModuleType("picamera2")
    fake_module.Picamera2 = FakePicamera2
    monkeypatch.setitem(sys.modules, "picamera2", fake_module)


class TestPicamera2ColorHandling:
    def test_xrgb8888_frame_is_not_channel_swapped(self, monkeypatch):
        """Regression test for the 'hindi normal ang display' bug.

        Picamera2's 'XRGB8888' format returns arrays already ordered (B, G, R, X)
        in memory. A correctly implemented `read()` must produce a frame whose
        blue channel matches the camera's blue channel, not the red channel.
        """
        # Simulate a camera array in Picamera2's actual XRGB8888 memory layout: (B, G, R, X)
        h, w = 10, 10
        fake_bgrx = np.zeros((h, w, 4), dtype=np.uint8)
        fake_bgrx[:, :, 0] = 10   # B
        fake_bgrx[:, :, 1] = 20   # G
        fake_bgrx[:, :, 2] = 200  # R
        fake_bgrx[:, :, 3] = 255  # X / alpha (unused)

        _install_fake_picamera2(monkeypatch, fake_bgrx)

        # Reload realtime_detector so it picks up the fake picamera2 module.
        import importlib
        import realtime_detector
        importlib.reload(realtime_detector)

        camera = realtime_detector.CameraStream(width=w, height=h)
        assert camera.use_picamera2 is True

        ret, frame_bgr = camera.read()
        assert ret is True

        b, g, r = frame_bgr[0, 0]
        assert b == 10, f"Blue channel corrupted: expected 10, got {b} (likely R/B swap bug)"
        assert g == 20
        assert r == 200, f"Red channel corrupted: expected 200, got {r} (likely R/B swap bug)"


class TestCv2VideoCaptureFallback:
    def test_falls_back_when_picamera2_missing(self, monkeypatch):
        """On a Mac (no picamera2 installed), CameraStream must transparently
        fall back to cv2.VideoCapture rather than raising."""
        monkeypatch.setitem(sys.modules, "picamera2", None)  # simulate ImportError

        import importlib
        import realtime_detector
        importlib.reload(realtime_detector)

        class FakeCapture:
            def __init__(self, *a, **kw):
                self._opened = True

            def isOpened(self):
                return self._opened

            def set(self, *a, **kw):
                pass

            def read(self):
                return True, np.zeros((240, 320, 3), dtype=np.uint8)

            def release(self):
                self._opened = False

        monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: FakeCapture())

        camera = realtime_detector.CameraStream(width=320, height=240)
        assert camera.use_picamera2 is False

        ret, frame = camera.read()
        assert ret is True
        assert frame.shape == (240, 320, 3)

    def test_resizes_mismatched_frame_sizes(self, monkeypatch):
        """If the webcam driver ignores the requested resolution, read() must
        resize the frame so downstream code (fixed-size feature buffer,
        FRAME_WIDTH/HEIGHT-based overlay drawing) doesn't break."""
        monkeypatch.setitem(sys.modules, "picamera2", None)

        import importlib
        import realtime_detector
        importlib.reload(realtime_detector)

        class OddSizeCapture:
            def isOpened(self):
                return True

            def set(self, *a, **kw):
                pass

            def read(self):
                return True, np.zeros((480, 640, 3), dtype=np.uint8)  # wrong size

            def release(self):
                pass

        monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: OddSizeCapture())

        camera = realtime_detector.CameraStream(width=320, height=240)
        ret, frame = camera.read()
        assert ret is True
        assert frame.shape == (240, 320, 3)
