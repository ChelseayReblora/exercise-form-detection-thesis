"""
Covers the SEQ_LEN=60 rolling buffer that feeds the GRU (RealtimeDetector.run,
'Buffer Management' section). Extracted here as a standalone shift/insert
routine so it can be tested without spinning up a camera or TF model.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="realtime_detector.py imports tensorflow at module scope")

from realtime_detector import SEQ_LEN, FEATURE_DIM


def shift_and_insert(buffer, new_features):
    """Mirrors RealtimeDetector.run()'s buffer update:
        self.feature_buffer[:-1] = self.feature_buffer[1:]
        self.feature_buffer[-1] = features
    """
    buffer = buffer.copy()
    buffer[:-1] = buffer[1:]
    buffer[-1] = new_features
    return buffer


class TestRollingBuffer:
    def test_buffer_is_fifo(self):
        buffer = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for i in range(SEQ_LEN):
            buffer = shift_and_insert(buffer, np.full(FEATURE_DIM, i, dtype=np.float32))

        # After SEQ_LEN inserts, buffer[0] should hold the oldest value (0),
        # buffer[-1] the most recent (SEQ_LEN - 1).
        assert buffer[0, 0] == 0
        assert buffer[-1, 0] == SEQ_LEN - 1
        # Values should be strictly increasing top to bottom.
        assert np.all(np.diff(buffer[:, 0]) == 1)

    def test_buffer_overflow_drops_oldest(self):
        buffer = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        for i in range(SEQ_LEN + 10):
            buffer = shift_and_insert(buffer, np.full(FEATURE_DIM, i, dtype=np.float32))

        # Only the most recent SEQ_LEN values should remain: 10 .. SEQ_LEN+9
        assert buffer[0, 0] == 10
        assert buffer[-1, 0] == SEQ_LEN + 9

    def test_frame_count_caps_at_seq_len(self):
        frame_count = 0
        for _ in range(SEQ_LEN + 20):
            frame_count = min(frame_count + 1, SEQ_LEN)
        assert frame_count == SEQ_LEN

    def test_inference_only_triggers_once_buffer_full(self):
        """RealtimeDetector only runs model inference once frame_count >= SEQ_LEN
        (and every 3rd frame after that) -- verify that gating condition here
        since a broken buffer count would either delay classification
        indefinitely or run inference on a buffer still full of zeros."""
        results = []
        frame_count = 0
        for _ in range(SEQ_LEN + 10):
            frame_count = min(frame_count + 1, SEQ_LEN)
            should_infer = frame_count >= SEQ_LEN and frame_count % 3 == 0
            results.append(should_infer)

        assert not any(results[: SEQ_LEN - 1]), "Inference triggered before buffer was full"
        assert any(results[SEQ_LEN - 1 :]), "Inference never triggered once buffer was full"
