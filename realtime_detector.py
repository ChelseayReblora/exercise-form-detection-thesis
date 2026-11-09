import os
import sys
import time
import cv2
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optical_flow_core import OpticalFlowTracker, pad_or_trim, FEATURE_DIM

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# ==============================================================================
# CONFIGURATION AND PARAMETERS
# ==============================================================================

# Camera resolution
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Sequence properties
SEQ_LEN = 60

# Rep-segmentation thresholds (time-based, not frame-count-based, so this
# behaves consistently regardless of the actual achieved FPS on a given
# machine -- a Raspberry Pi and a laptop will process frames at very
# different rates).
MOTION_ON_THRESHOLD = 0.6    # mean flow magnitude above this counts as "moving"
MOTION_OFF_THRESHOLD = 0.4   # lower than ON (hysteresis) so we don't flicker at the boundary
MOTION_ON_SECONDS = 0.15     # sustained motion required before a rep is confirmed to have started
MOTION_OFF_SECONDS = 0.6     # sustained stillness required before a rep is confirmed to have ended
MIN_REP_SECONDS = 0.5        # shorter bursts are treated as noise, not a rep, and discarded
PREROLL_SECONDS = 0.5        # frames kept before motion is confirmed, so the true start of the rep isn't cut off
MAX_REP_SECONDS = 15.0       # safety valve: finalize anyway if motion never stops (stuck tracker, etc.)

# Shi-Tomasi keypoint detection parameters
FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# ==============================================================================
# HARDWARE / CAMERA ABSTRACTION
# ==============================================================================

class CameraStream:
    """Handles camera connection with Picamera2 (preferred) or cv2.VideoCapture fallback.
    If video_path is given, skips hardware entirely and reads from that file instead --
    lets the full pipeline be tested against a recorded clip with no camera/Pi needed."""
    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT, video_path=None):
        self.width = width
        self.height = height
        self.use_picamera2 = False
        self.picam2 = None
        self.cap = None
        self.is_video_file = False

        if video_path is not None:
            self._init_cv2(video_path)
            return

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": "XRGB8888"
                }
            )
            self.picam2.configure(config)
            self.picam2.start()
            self.picam2.set_controls({
                "AwbEnable": True,
                "ColourGains": (2.0, 2.0)
})
            self.use_picamera2 = True
            print("Successfully initialized Picamera2.")
        except ImportError:
            print("Picamera2 not found. Falling back to cv2.VideoCapture(0).")
            self._init_cv2()
        except Exception as e:
            print(f"Failed to initialize Picamera2: {e}. Falling back to cv2.VideoCapture(0).")
            self._init_cv2()

    def _init_cv2(self, video_path=None):
        self.is_video_file = video_path is not None
        self.cap = cv2.VideoCapture(video_path if video_path is not None else 0)
        if not self.cap.isOpened():
            source = video_path if video_path is not None else "camera index 0"
            raise RuntimeError(f"Failed to open {source} via OpenCV.")
        if not self.is_video_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            # Video files decode as fast as the CPU allows -- with no throttle,
            # a short clip finishes in well under a second and the window never
            # renders anything a human can actually see. Cap it to the source's
            # own frame rate (falls back to 30 if the file doesn't report one).
            source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_fps = source_fps if source_fps and source_fps > 1 else 30.0

    def read(self):
        """Returns (ret, frame). Frame is in BGR format for OpenCV compatibility."""
        if self.use_picamera2:
            try:
                frame_rgb = self.picam2.capture_array('main')
                # Picamera2's "XRGB8888" format returns arrays already ordered (B, G, R, X)
                # in memory -- just drop the alpha channel, don't re-swap R/B.
                # FIX: Add .copy() to ensure a contiguous memory allocation
                frame_bgr = frame_rgb[:, :, :3].copy()
                return True, frame_bgr
            except Exception as e:
                print(f"Picamera2 capture error: {e}")
                return False, None
        else:
            ret, frame = self.cap.read()
            if ret and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                frame = cv2.resize(frame, (self.width, self.height))
            return ret, frame

    def release(self):
        if self.use_picamera2:
            self.picam2.stop()
        else:
            if self.cap:
                self.cap.release()

# ==============================================================================
# OPTICAL FLOW FEATURE EXTRACTOR
# ==============================================================================

class OpticalFlowExtractor(OpticalFlowTracker):
    """Shi-Tomasi + Lucas-Kanade point tracking, constrained to an ROI to
    avoid tracking stray points on background clutter. Thin wrapper around
    the shared OpticalFlowTracker (src/optical_flow_core.py) so the live
    inference pipeline and the offline training feature extractor
    (extract_features.py) cannot silently diverge -- see git history for the
    bug this caused when the two were separate implementations."""
    def __init__(self):
        super().__init__(feature_params=FEATURE_PARAMS, lk_params=LK_PARAMS)


# ==============================================================================
# MAIN DETECTOR APPLICATION
# ==============================================================================

class RealtimeDetector:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        
        # Dependencies paths
        self.model_path = self.model_dir / "gru_model.keras"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.ex_enc_path = self.model_dir / "exercise_encoder.pkl"
        self.form_enc_path = self.model_dir / "form_encoder.pkl"

        # Hardware/Model variables
        self.camera = None
        self.model = None
        self.scaler = None
        self.exercise_encoder = None
        self.form_encoder = None

        # Rep-segmentation state (replaces the old fixed sliding window --
        # see MOTION_ON_THRESHOLD etc. above). We only run inference once a
        # full rep (motion start -> motion stop) has been captured, instead
        # of continuously re-predicting on a rolling window of whatever
        # happened in the last N seconds.
        self.preroll = deque()       # (timestamp, features) kept before a rep is confirmed
        self.rep_buffer = []         # features accumulated during the current rep
        self.rep_state = "idle"      # "idle" or "in_rep"
        self.motion_on_since = None
        self.motion_off_since = None
        self.rep_start_time = None
        self.has_result = False      # becomes True after the first completed rep

        # Output states
        self.current_exercise = "Waiting for first rep..."
        self.current_form = "Waiting for first rep..."
        self.exercise_conf = 0.0
        self.form_conf = 0.0
        self.exec_speed = "Normal"
        
        self.extractor = OpticalFlowExtractor()

    def _load_assets(self):
        """Loads TF model, Scaler, and Encoders."""
        try:
            print("Loading GRU model...")
            self.model = tf.keras.models.load_model(self.model_path)
            
            print("Loading preprocessing objects...")
            self.scaler = joblib.load(self.scaler_path)
            self.exercise_encoder = joblib.load(self.ex_enc_path)
            self.form_encoder = joblib.load(self.form_enc_path)
            
            # Warmup the model with dummy data
            dummy_input = np.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
            self.model(dummy_input, training=False)
            print("Assets loaded and model warmed up successfully.")
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print("Please ensure gru_model.keras, scaler.pkl, exercise_encoder.pkl, and form_encoder.pkl exist in the 'models' directory.")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during initialization: {e}")
            sys.exit(1)

    def _classify_speed(self, mean_magnitude):
        """Classifies execution speed based on optical flow magnitude."""
        if mean_magnitude < 0.5:
            return "Slow"
        elif mean_magnitude > 4.0:
            return "Fast"
        return "Normal"

    def _update_rep_state(self, features, now):
        """Feeds one frame's features into the rep state machine. Returns
        the finished rep's feature list once a full rep has been detected
        (motion started, then sustained stillness confirms it ended), or
        None if we're still mid-rep or idle."""
        magnitude = features[2]
        moving = magnitude > (MOTION_ON_THRESHOLD if self.rep_state == "idle" else MOTION_OFF_THRESHOLD)

        # Keep a short rolling pre-roll at all times so that once a rep is
        # confirmed, we can seed the rep buffer with the frames leading up
        # to it -- otherwise the ~0.15s spent confirming motion has truly
        # started would be silently cut off the front of every rep.
        self.preroll.append((now, features))
        while self.preroll and now - self.preroll[0][0] > PREROLL_SECONDS:
            self.preroll.popleft()

        if self.rep_state == "idle":
            self.motion_off_since = None
            if moving:
                if self.motion_on_since is None:
                    self.motion_on_since = now
                elif now - self.motion_on_since >= MOTION_ON_SECONDS:
                    self.rep_state = "in_rep"
                    self.rep_buffer = [f for _, f in self.preroll]
                    self.rep_start_time = self.preroll[0][0] if self.preroll else now
                    self.motion_on_since = None
            else:
                self.motion_on_since = None
            return None

        # state == "in_rep"
        self.rep_buffer.append(features)
        if (now - self.rep_start_time) > MAX_REP_SECONDS:
            return self._finish_rep(now)

        if not moving:
            if self.motion_off_since is None:
                self.motion_off_since = now
            elif now - self.motion_off_since >= MOTION_OFF_SECONDS:
                return self._finish_rep(now)
        else:
            self.motion_off_since = None

        return None

    def _finish_rep(self, now):
        """Closes out the current rep. Returns the frame list if it's long
        enough to be a real rep, or None if it was too brief and gets
        discarded as noise (a twitch, someone walking past, etc.)."""
        frames = self.rep_buffer
        duration = now - self.rep_start_time if self.rep_start_time else 0.0

        self.rep_state = "idle"
        self.rep_buffer = []
        self.motion_off_since = None
        self.motion_on_since = None

        if duration < MIN_REP_SECONDS or len(frames) < 3:
            print(f"Discarded short motion burst ({duration:.2f}s, {len(frames)} frames) -- likely noise, not a rep.")
            return None

        return frames

    def _draw_overlay(self, frame, good_new, good_old, fps):
        """Renders tracking visualization and predictions onto the frame."""
        # Visualization: Draw optical flow tracks
        if good_new is not None and good_old is not None:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Draw motion vector line
                frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), COLOR_YELLOW, 2)
                # Draw feature point
                frame = cv2.circle(frame, (int(a), int(b)), 3, COLOR_BLUE, -1)

            # Optional Bounding Box based on extreme flow points
            if len(good_new) > 4:
                x, y, w, h = cv2.boundingRect(np.array(good_new))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
        
        # Color conditional on form
        form_color = COLOR_GREEN if self.current_form.lower() == "good" else COLOR_RED
        if not self.has_result:
            form_color = COLOR_WHITE

        # Overlay text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Rendering texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        cv2.putText(frame, f"Exercise: {self.current_exercise} ({self.exercise_conf:.2f})", (5, 15), font, font_scale, COLOR_WHITE, thickness)
        cv2.putText(frame, f"Form: {self.current_form} ({self.form_conf:.2f})", (5, 30), font, font_scale, form_color, thickness)
        cv2.putText(frame, f"Speed: {self.exec_speed}", (5, 45), font, font_scale, COLOR_WHITE, thickness)
        cv2.putText(frame, f"Tracked Pts: {len(good_new) if good_new is not None else 0}", (5, 60), font, font_scale, COLOR_WHITE, thickness)
        cv2.putText(frame, f"FPS: {fps:.1f}", (5, 75), font, font_scale, COLOR_WHITE, thickness)

        # Status line: shows whether we're resting or mid-rep, instead of
        # the old fixed-buffer progress bar -- results only update once a
        # rep actually finishes, so this tells you what's happening in
        # between updates.
        if self.rep_state == "in_rep" and self.rep_start_time is not None:
            elapsed = time.perf_counter() - self.rep_start_time
            status_text = f"Rep in progress... ({elapsed:.1f}s)"
            status_color = COLOR_YELLOW
        else:
            status_text = "Resting / waiting for next rep"
            status_color = COLOR_WHITE
        cv2.putText(frame, status_text, (5, 90), font, font_scale, status_color, thickness)

        return frame

    def run(self, video_path=None):
        """Main execution loop. Pass video_path to test against a recorded clip
        instead of a live camera -- same pipeline end to end, no hardware needed."""
        self._load_assets()

        if video_path is not None:
            print(f"Starting video stream from file: {video_path}")
        else:
            print("Starting camera stream...")
        try:
            self.camera = CameraStream(video_path=video_path)
        except RuntimeError as e:
            print(e)
            sys.exit(1)

        cv2.namedWindow("Realtime Exercise Form Monitor", cv2.WINDOW_AUTOSIZE)
        
        prev_time = time.perf_counter()
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    if getattr(self.camera, "is_video_file", False):
                        print("End of video file reached.")
                        break
                    print("Camera disconnected or frame dropped. Retrying...")
                    time.sleep(0.5)
                    continue

                curr_time = time.perf_counter()
                fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time

                # Throttle video-file playback to the source's own frame rate --
                # otherwise a short clip finishes in a fraction of a second and
                # the window never shows anything readable.
                if getattr(self.camera, "is_video_file", False):
                    target_dt = 1.0 / getattr(self.camera, "video_fps", 30.0)
                    elapsed = time.perf_counter() - curr_time
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 1. Feature Extraction
                features, good_new, good_old = self.extractor.process_frame(frame_gray)

                # Update speed classification based on current magnitude
                self.exec_speed = self._classify_speed(features[2])

                # 2. Rep segmentation -- accumulate frames while a rep is in
                # progress, and only get a result back once motion has
                # started AND then stopped (a full rep), instead of
                # re-predicting continuously on a rolling window.
                finished_rep = self._update_rep_state(features, curr_time)

                # 3. Model Inference (only once a full rep has been captured)
                if finished_rep is not None:
                    sequence = pad_or_trim(finished_rep, SEQ_LEN)
                    scaled_buffer = self.scaler.transform(sequence).astype(np.float32)
                    model_input = scaled_buffer.reshape(1, SEQ_LEN, FEATURE_DIM)

                    # Predict (Using __call__ for performance on single samples)
                    preds = self.model(model_input, training=False)
                    exercise_probs, form_probs = preds[0].numpy(), preds[1].numpy()
                    
                    # Extract predictions
                    ex_idx = np.argmax(exercise_probs, axis=1)[0]
                    form_idx = np.argmax(form_probs, axis=1)[0]
                    
                    self.current_exercise = self.exercise_encoder.inverse_transform([ex_idx])[0]
                    self.current_form = self.form_encoder.inverse_transform([form_idx])[0]
                    
                    self.exercise_conf = exercise_probs[0][ex_idx]
                    self.form_conf = form_probs[0][form_idx]
                    self.has_result = True

                    # Always log completed-rep predictions to console too --
                    # useful as a permanent record even if the window is
                    # hard to read in the moment.
                    print(
                        f"[rep complete, {len(finished_rep)} frames] "
                        f"Exercise: {self.current_exercise} ({self.exercise_conf:.2f})  "
                        f"Form: {self.current_form} ({self.form_conf:.2f})"
                    )

                # 4. Display
                display_frame = self._draw_overlay(frame, good_new, good_old, fps)
                cv2.imshow("Realtime Exercise Form Monitor", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Shutdown requested by user.")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down...")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Back exercise form detector.")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file to test the pipeline against, instead of a live camera. "
             "No Pi/camera hardware needed -- reads frames from this file instead.",
    )
    args = parser.parse_args()

    # Setup TensorFlow optimizations for RPi
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    
    BASE_DIR = Path(__file__).resolve().parent.parent / "back_exercise_form_detector"
    MODEL_DIR = BASE_DIR / "models"

    detector = RealtimeDetector(model_dir=MODEL_DIR)
    detector.run(video_path=args.video)

if __name__ == "__main__":
    main()
