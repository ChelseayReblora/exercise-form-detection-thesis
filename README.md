# backex — Back Exercise Form Detector

Detects proper vs. improper form on three back exercises (lat pulldown, seated
cable row, cable pullover) from live video, using Lucas-Kanade optical flow
for motion tracking and a GRU network for classification. Built for
deployment on a Raspberry Pi with a Picamera2 camera and a touchscreen
display, based on the accompanying thesis (Macaro & Reblora, Mapúa
University, 2024).

## Repo layout

```
src/
  extract_features.py    # offline: video -> optical flow features (training data prep)
  train_model.py          # offline: trains the two-headed GRU (exercise + form)
  convert_to_tflite.py    # offline: exports gru_model.keras to .tflite / quantized .tflite
  realtime_detector.py    # the live app: camera -> optical flow -> GRU -> on-screen overlay
  optical_flow_core.py    # shared Shi-Tomasi + Lucas-Kanade tracker, used by both
                           # extract_features.py and realtime_detector.py so the training
                           # and live pipelines cannot silently diverge
  roi.py                   # Region-of-Interest masks that constrain keypoint search to
                           # where the exerciser is expected to be, filtering out stray
                           # points from background gym equipment / clutter
  metrics.py               # Percentage Error metric (thesis Eq. 3.1) as a reusable function
back_exercise_form_detector/
  features/                # saved X.npy / label arrays from extract_features.py
  models/                  # trained model + scaler + encoders (already included)
scripts/
  setup_pi.sh              # one-shot environment setup for Raspberry Pi 5 / Raspberry Pi OS
tests/                      # pytest suite (see "Testing" below)
```

## Running on a Raspberry Pi 5 (target hardware)

Tested against the latest Raspberry Pi OS (Bookworm, 64-bit).

```bash
git clone https://github.com/engi-neil/backex.git
cd backex
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
source venv/bin/activate
python3 src/realtime_detector.py
```

`setup_pi.sh` installs the system camera bindings via `apt` (`picamera2`
cannot be installed correctly from an isolated pip venv), creates a venv that
can still see them, and installs the remaining Python dependencies,
preferring the lightweight `tflite-runtime` over full `tensorflow` where a
wheel is available. See the script itself for the exact steps.

**Before running:** make sure a camera is actually connected and detected:
```bash
rpicam-hello --list-cameras
```
If this reports "No cameras available", fix that before touching the app —
it's a hardware/ribbon-cable issue, not a code issue.

## Running on a Mac or Linux dev machine (no Pi hardware)

`realtime_detector.py` automatically falls back to your machine's built-in
webcam via `cv2.VideoCapture` if `picamera2` isn't importable — useful for
developing/testing the optical-flow and GRU pipeline without a Pi.

TensorFlow does **not** yet support the newest Python releases (e.g. 3.13+),
so use Python 3.11:

```bash
brew install python@3.11        # macOS
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install opencv-python numpy scikit-learn joblib pytest tensorflow
python3 src/realtime_detector.py
```

Do **not** install `tensorflow-macos` / `tensorflow-metal` on recent
TensorFlow versions — Apple folded macOS support into the regular
`tensorflow` package, and the old `-macos` package name is now a stale stub
that will fail to import (`dlopen` / `_pywrap_tensorflow_internal.so` error).

On first run, macOS will prompt for camera permission — grant it to your
terminal app, or `cv2.VideoCapture(0)` will silently fail to produce frames.

## What you'll see

A window titled **"Realtime Exercise Form Monitor"** opens, showing the live
feed with:
- yellow motion vectors + blue dots on tracked optical-flow keypoints
- an orange bounding box around the tracked point cluster
- overlay text: exercise name, form (good/bad), execution speed, tracked
  point count, FPS
- a "Buffering..." progress bar for the first 60 frames — no
  classification appears until that sequence window fills, then it
  re-predicts every 3rd frame

Press `q` in the window (or Ctrl+C in the terminal) to quit.

This needs a real display session (the Pi's touchscreen over HDMI, or your
Mac's desktop) — it will not work over a headless SSH session.

## Testing

```bash
source venv/bin/activate
pip install -r requirements-test.txt   # adds pytest if not already present
pytest -v
```

The suite (`tests/`) validates each stage of the pipeline independently
using synthetic frames, so it doesn't require a camera or GPU:
- camera color handling + `cv2.VideoCapture` fallback
- Lucas-Kanade tracking correctness on known synthetic motion
- ROI isolation (`tests/test_roi.py`) and stray-point exclusion end-to-end
- training/inference tracker equivalence (`tests/test_train_inference_consistency.py`)
- the SEQ_LEN=60 rolling feature buffer
- the Percentage Error formula from the thesis (Chapter 3, Eq. 3.1)
- end-to-end inference against the actual trained `gru_model.keras`

Model-inference tests are skipped automatically if TensorFlow isn't
installed in your environment.

## Troubleshooting

### "hindi normal ang display" (camera colors look wrong / tinted)
Fixed as of this branch. Root cause: Picamera2's `"XRGB8888"` format returns
frames already ordered `(B, G, R, X)` in memory (not `X, R, G, B` as the name
suggests), and the old code re-applied `cv2.COLOR_RGB2BGR` on top of that,
swapping red and blue a second time. If you see this again, check
`CameraStream.read()` in `realtime_detector.py` hasn't regressed back to
`cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)`.

### "walang nattrack na points" (Tracked Pts stays at 0)
Fixed as of this branch. Two compounding bugs caused this:
1. `_compute_motion_features` indexed OpenCV's `(N,1,2)`-shaped point arrays
   incorrectly after boolean-filtering by `status`, throwing a silent
   `IndexError` on almost every frame with any tracked point (caught and
   printed as `"Optical Flow Error: ..."`, then the tracker reset itself).
2. When keypoints were refreshed via `goodFeaturesToTrack`, they were
   detected on the *current* frame but immediately compared against the
   *previous* frame's `old_gray` in the next Lucas-Kanade call — a
   one-frame mismatch that made freshly-detected points look lost
   immediately.

If tracking breaks again, check the terminal for `"Optical Flow Error"`
being printed repeatedly — that's the symptom of bug #1 recurring.

### Tracked Pts stays low even with the fix
`goodFeaturesToTrack`'s `qualityLevel=0.3` (in `FEATURE_PARAMS`) is fairly
strict — it needs a reasonably textured, well-lit scene. A blank wall or flat
lighting will legitimately produce few or no corners. This isn't a bug; test
against a textured background and make sure the exerciser stands out from it
(per the thesis's own data-gathering guidance: "ensure good lighting, and a
clean and uncluttered background").

### Stray points tracked on background equipment / other people
Tracking is now constrained by an ROI (`src/roi.py`), layering:
1. a static ellipse covering the region the exerciser is expected to occupy,
   given the thesis's fixed camera placement (200cm distance, 100cm height),
   used as a fallback and outer bound;
2. a bounding box around whatever is currently being tracked, tightening the
   search area once tracking is established.

Both `extract_features.py` and `realtime_detector.py` get this for free by
building on the shared `OpticalFlowTracker` in `src/optical_flow_core.py` --
neither file calls `cv2.goodFeaturesToTrack` or `cv2.calcOpticalFlowPyrLK`
directly anymore. If stray points reappear, check whether `STATIC_ROI_X_FRAC`
/ `STATIC_ROI_Y_FRAC` in `roi.py` still match the actual camera framing (a
repositioned tripod or different room can invalidate the assumption). Note
this is an isolation layer on top of optical flow, not a replacement for
it -- Shi-Tomasi corner detection and Lucas-Kanade tracking are unchanged.

### `ModuleNotFoundError: No module named 'picamera2'` on the Pi
`picamera2` must come from `apt` (`sudo apt install python3-picamera2
python3-libcamera`), not `pip`, and your venv must be created with
`--system-site-packages` to see it. `scripts/setup_pi.sh` handles both.

### `tensorflow` fails to import with a `dlopen` / `.dylib` / `.so` error
Usually a mismatched `tensorflow-metal` version against your `tensorflow`
version (macOS), or a `tflite-runtime` wheel built for a different Python
minor version (Pi). Uninstall and reinstall the matching pair, or drop the
GPU/accelerator plugin entirely — it's optional for running or testing this
app.

### `cv2.imshow` errors about display / Qt / Cocoa backend
You're running headless (e.g. over plain SSH without X forwarding). This app
needs a real graphical session — run it directly on the Pi's desktop (or
its attached touchscreen), or locally on your Mac, not through a headless
shell.

### Camera opens but frame is `None` / capture fails intermittently
Check `rpicam-hello --list-cameras` (Pi) or macOS camera permissions
(System Settings → Privacy & Security → Camera) first — this is almost
always a hardware/permission issue rather than an app bug.

## Known gaps / follow-ups

- `realtime_detector.py` still loads `gru_model.keras` via full
  `tf.keras.models.load_model`, even though `convert_to_tflite.py` already
  produces `gru_model_quantized.tflite`. Switching inference to
  `tflite_runtime.interpreter.Interpreter` would make the Pi deployment
  meaningfully lighter and faster, and `setup_pi.sh` already installs
  `tflite-runtime` in anticipation of this.
- The thesis's stated Percentage Error formula (Eq. 3.1: `(VA - VE) / VE *
  100`, signed negative) and `train_model.py`'s actual computation
  (`incorrect / total * 100`, unsigned) report the same magnitude with
  opposite signs — worth reconciling before finalizing Table 3.2 in the
  written thesis. See `tests/test_metrics.py::TestFormulaSignConventionDiscrepancy`.
