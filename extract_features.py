import os
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optical_flow_core import OpticalFlowTracker, pad_or_trim

# PATH CONFIGURATION

DATASET_DIR = Path(r"C:\Users\Neil\Downloads\backex-feature-roi-point-isolation\backex-feature-roi-point-isolation - Copy\CUT VIDS")

OUTPUT_DIR = Path(r"C:\Users\Neil\Downloads\backex-feature-roi-point-isolation\backex-feature-roi-point-isolation - Copy\back_exercise_form_detector")
FEATURE_DIR = OUTPUT_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# Folder name mapping
exercise_name_map = {
    "ex1": "lat_pulldown",
    "ex2": "seated_cable_row",
    "ex3": "cable_pullover"
}

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# Sequence length
SEQ_LEN = 60

# Frame size for faster processing
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Shi-Tomasi keypoint detection parameters
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

def extract_features_from_video(video_path):
    """
    Extracts Lucas-Kanade optical flow features from one video, using the
    same OpticalFlowTracker (ROI-constrained Shi-Tomasi + LK) AND the same
    pad_or_trim (last-frame-repeat padding, not zero-padding) as the live
    detector, so training data and live inference stay in lockstep on both
    tracking behavior and sequence-length handling.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    tracker = OpticalFlowTracker(feature_params=feature_params, lk_params=lk_params)
    sequence_features = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        features, _, _ = tracker.process_frame(frame_gray)
        sequence_features.append(features)

    cap.release()

    return pad_or_trim(sequence_features, SEQ_LEN)


# PROCESS DATASET

def process_dataset():
    X = []
    exercise_labels = []
    form_labels = []

    for exercise_folder in DATASET_DIR.iterdir():
        if not exercise_folder.is_dir():
            continue

        folder_name = exercise_folder.name

        if folder_name not in exercise_name_map:
            print(f"Skipping unknown folder: {folder_name}")
            continue

        exercise_label = exercise_name_map[folder_name]

        for form_folder in exercise_folder.iterdir():
            if not form_folder.is_dir():
                continue

            form_label = form_folder.name.lower()

            if form_label not in ["good", "bad"]:
                print(f"Skipping unknown form folder: {form_folder}")
                continue

            for video_file in form_folder.iterdir():
                if video_file.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                print(f"Processing: {video_file}")
                print(f"Exercise: {exercise_label}")
                print(f"Form: {form_label}")

                features = extract_features_from_video(video_file)

                if features is not None:
                    X.append(features)
                    exercise_labels.append(exercise_label)
                    form_labels.append(form_label)

    X = np.array(X, dtype=np.float32)
    exercise_labels = np.array(exercise_labels)
    form_labels = np.array(form_labels)

    np.save(FEATURE_DIR / "X.npy", X)
    np.save(FEATURE_DIR / "exercise_labels.npy", exercise_labels)
    np.save(FEATURE_DIR / "form_labels.npy", form_labels)

    print("\nFeature extraction finished.")
    print("Saved to:", FEATURE_DIR)
    print("X shape:", X.shape)
    print("Exercise labels:", exercise_labels.shape)
    print("Form labels:", form_labels.shape)


if __name__ == "__main__":
    process_dataset()
