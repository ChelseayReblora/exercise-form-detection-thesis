"""
Scans the Cut Vids Final dataset for duplicate video files (byte-identical
content, regardless of filename) using SHA-256 hashing. Also flags videos
under ~2 seconds (60 frames @ 30fps) as candidates for the zero-padding
issue discussed separately, and reports the good/bad split per exercise.

Usage:
    python3 find_duplicate_videos.py "C:\\Users\\Neil\\Downloads\\backex\\Cut Vids Final"
"""
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    print("Note: opencv-python not installed, skipping frame-count check "
          "(duplicate detection still works). pip install opencv-python to enable it.\n")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SEQ_LEN = 60  # matches extract_features.py


def hash_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def get_frame_count(path):
    if not HAVE_CV2:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 find_duplicate_videos.py <path to Cut Vids Final>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"Not a directory: {root}")
        sys.exit(1)

    hash_to_paths = defaultdict(list)
    class_counts = defaultdict(int)
    short_clips = []

    all_videos = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(f"Scanning {len(all_videos)} video files under {root} ...\n")

    for path in all_videos:
        # Expect structure: .../exN/good|bad/filename
        try:
            form_label = path.parent.name.lower()
            exercise_label = path.parent.parent.name.lower()
        except IndexError:
            form_label, exercise_label = "unknown", "unknown"

        class_counts[(exercise_label, form_label)] += 1

        file_hash = hash_file(path)
        hash_to_paths[file_hash].append(path)

        frame_count = get_frame_count(path)
        if frame_count is not None and frame_count < SEQ_LEN:
            short_clips.append((path, frame_count))

    # --- Report duplicates ---
    duplicate_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    total_duplicate_files = sum(len(paths) - 1 for paths in duplicate_groups.values())

    print("=" * 70)
    print(f"DUPLICATE CHECK: {len(duplicate_groups)} distinct clips have duplicates")
    print(f"  -> {total_duplicate_files} redundant files total (same content, different name)")
    print("=" * 70)
    for h, paths in sorted(duplicate_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  Identical content ({len(paths)} copies):")
        for p in paths:
            print(f"    {p.relative_to(root)}")

    # --- Report per-class counts ---
    print("\n" + "=" * 70)
    print("CLASS COUNTS (as found on disk, including duplicates)")
    print("=" * 70)
    for (exercise, form), count in sorted(class_counts.items()):
        print(f"  {exercise:20s} {form:6s} : {count}")

    # --- Report short clips (padding risk) ---
    if HAVE_CV2:
        print("\n" + "=" * 70)
        print(f"SHORT CLIPS (< {SEQ_LEN} frames, will be zero-padded by pad_or_trim)")
        print("=" * 70)
        if short_clips:
            for p, fc in sorted(short_clips, key=lambda x: x[1]):
                print(f"  {fc:3d} frames  {p.relative_to(root)}")
        else:
            print("  None found -- all clips are >= 60 frames.")

    print("\nDone.")


if __name__ == "__main__":
    main()
