"""
Finds byte-identical duplicate videos in the Cut Vids Final dataset (same
hashing logic as find_duplicate_videos.py) and removes the redundant copies,
keeping exactly one file per group of identical content.

SAFE BY DEFAULT:
  - Dry run unless you pass --execute. Always look at the printed plan first.
  - Redundant copies are MOVED to a quarantine folder, not permanently
    deleted, unless you explicitly pass --permanent-delete. If a "kept" file
    turns out to be the wrong pick, you can move things back by hand.
  - A log file (dedup_log.csv) is written next to the quarantine folder
    recording exactly what happened to each file, so the run is auditable
    and reversible.

Which copy is kept per duplicate group:
  1. Prefer a filename that does NOT contain "copy" (case-insensitive) and
     does NOT end in a "(N)" suffix, e.g. prefer "r3.mp4" over "Copy of
     r3.mp4" or "r3(1).mp4".
  2. Among remaining candidates, prefer the shortest filename.
  3. Tie-break alphabetically, for a fully deterministic choice.

Usage:
    # 1. Dry run first -- always do this and read the output.
    python3 remove_duplicate_videos.py "C:\\Users\\Neil\\Downloads\\backex\\Cut Vids Final"

    # 2. Once you're happy with the plan, actually move the duplicates:
    python3 remove_duplicate_videos.py "C:\\Users\\Neil\\Downloads\\backex\\Cut Vids Final" --execute

    # 3. (Optional, only after checking the quarantine folder yourself)
    #    Permanently delete instead of quarantining:
    python3 remove_duplicate_videos.py "C:\\Users\\Neil\\Downloads\\backex\\Cut Vids Final" --execute --permanent-delete
"""
import sys
import csv
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def hash_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_copy_like(name: str) -> bool:
    """True if the filename looks like a duplicate export/copy, e.g.
    'Copy of r3.mp4' or 'r3(1).mp4'."""
    stem = Path(name).stem.lower()
    if "copy" in stem:
        return True
    if stem.endswith(")") and "(" in stem:
        inside = stem[stem.rfind("(") + 1: -1]
        if inside.isdigit():
            return True
    return False


def choose_keeper(paths):
    """Pick which file in a duplicate group to keep, per the rules in the
    module docstring. Returns (keeper, [the rest])."""
    def sort_key(p):
        return (is_copy_like(p.name), len(p.name), p.name.lower())

    ordered = sorted(paths, key=sort_key)
    keeper = ordered[0]
    redundant = ordered[1:]
    return keeper, redundant


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="Path to the Cut Vids Final folder")
    parser.add_argument("--execute", action="store_true",
                         help="Actually move/delete files. Without this, only prints the plan.")
    parser.add_argument("--permanent-delete", action="store_true",
                         help="Permanently delete redundant copies instead of quarantining them. "
                              "Only takes effect together with --execute.")
    parser.add_argument("--quarantine-dir", default=None,
                         help="Where to move redundant copies (default: <root>/_duplicates_removed)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        sys.exit(1)

    quarantine_dir = Path(args.quarantine_dir) if args.quarantine_dir else root / "_duplicates_removed"

    all_videos = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        and quarantine_dir not in p.parents
    ]

    print(f"Scanning {len(all_videos)} video files under {root} ...\n")

    hash_to_paths = defaultdict(list)
    for path in all_videos:
        hash_to_paths[hash_file(path)].append(path)

    duplicate_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    if not duplicate_groups:
        print("No duplicate files found. Nothing to do.")
        return

    plan = []  # (keeper, redundant_path)
    for h, paths in duplicate_groups.items():
        keeper, redundant = choose_keeper(paths)
        for r in redundant:
            plan.append((keeper, r))

    print("=" * 70)
    print(f"PLAN: {len(duplicate_groups)} duplicate groups, {len(plan)} redundant files")
    action_word = "PERMANENTLY DELETE" if args.permanent_delete else f"MOVE to {quarantine_dir}"
    print(f"Action: {action_word}")
    print("=" * 70)
    for keeper, redundant in plan:
        rel_keep = keeper.relative_to(root)
        rel_red = redundant.relative_to(root)
        print(f"  KEEP:   {rel_keep}")
        print(f"  REMOVE: {rel_red}\n")

    if not args.execute:
        print("Dry run only -- no files were changed. Re-run with --execute to apply this plan.")
        return

    # --- Execute ---
    if not args.permanent_delete:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    log_path = quarantine_dir.parent / "dedup_log.csv" if not args.permanent_delete else root / "dedup_log.csv"
    with open(log_path, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["timestamp", "action", "kept_file", "removed_file", "new_location"])
        timestamp = datetime.now().isoformat(timespec="seconds")

        for keeper, redundant in plan:
            rel_keep = str(keeper.relative_to(root))
            rel_red = str(redundant.relative_to(root))

            if args.permanent_delete:
                redundant.unlink()
                writer.writerow([timestamp, "permanently_deleted", rel_keep, rel_red, ""])
            else:
                # Preserve the original relative subfolder structure inside quarantine
                dest = quarantine_dir / redundant.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(redundant), str(dest))
                writer.writerow([timestamp, "quarantined", rel_keep, rel_red, str(dest)])

    print(f"\nDone. {len(plan)} redundant files processed.")
    print(f"Log written to: {log_path}")
    if not args.permanent_delete:
        print(f"Redundant copies moved to: {quarantine_dir}")
        print("Once you've confirmed everything looks right, you can delete that folder by hand.")


if __name__ == "__main__":
    main()
