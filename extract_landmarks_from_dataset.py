"""
Extract hand landmarks from the Kaggle "ASL Alphabet" dataset and append
them to data/landmarks.csv in the same format collect_data.py produces —
so train.py works unchanged on top of either source (or both combined).

Get the dataset first (requires a free Kaggle account):
  1. Via the Kaggle website: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
     Download and unzip it.
  2. Or via the Kaggle CLI (pip install kaggle, then set up your API token
     as per https://www.kaggle.com/docs/api):
       kaggle datasets download -d grassknoted/asl-alphabet
       unzip asl-alphabet.zip

Either way you should end up with a folder structure like:
  asl_alphabet_train/asl_alphabet_train/A/*.jpg
  asl_alphabet_train/asl_alphabet_train/B/*.jpg
  ...

Usage:
  python extract_landmarks_from_dataset.py --data_dir path/to/asl_alphabet_train/asl_alphabet_train

By default this samples up to 300 images per letter (enough for a solid
model without making processing take forever on CPU — increase with
--max_per_class if you want more).
"""

import argparse
import csv
import os
import string

import cv2

from hand_tracking import create_landmarker
from mediapipe.tasks.python import vision

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
VALID_LABELS = [c for c in string.ascii_lowercase if c not in ("j", "z")]


def ensure_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
            writer.writerow(header)


def landmarks_to_row(hand_landmarks, label):
    row = [label]
    for lm in hand_landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the dataset folder containing one subfolder per letter (A, B, C, ...)",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=300,
        help="Max images to process per letter (default 300)",
    )
    args = parser.parse_args()

    ensure_csv()
    landmarker = create_landmarker(running_mode=vision.RunningMode.IMAGE)

    total_written = 0
    for label in VALID_LABELS:
        folder = os.path.join(args.data_dir, label.upper())
        if not os.path.isdir(folder):
            print(f"Skipping '{label}' — folder not found: {folder}")
            continue

        files = sorted(os.listdir(folder))[: args.max_per_class]
        written = 0
        rows = []

        for fname in files:
            path = os.path.join(folder, fname)
            image = cv2.imread(path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)
            if result.hand_landmarks:
                rows.append(landmarks_to_row(result.hand_landmarks[0], label))
                written += 1

        if rows:
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerows(rows)

        print(f"'{label}': {written}/{len(files)} images had a detected hand")
        total_written += written

    landmarker.close()
    print(f"\nDone. Appended {total_written} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
