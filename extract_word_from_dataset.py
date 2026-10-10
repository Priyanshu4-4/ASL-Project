"""
Extract whole-word hand-landmark sequences from the WLASL dataset and save
them as data/words/<word>.npy — the same format record_word.py produces —
so text_to_gesture.py can use them automatically.

Get the dataset first (requires a free Kaggle account):
  pip install kaggle
  # set up your API token first: https://www.kaggle.com/docs/api
  kaggle datasets download -d risangbaskoro/wlasl-processed
  unzip wlasl-processed.zip -d wlasl-processed

You should end up with:
  wlasl-processed/WLASL_v0.3.json
  wlasl-processed/videos/<video_id>.mp4

IMPORTANT — LICENSE: WLASL is released under the Computational Use of Data
Agreement (C-UDA), which is research/non-commercial use only. Fine for
prototyping and learning; if this project becomes a commercial product,
this data (or a model trained on it) isn't cleared for that without
separate permission from the dataset authors.

Usage:
  python extract_word_from_dataset.py --data_dir wlasl-processed
  python extract_word_from_dataset.py --data_dir wlasl-processed --words hello,thankyou,please
  python extract_word_from_dataset.py --data_dir wlasl-processed --all   # all ~2000 words (slow!)

NOTE: not every video referenced in WLASL_v0.3.json is actually present in
the Kaggle mirror's videos/ folder — several are missing. The script tries
up to --instances_per_word candidates per word (default 8) before giving up,
and reports how many were missing vs. had no hand detected, so you can see
which is the actual bottleneck for words that fail.
"""

import argparse
import json
import os

import cv2
import numpy as np

from hand_tracking import create_landmarker, frame_to_mp_image
from mediapipe.tasks.python import vision

WORDS_DIR = os.path.join("data", "words")

# A reasonably useful starting vocabulary if you don't pass --words or --all.
DEFAULT_WORDS = [
    "hello", "thankyou", "please", "sorry", "yes", "no",
    "love", "friend", "family", "help", "name", "what",
    "where", "who", "eat", "drink", "water", "food",
    "work", "school", "good", "bad", "happy", "sad",
]


def load_gloss_map(data_dir):
    json_path = os.path.join(data_dir, "WLASL_v0.3.json")
    with open(json_path) as f:
        entries = json.load(f)
    return {e["gloss"].lower(): e["instances"] for e in entries}


def extract_landmarks_from_video(landmarker, video_path, frame_start, frame_end):
    """Uses IMAGE mode (detect() per frame) rather than VIDEO mode, since
    VIDEO mode requires timestamps to strictly increase across every call on
    the same landmarker instance — which breaks when the same landmarker is
    reused across multiple, separately-timestamped video files."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx < frame_start:
            continue
        if frame_end != -1 and idx > frame_end:
            break

        mp_image = frame_to_mp_image(frame)
        result = landmarker.detect(mp_image)
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            frames.append([[p.x, p.y, p.z] for p in lm])

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="Folder containing WLASL_v0.3.json and videos/")
    parser.add_argument("--words", default=None,
                         help="Comma-separated list of glosses to extract (default: a small curated list)")
    parser.add_argument("--all", action="store_true",
                         help="Extract every gloss in the dataset (slow — ~2000 words)")
    parser.add_argument("--instances_per_word", type=int, default=8,
                         help="How many video instances to try per word before giving up (default 8 — "
                              "many entries in the JSON reference videos missing from the Kaggle mirror, "
                              "so trying just 1 fails often)")
    args = parser.parse_args()

    os.makedirs(WORDS_DIR, exist_ok=True)
    gloss_map = load_gloss_map(args.data_dir)

    if args.all:
        target_words = list(gloss_map.keys())
    elif args.words:
        target_words = [w.strip().lower() for w in args.words.split(",")]
    else:
        target_words = DEFAULT_WORDS

    landmarker = create_landmarker(running_mode=vision.RunningMode.IMAGE)
    videos_dir = os.path.join(args.data_dir, "videos")

    done, skipped = 0, 0
    for word in target_words:
        if word not in gloss_map:
            print(f"'{word}': not found in dataset gloss list — skipping")
            skipped += 1
            continue

        instances = gloss_map[word][: args.instances_per_word]
        saved = False
        missing_count = 0
        no_hand_count = 0

        for inst in instances:
            video_path = os.path.join(videos_dir, f"{inst['video_id']}.mp4")
            if not os.path.exists(video_path):
                missing_count += 1
                continue

            frames = extract_landmarks_from_video(
                landmarker, video_path,
                inst.get("frame_start", 1), inst.get("frame_end", -1),
            )
            if len(frames) >= 5:  # need a minimally usable sequence
                arr = np.array(frames)
                np.save(os.path.join(WORDS_DIR, f"{word}.npy"), arr)
                print(f"'{word}': saved {arr.shape[0]} frames from {inst['video_id']}.mp4 "
                      f"(after {missing_count} missing file(s))")
                saved = True
                done += 1
                break
            else:
                no_hand_count += 1

        if not saved:
            print(f"'{word}': tried {len(instances)} instance(s) — "
                  f"{missing_count} file(s) missing from videos/, "
                  f"{no_hand_count} had no hand reliably detected — skipping")
            skipped += 1

    landmarker.close()
    print(f"\nDone. Saved {done} words, skipped {skipped}.")


if __name__ == "__main__":
    main()
