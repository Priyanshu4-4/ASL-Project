"""
Text -> ASL gesture animation.

Pipeline:
  1. English sentence -> simplified ASL gloss (gloss.py): drops articles/
     "to be", reorders time words to the front, moves wh-words per ASL
     question structure.
  2. For each gloss token: if a recorded whole-word sign exists
     (data/words/<word>.npy, from record_word.py), play that landmark
     sequence back. Otherwise, fingerspell the word letter by letter using
     the per-letter templates learned in data/landmarks.csv.

Usage:
  python text_to_gesture.py "Did you eat yesterday?"
  python text_to_gesture.py --no-gloss "HELLO"   # skip grammar conversion,
                                                   # spell/sign literally
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from gloss import to_gloss
from hand_tracking import draw_skeleton_xyz

LETTERS_CSV = "data/landmarks.csv"
WORDS_DIR = os.path.join("data", "words")
CANVAS_SIZE = 500
LETTER_HOLD_MS = 700
WORD_FRAME_MS = 60  # ms per frame when playing a recorded word clip


def load_letter_templates():
    if not os.path.exists(LETTERS_CSV):
        return {}
    df = pd.read_csv(LETTERS_CSV)
    templates = {}
    for label, group in df.groupby("label"):
        coords = group.drop(columns=["label"]).mean().values
        templates[label] = coords.reshape(21, 3)
    return templates


def load_word_clip(word):
    path = os.path.join(WORDS_DIR, f"{word.lower()}.npy")
    if os.path.exists(path):
        return np.load(path)  # (T, 21, 3)
    return None


def render_frame(landmarks_xyz, caption):
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 30, dtype=np.uint8)
    draw_skeleton_xyz(canvas, landmarks_xyz)
    cv2.putText(canvas, caption, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (255, 255, 255), 3)
    return canvas


def play_word_clip(clip, word):
    for frame_landmarks in clip:
        img = render_frame(frame_landmarks, word.upper())
        cv2.imshow("Text to ASL", img)
        if (cv2.waitKey(WORD_FRAME_MS) & 0xFF) == 27:
            return False
    return True


def fingerspell_word(word, letter_templates):
    missing = [c for c in word if c not in letter_templates]
    if missing:
        print(f"  (skipping '{word}' — no trained shape for: {sorted(set(missing))})")
        return True
    for c in word:
        img = render_frame(letter_templates[c], c.upper())
        cv2.imshow("Text to ASL", img)
        if (cv2.waitKey(LETTER_HOLD_MS) & 0xFF) == 27:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", help="Text to translate")
    parser.add_argument("--no-gloss", action="store_true",
                         help="Skip grammar conversion; treat each input word literally")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else input("Enter text: ")

    letter_templates = load_letter_templates()

    if args.no_gloss:
        tokens = [w.lower() for w in text.split() if w.isalpha()]
    else:
        tokens = [t.lower() for t in to_gloss(text)]
        print("Gloss:", " ".join(t.upper() for t in tokens))

    if not tokens:
        print("Nothing to show.")
        return

    print("Playing. Press ESC to stop early.")
    for word in tokens:
        clip = load_word_clip(word)
        if clip is not None:
            print(f"  '{word}': using recorded whole-word sign")
            if not play_word_clip(clip, word):
                break
        else:
            print(f"  '{word}': no recorded sign — fingerspelling")
            if not fingerspell_word(word, letter_templates):
                break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
