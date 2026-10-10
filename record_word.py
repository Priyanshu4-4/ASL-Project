"""
Record a short hand-landmark sequence for a whole-word sign, so
text_to_gesture.py can play it back instead of spelling the word out
letter by letter.

Controls:
  - Type a word and press Enter to select it.
  - Press SPACE to start a ~1.5 second recording of your hand performing
    the sign. Landmarks are captured every frame during that window.
  - The clip is saved to data/words/<word>.npy (shape: [frames, 21, 3]),
    overwriting any previous recording for that word.
  - Type 'list' to see already-recorded words, or 'quit' to exit.

Requires the model file — run download_model.py first.
"""

import os
import time

import cv2
import numpy as np

from hand_tracking import create_landmarker, draw_landmarks, frame_to_mp_image

WORDS_DIR = os.path.join("data", "words")
RECORD_SECONDS = 1.5


def list_recorded_words():
    if not os.path.isdir(WORDS_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(WORDS_DIR) if f.endswith(".npy"))


def record_clip(landmarker, cap):
    """Show a live preview; on SPACE, record landmarks for RECORD_SECONDS."""
    start_time = time.time()
    recording = False
    record_start = None
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)
        mp_image = frame_to_mp_image(frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        if landmarks:
            draw_landmarks(frame, landmarks)

        if recording:
            elapsed = time.time() - record_start
            if landmarks:
                frames.append([[lm.x, lm.y, lm.z] for lm in landmarks])
            cv2.putText(frame, f"REC {elapsed:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if elapsed >= RECORD_SECONDS:
                cv2.imshow("Record Word Sign", frame)
                cv2.waitKey(1)
                return frames
        else:
            cv2.putText(frame, "Press SPACE to record", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Record Word Sign", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC cancels
            return None
        if key == ord(" ") and not recording:
            recording = True
            record_start = time.time()
            frames = []


def main():
    os.makedirs(WORDS_DIR, exist_ok=True)
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(0)

    print("Already recorded:", list_recorded_words() or "(none yet)")

    while True:
        word = input("\nWord to record (or 'list' / 'quit'): ").strip().lower()
        if word == "quit":
            break
        if word == "list":
            print("Recorded words:", list_recorded_words() or "(none yet)")
            continue
        if not word.isalpha():
            print("Please enter a single alphabetic word.")
            continue

        print(f"Get ready to sign '{word}'. Press SPACE in the window to start recording.")
        frames = record_clip(landmarker, cap)

        if not frames:
            print("No frames captured (no hand detected, or cancelled). Try again.")
            continue

        arr = np.array(frames)  # (T, 21, 3)
        path = os.path.join(WORDS_DIR, f"{word}.npy")
        np.save(path, arr)
        print(f"Saved {arr.shape[0]} frames to {path}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
