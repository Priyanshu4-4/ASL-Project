"""
Collect labeled ASL handshape samples from your webcam.

Controls:
  - Press a letter key (a-y, skipping j and z — those need motion, not covered here)
    to set the current label.
  - Press SPACE to capture the current hand landmarks under that label.
  - Press ESC to quit.

Each captured sample is one row in data/landmarks.csv:
  label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
(21 hand landmarks x 3 coordinates = 63 feature columns)

Requires the model file — run download_model.py first.
"""

import csv
import os
import string
import time

import cv2

from hand_tracking import create_landmarker, draw_landmarks, frame_to_mp_image

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
    ensure_csv()
    landmarker = create_landmarker(num_hands=2)

    cap = cv2.VideoCapture(0)
    current_label = None
    counts = {label: 0 for label in VALID_LABELS}

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["label"]
            if lbl in counts:
                counts[lbl] += 1

    print("Press a letter key (a-y, no j/z) to select a label, SPACE to capture, ESC to quit.")

    start_time = time.time()
    latest_landmarks = None

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        mp_image = frame_to_mp_image(frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        latest_landmarks = None
        if result.hand_landmarks:
            latest_landmarks = result.hand_landmarks
            for hand in latest_landmarks:
                draw_landmarks(frame, hand)

        label_text = f"Label: {current_label or '-'}"
        count_text = f"Count: {counts.get(current_label, 0) if current_label else 0}"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, count_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collect ASL Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in [ord(c) for c in VALID_LABELS]:
            current_label = chr(key)
        elif key == ord(" ") and current_label and latest_landmarks:
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                for hand in latest_landmarks:
                    writer.writerow(landmarks_to_row(hand, current_label))
                    counts[current_label] += 1
            print(f"Captured '{current_label}' (total: {counts[current_label]})")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
