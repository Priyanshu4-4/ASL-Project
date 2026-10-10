"""
Real-time ASL fingerspelling recognition from webcam.
Loads model.joblib (trained by train.py) and overlays the predicted
letter live on the video feed.

Requires the model file — run download_model.py first.
"""

import time

import cv2
import joblib
import numpy as np

from hand_tracking import create_landmarker, draw_landmarks, frame_to_mp_image

MODEL_PATH = "model.joblib"


def landmarks_to_features(hand_landmarks):
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features).reshape(1, -1)


def main():
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"No model found at {MODEL_PATH}. Run train.py first.")
        return

    landmarker = create_landmarker()
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        mp_image = frame_to_mp_image(frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        prediction_text = "No hand detected"
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            features = landmarks_to_features(hand_landmarks)
            pred = clf.predict(features)[0]
            proba = clf.predict_proba(features).max()
            prediction_text = f"{pred.upper()} ({proba:.0%})"

        cv2.putText(
            frame, prediction_text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
        )
        cv2.imshow("ASL Fingerspelling Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
