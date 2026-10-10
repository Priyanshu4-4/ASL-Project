# ASL Fingerspelling Recognizer (Phase 1 MVP)

A beginner-friendly starter project: recognize static ASL alphabet handshapes
(A-Y, excluding J and Z which require motion) in real time from a webcam,
using MediaPipe hand landmarks + a simple classifier.

## Why this approach
Instead of training a CNN on raw images (slow, data-hungry, hard for a
beginner), we use MediaPipe to extract 21 hand landmark points (x, y, z)
per frame. That turns "recognize a handshape" into a simple tabular
classification problem — a Random Forest or small neural net can hit
90%+ accuracy with a modest amount of data.

## Project structure
```
asl-fingerspelling/
├── requirements.txt
├── download_model.py    # one-time download of the MediaPipe hand model
├── hand_tracking.py     # shared helper: hand detection + landmark drawing
├── extract_landmarks_from_dataset.py  # convert a downloaded image dataset into landmarks.csv
├── collect_data.py      # capture your own labeled samples via webcam
├── train.py              # train a classifier on collected landmarks
├── predict.py             # real-time recognition from webcam (gesture -> text)
├── record_word.py         # record whole-word sign clips via webcam
├── extract_word_from_dataset.py  # pull whole-word signs from the WLASL dataset
├── gloss.py                # spaCy-based English -> ASL gloss (grammar) converter
├── test_gloss.py            # regression test sentences for gloss.py — run after changes
├── vocab_coverage.py         # tracks recorded vs. target vocabulary by category
├── text_to_gesture.py     # gloss + word playback + fingerspelling fallback (text -> gesture)
├── models/
│   └── hand_landmarker.task   # downloaded by download_model.py
└── data/
    ├── landmarks.csv     # collected training data (from the dataset script and/or collect_data.py)
    └── words/            # recorded whole-word sign clips (from record_word.py)
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_model.py        # downloads the hand-tracking model (~10MB, one time)
```

**Note on MediaPipe versions:** as of MediaPipe 1.0, Google removed the old
`mp.solutions.hands` API entirely in favor of the newer Tasks API, which is
what these scripts use (`mediapipe.tasks.python.vision.HandLandmarker`). It
needs the separate model file that `download_model.py` fetches — that's why
this step exists.

**gloss.py also needs a spaCy language model**, downloaded separately from
the `pip install`:
```bash
python -m spacy download en_core_web_sm
```

## Workflow

### 1. Get training data — from a public dataset

We use the **Kaggle ASL Alphabet dataset** (~87,000 images, 29 classes).
It requires a free Kaggle account to download.

**Option A — browser:**
Go to https://www.kaggle.com/datasets/grassknoted/asl-alphabet, click Download,
and unzip it into this project folder.

**Option B — Kaggle CLI:**
```bash
pip install kaggle
# set up your API token first: https://www.kaggle.com/docs/api
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip
```

Either way you'll end up with a structure like:
```
asl_alphabet_train/asl_alphabet_train/A/*.jpg
asl_alphabet_train/asl_alphabet_train/B/*.jpg
...
```

Then extract hand landmarks from the images into `data/landmarks.csv`:
```bash
python extract_landmarks_from_dataset.py --data_dir asl_alphabet_train/asl_alphabet_train
```
This samples up to 300 images per letter by default (increase with
`--max_per_class` for a larger, more robust dataset — at the cost of
longer processing time on CPU). It skips J and Z, since those are
motion-based signs this static-frame pipeline doesn't cover yet.

*(Optional, later: you can also run `collect_data.py` to add your own
webcam samples to the same CSV — useful if the model struggles with your
specific hand/lighting.)*

### 1b. Or: collect your own data instead
```bash
python collect_data.py
```
- Press a letter key (a-y, skipping j/z) to set the current label.
- Press SPACE to capture a sample of your hand in that shape.
- Aim for 100-200 samples per letter, varying hand angle/distance slightly.
- Press ESC to quit. Data is appended to `data/landmarks.csv`.

### 2. Train the model
```bash
python train.py
```
- Trains a Random Forest classifier on `data/landmarks.csv`.
- Prints accuracy on a held-out test split.
- Saves the trained model to `model.joblib`.

### 3. Run real-time recognition
```bash
python predict.py
```
- Opens your webcam and overlays the predicted letter live.

### 4. Add whole-word signs
Two ways to get them — use either or both:

**Option A — record your own (fast, small vocabulary):**
```bash
python record_word.py
```
Type a word (e.g. `hello`), press SPACE, perform the sign for ~1.5 seconds.
Saves to `data/words/hello.npy`.

**Option B — pull from the WLASL public dataset (slower setup, bigger vocabulary):**
WLASL has ~21,000 clips across ~2,000 ASL words. Get it (free Kaggle account
required):
```bash
pip install kaggle
# set up your API token first: https://www.kaggle.com/docs/api
kaggle datasets download -d risangbaskoro/wlasl-processed
unzip wlasl-processed.zip -d wlasl-processed
```
Then extract landmark sequences for the words you want:
```bash
python extract_word_from_dataset.py --data_dir wlasl-processed
# or specify your own list:
python extract_word_from_dataset.py --data_dir wlasl-processed --words hello,thankyou,please,love
# or grab everything (slow — ~2000 words):
python extract_word_from_dataset.py --data_dir wlasl-processed --all
```
This saves the same `data/words/<word>.npy` format as `record_word.py`, so
`text_to_gesture.py` uses either source without any extra setup.

**⚠️ License note:** WLASL is released under the Computational Use of Data
Agreement (C-UDA) — **research/non-commercial use only**. Fine for
prototyping and learning; if this project becomes a commercial product,
you'd need separate permission from the WLASL authors (or rely on
self-recorded data instead, which you fully own).

Words with no recording (from either source) automatically fall back to
fingerspelling.

### 5. Run text → gesture (the vice-versa direction)
```bash
python text_to_gesture.py "Did you eat yesterday?"
```
This does two things:
1. **Grammar conversion** (`gloss.py`): uses spaCy's dependency parser to
   analyze each sentence's grammatical structure (part of speech, subject/
   verb/object roles) and rewrite it into simplified ASL gloss order —
   drops articles and "to be" verbs, fronts time words (yesterday, tomorrow,
   now...), moves question words to the end, detaches negation and places
   it after the verb, and topicalizes simple subject-verb-object sentences
   (object moved to the front — e.g. "I like pizza" → `PIZZA I LIKE`).
   Because it's POS-based rather than keyword-matched, it generalizes to
   words it's never explicitly seen. Still a rule-based approximation, not
   a linguistically complete ASL grammar engine — real ASL grammar
   (classifiers, facial grammar, spatial referencing) goes well beyond
   word reordering; see the limitations note at the bottom of `gloss.py`.
2. **Playback**: for each gloss word, plays your recorded whole-word sign if
   one exists (`data/words/<word>.npy`), otherwise fingerspells it letter by
   letter using the per-letter templates from `data/landmarks.csv`.

Use `--no-gloss` to skip the grammar step and sign/spell the input literally,
word for word.

### Testing gloss.py as you extend it
```bash
python test_gloss.py
```
Runs a fixed set of varied example sentences (statements, questions,
negation, time-fronting, compound sentences) through `to_gloss()` so you can
eyeball the output after any change — useful for catching when a new rule
breaks an existing case.

### Tracking vocabulary coverage
```bash
python vocab_coverage.py
```
Checks `data/words/` against a target word list organized by category
(pronouns, question words, common verbs, time words, social phrases,
feelings) and reports what's recorded vs. still missing. Edit
`TARGET_VOCAB` in the script to adjust your plan.

## Next steps once this works
- Add J and Z (motion-based — needs a short sequence of frames, not one).
- Move from single letters to full WLASL words for the *recognition*
  direction too (sequence model: LSTM/Transformer over a window of frames,
  trained on your recorded word clips or the WLASL dataset).
- Swap the Random Forest for a small neural net once you have more data.
- Replace the skeleton animation with a proper 3D avatar for a more natural
  look.
- Extend `gloss.py`'s topicalization beyond simple single-clause SVO
  sentences, and add handling for compound/multi-clause sentences.

## Notes
- Works with a normal laptop webcam — no GPU required for this phase.
- All processing (MediaPipe + classifier) runs on CPU in real time.
