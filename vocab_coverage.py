"""
Tracks which target vocabulary words you've recorded (data/words/<word>.npy)
versus which are still missing, organized by category — so you can see
coverage gaps at a glance instead of guessing.

Edit TARGET_VOCAB below to add/remove categories or words as your plan
changes. Run after recording a batch of words to see progress.

Usage:
  python vocab_coverage.py
"""

import os

WORDS_DIR = os.path.join("data", "words")

TARGET_VOCAB = {
    "pronouns": [
        "i", "you", "he", "she", "we", "they", "it", "this", "that",
    ],
    "question_words": [
        "what", "where", "who", "why", "how", "when", "which",
    ],
    "common_verbs": [
        "want", "need", "go", "have", "like", "eat", "help", "see",
        "know", "think", "make", "give", "take", "come",
    ],
    "time_words": [
        "now", "later", "today", "yesterday", "tomorrow", "morning",
        "evening", "night", "already",
    ],
    "social_phrases": [
        "hello", "thankyou", "please", "sorry", "yes", "no",
        "goodbye", "welcome",
    ],
    "feelings": [
        "happy", "sad", "angry", "tired", "hungry", "excited", "scared",
    ],
}


def recorded_words():
    if not os.path.isdir(WORDS_DIR):
        return set()
    return {f[:-4] for f in os.listdir(WORDS_DIR) if f.endswith(".npy")}


def main():
    have = recorded_words()
    total_target = 0
    total_have = 0

    for category, words in TARGET_VOCAB.items():
        done = [w for w in words if w in have]
        missing = [w for w in words if w not in have]
        total_target += len(words)
        total_have += len(done)

        print(f"\n{category} ({len(done)}/{len(words)}):")
        print(f"  have:    {', '.join(done) if done else '(none yet)'}")
        print(f"  missing: {', '.join(missing) if missing else '(all done!)'}")

    extra = have - {w for words in TARGET_VOCAB.values() for w in words}
    if extra:
        print(f"\nRecorded but not in target list ({len(extra)}): {', '.join(sorted(extra))}")

    print(f"\nOverall: {total_have}/{total_target} target words recorded "
          f"({total_have / total_target:.0%})" if total_target else "No target words defined.")


if __name__ == "__main__":
    main()
