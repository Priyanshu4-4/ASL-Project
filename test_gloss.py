"""
Regression test set for gloss.py. Run this after any change to gloss.py
and eyeball the output — gloss "correctness" is subjective, so there's no
automated pass/fail, just a fixed set of varied sentences to check by eye
each time so you notice if a new rule breaks an old case.

Usage:
  python test_gloss.py
"""

from gloss import to_gloss

TEST_SENTENCES = [
    # Simple statements (SVO / topicalization candidates)
    "I like pizza.",
    "She eats breakfast.",
    "We love dogs.",

    # Questions
    "What is your name?",
    "Where do you work?",
    "Did you eat yesterday?",
    "Are you hungry?",

    # Negation
    "I do not like coffee.",
    "She is not happy.",
    "I don't understand.",

    # Time-fronting
    "I will go tomorrow.",
    "We talked yesterday.",
    "I am busy now.",

    # Articles / be-verbs stripped
    "The cat is sleeping.",
    "He is a teacher.",

    # More complex / compound (expect imperfect output — these are the
    # cases gloss.py's limitations note calls out)
    "I went to the store and bought milk.",
    "If you are free, let's meet tomorrow.",

    # Plain greetings / short phrases
    "Hello, how are you?",
    "Thank you very much.",
    "Nice to meet you.",
]


def main():
    for sentence in TEST_SENTENCES:
        gloss = to_gloss(sentence)
        print(f"{sentence!r:45} -> {' '.join(gloss)}")


if __name__ == "__main__":
    main()
