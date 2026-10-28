"""
Rule-based English -> ASL gloss converter.

ASL has its own grammar, not word-for-word English. This is a simplified,
beginner-level approximation of real gloss conventions:
  - Drop articles (a, an, the) — ASL doesn't use them.
  - Drop "to be" verbs (am, is, are, was, were) — ASL usually omits them.
  - Drop "do/does/did" auxiliary verbs in questions.
  - Move time words (yesterday, tomorrow, now, later, today...) to the
    front — ASL commonly states time first (time-topic-comment order).
  - Move question words (what, where, when, who, why, how) to the end —
    common in ASL questions, though front-placement also occurs.
  - Lightly lemmatize remaining words (strip -ing/-ed/-s) so different
    tenses map to the same sign; a handful of irregular verbs are
    special-cased.

This is NOT linguistically complete — real ASL grammar (classifiers,
non-manual markers, spatial referencing, aspect marking) is far richer
than word reordering. Treat this as a rough approximation for a starter
project, not a certified translation.
"""

import re

ARTICLES = {"a", "an", "the"}
BE_VERBS = {"am", "is", "are", "was", "were", "be", "been", "being"}
DO_AUX = {"do", "does", "did"}
TIME_WORDS = {
    "yesterday", "today", "tomorrow", "now", "later", "soon",
    "morning", "afternoon", "evening", "night", "already", "before", "after",
}
WH_WORDS = {"what", "where", "when", "who", "why", "how", "which"}

IRREGULAR_LEMMAS = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "went": "go", "gone": "go", "goes": "go",
    "had": "have", "has": "have",
    "did": "do", "does": "do",
    "said": "say", "says": "say",
    "ate": "eat", "eaten": "eat",
    "saw": "see", "seen": "see",
    "came": "come", "comes": "come",
}


def _lemmatize(word):
    if word in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[word]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def to_gloss(sentence):
    """Convert an English sentence into a list of ASL gloss tokens (uppercase)."""
    words = re.findall(r"[a-zA-Z']+", sentence.lower())

    is_question = sentence.strip().endswith("?")
    wh_word = None

    kept = []
    for w in words:
        if w in ARTICLES:
            continue
        if w in BE_VERBS:
            continue
        if w in DO_AUX:
            continue
        if w in WH_WORDS:
            wh_word = w
            continue
        kept.append(w)

    time_tokens = [w for w in kept if w in TIME_WORDS]
    rest = [w for w in kept if w not in TIME_WORDS]

    lemmatized = [_lemmatize(w) for w in rest]

    gloss_tokens = time_tokens + lemmatized
    if is_question and wh_word:
        gloss_tokens.append(wh_word)
    elif wh_word:
        gloss_tokens.insert(0, wh_word)

    return [t.upper() for t in gloss_tokens]


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("English sentence: ")
    print(" ".join(to_gloss(text)))
