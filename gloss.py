"""
English -> ASL gloss converter, using spaCy's dependency parse (POS tags +
grammatical roles) instead of fixed word lists. This generalizes much
better than keyword matching — it correctly handles verbs, pronouns, and
objects it has never explicitly seen, because it reasons about grammatical
role rather than specific words.

Patterns implemented (still a simplified approximation — see note at the
bottom of this file):
  - Drop articles (a, an, the) and "to be" verbs (am, is, are...).
  - Drop "do/does/did" auxiliaries (their grammatical job, e.g. marking a
    question, is handled separately).
  - Time-fronting: time expressions (yesterday, tomorrow, now, later...)
    move to the front of the sentence.
  - Wh-movement: question words (what, where, who...) move to the end for
    statements-turned-questions, matching common ASL question structure.
  - Negation: "not"/"n't" is detached from its auxiliary and placed as its
    own NOT token right after the main verb, which is where negation
    typically sits in ASL rather than attached to an auxiliary.
  - Topicalization: for simple statements with a clear subject-verb-object
    structure, the object is fronted (topic-comment order) — e.g.
    "I like pizza" -> "PIZZA I LIKE". This only fires for a single, simple
    main clause; it backs off for complex sentences, since getting this
    wrong is worse than leaving the order alone.
  - Lemmatization: words are reduced to base form (eating -> eat, ate -> eat)
    using spaCy's lemmatizer, so different tenses map to the same sign.

Setup:
  pip install spacy
  python -m spacy download en_core_web_sm

If spaCy or the model isn't installed, to_gloss() raises a clear error
telling you how to install it — there's no silent fallback, since mixing
two different gloss strategies would make output inconsistent.
"""

try:
    import spacy
except ImportError:
    spacy = None

_NLP = None

TIME_WORDS = {
    "yesterday", "today", "tomorrow", "now", "later", "soon",
    "morning", "afternoon", "evening", "night", "already", "before", "after",
}


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:
        raise RuntimeError(
            "spaCy is not installed. Run: pip install spacy && "
            "python -m spacy download en_core_web_sm"
        )
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Run: "
            "python -m spacy download en_core_web_sm"
        )
    return _NLP


def to_gloss(sentence):
    """Convert an English sentence into a list of ASL gloss tokens (uppercase)."""
    nlp = _get_nlp()
    doc = nlp(sentence)

    is_question = sentence.strip().endswith("?")

    wh_token = None
    neg_present = False
    time_tokens = []
    subj_token = None
    verb_token = None
    obj_token = None
    kept = []

    for tok in doc:
        lemma = tok.lemma_.lower()
        text_lower = tok.text.lower()

        if tok.pos_ == "DET":  # articles: a, an, the
            continue
        if lemma == "be" and tok.pos_ in ("AUX", "VERB"):
            continue
        if tok.pos_ == "AUX" and lemma == "do":
            continue
        if tok.dep_ == "neg" or text_lower in ("not", "n't"):
            neg_present = True
            continue
        if tok.tag_ in ("WP", "WRB", "WDT") or lemma in (
            "what", "where", "when", "who", "why", "how", "which"
        ):
            wh_token = lemma
            continue
        if text_lower in TIME_WORDS:
            time_tokens.append(text_lower)
            continue

        if tok.dep_ == "nsubj" and subj_token is None:
            subj_token = lemma
        elif tok.dep_ == "ROOT" and tok.pos_ == "VERB" and verb_token is None:
            verb_token = lemma
        elif tok.dep_ in ("dobj", "attr") and obj_token is None:
            obj_token = lemma

        kept.append(lemma)

    # Try topicalization only for a clean, simple subject-verb-object
    # statement — if the sentence doesn't cleanly fit that shape, leave
    # word order as encountered rather than guessing.
    simple_svo = (
        not is_question
        and subj_token and verb_token and obj_token
        and {subj_token, verb_token, obj_token} <= set(kept)
    )

    if simple_svo:
        remaining = [w for w in kept if w not in (subj_token, verb_token, obj_token)]
        ordered = [obj_token, subj_token, verb_token] + remaining
    else:
        ordered = kept

    if neg_present and verb_token in ordered:
        idx = ordered.index(verb_token)
        ordered = ordered[: idx + 1] + ["not"] + ordered[idx + 1:]
    elif neg_present:
        ordered = ordered + ["not"]

    gloss_tokens = time_tokens + ordered
    if wh_token:
        if is_question:
            gloss_tokens = gloss_tokens + [wh_token]
        else:
            gloss_tokens = [wh_token] + gloss_tokens

    return [t.upper() for t in gloss_tokens]


# NOTE ON LIMITATIONS:
# This is still a rough, rule-based approximation, not a linguistically
# complete ASL grammar engine. It does not model classifiers, spatial
# referencing, non-manual grammar (facial expression, eyebrow raises),
# aspectual marking, or topicalization beyond simple single-clause SVO
# sentences. Complex, compound, or multi-clause sentences will often come
# out close to their original English order rather than "correct" ASL
# order. Treat gloss output as a reasonable first-pass approximation for a
# prototype, not a certified translation.


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("English sentence: ")
    print(" ".join(to_gloss(text)))
