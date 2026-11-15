"""Step 3.3 — language identification with GlotLID (fastText). Keep Somali only."""
import fasttext

_MODEL = None
SOMALI = "__label__som_Latn"


def load(path):
    global _MODEL
    _MODEL = fasttext.load_model(path)


def _probe(chunk):
    labels, probs = _MODEL.predict(chunk.replace("\n", " "), k=1)
    return labels[0], float(probs[0])


def lang_of(text):
    """Label and probability for a short text (sentence level)."""
    return _probe(text[:1500])


def is_somali(text, min_prob=0.7):
    """Document level: the start and, for long documents, the middle must both be Somali."""
    lab, p = _probe(text[:1500])
    if lab != SOMALI or p < min_prob:
        return False
    if len(text) > 4000:
        mid = len(text) // 2
        lab, p = _probe(text[mid:mid + 1500])
        if lab != SOMALI or p < min_prob:
            return False
    return True
