"""Guard: no text from the quarantined test sets may enter training data.

Builds hashes of every 10-word window of the test texts (Belebele passages and
questions, which are FLORES-200 passages) and flags any training text that
contains one of those windows.
"""
import json
import re

import numpy as np
import xxhash

W = 10
_TOK = re.compile(rb"[a-z']+")
_M = np.uint64(0x9E3779B97F4A7C15)
_C = np.uint64(0xD6E8FEB86659FD93)
_INDEX = None
_PASSAGES = ""


def _windows(text):
    words = _TOK.findall(text.lower().replace("’", "'").encode("utf-8"))
    n = len(words)
    if n < W:
        return np.empty(0, dtype=np.uint64)
    wh = np.fromiter((xxhash.xxh64_intdigest(w) for w in words), dtype=np.uint64, count=n)
    m = n - W + 1
    with np.errstate(over="ignore"):
        s = np.zeros(m, dtype=np.uint64)
        mult = _M
        for j in range(W):
            s = s + wh[j:j + m] * mult
            mult = mult * _M + np.uint64(j + 1)
        s ^= s >> np.uint64(33)
        s = s * _C
        s ^= s >> np.uint64(29)
    return s


def _norm(text):
    return " ".join(w.decode() for w in _TOK.findall(text.lower().replace("’", "'").encode("utf-8")))


def load_belebele(path):
    """Index a Belebele jsonl file. Returns number of distinct test texts."""
    global _INDEX, _PASSAGES
    texts = set()
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        for k in ("flores_passage", "question"):
            if r.get(k):
                texts.add(r[k])
    hashes = [_windows(t) for t in texts]
    _INDEX = np.unique(np.concatenate(hashes)) if hashes else np.empty(0, dtype=np.uint64)
    _PASSAGES = " \n ".join(_norm(t) for t in texts)
    return len(texts)


def contains_test_text(text):
    """Document level: True if any 10-word window of a test text occurs in `text`."""
    s = _windows(text)
    if not len(s) or not len(_INDEX):
        return False
    pos = np.searchsorted(_INDEX, s)
    pos[pos >= len(_INDEX)] = len(_INDEX) - 1
    return bool((_INDEX[pos] == s).any())


def sentence_in_test(sentence):
    """Sentence level: True if the sentence (>= 5 words) appears inside a test passage."""
    n = _norm(sentence)
    return len(n.split()) >= 5 and n in _PASSAGES
