"""Step 3.5 — exact duplicates (hash) and near duplicates (MinHash + LSH).

Exact key: xxh64 of lower-cased, whitespace-collapsed text.
Near duplicates: one-permutation MinHash over word 5-gram shingles, 32 bins,
LSH with 8 bands x 4 rows, verified at >= 26/32 matching bins (Jaccard ~0.8).
The longest document of each cluster is kept.
"""
import re

import numpy as np
import xxhash

K = 32
BANDS, ROWS = 8, 4
MIN_MATCH = 26
_TOK = re.compile(rb"[a-z']+")
_WS = re.compile(r"\s+")
_P = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
               0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD], dtype=np.uint64)
_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)


def exact_key(text):
    return xxhash.xxh64_intdigest(_WS.sub(" ", text.lower()).strip().encode("utf-8"))


def signature(text):
    words = _TOK.findall(text.lower().encode("utf-8"))
    n = len(words)
    sig = np.full(K, _MAX, dtype=np.uint64)
    if n < 5:
        return sig
    wh = np.fromiter((xxhash.xxh64_intdigest(w) for w in words), dtype=np.uint64, count=n)
    m = n - 4
    with np.errstate(over="ignore"):
        s = wh[0:m] * _P[0]
        for j in range(1, 5):
            s = s + wh[j:j + m] * _P[j]
        s ^= s >> np.uint64(33)
        s = s * _P[3]
        s ^= s >> np.uint64(29)
    np.minimum.at(sig, (s >> np.uint64(59)).astype(np.int64), s)
    empty = sig == _MAX
    if empty.any():  # densify: borrow from the next filled bin
        filled = np.flatnonzero(~empty)
        if len(filled):
            for i in np.flatnonzero(empty):
                nxt = filled[np.searchsorted(filled, i) % len(filled)]
                sig[i] = sig[nxt]
    return sig


def near_duplicate_clusters(sigs, n_words):
    """sigs: (N, 32) uint64. Returns boolean keep-mask (longest doc per cluster)."""
    n = len(sigs)
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    with np.errstate(over="ignore"):
        for b in range(BANDS):
            cols = sigs[:, b * ROWS:(b + 1) * ROWS]
            key = cols[:, 0] * _P[0]
            for j in range(1, ROWS):
                key = (key ^ (key >> np.uint64(31))) * _P[1] + cols[:, j] * _P[j]
            order = np.argsort(key, kind="stable")
            sk = key[order]
            starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
            ends = np.r_[starts[1:], n]
            multi = np.flatnonzero(ends - starts > 1)
            for g in multi:
                members = order[starts[g]:ends[g]]
                rep = members[0]
                match = (sigs[members[1:]] == sigs[rep]).sum(axis=1)
                for m_ in members[1:][match >= MIN_MATCH]:
                    ra, rb = find(rep), find(m_)
                    if ra != rb:
                        parent[rb] = ra
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    # keep the longest document in each cluster (ties: first seen)
    order = np.lexsort((np.arange(n), -n_words.astype(np.int64), roots))
    first = np.r_[True, roots[order][1:] != roots[order][:-1]]
    keep = np.zeros(n, dtype=bool)
    keep[order[first]] = True
    return keep
