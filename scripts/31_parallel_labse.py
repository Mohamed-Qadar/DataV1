"""Parallel pairs, step 3.7 — LaBSE cosine similarity for every candidate pair.

Usage: python3 -I 31_parallel_labse.py <work_dir> <labse_dir>
Writes work/parallel_labse.npy (one similarity per line of parallel_candidates.jsonl).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

WORK, LABSE = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
model = SentenceTransformer(LABSE, device="cpu")
model.max_seq_length = 128
rows = [json.loads(l) for l in open(os.path.join(WORK, "parallel_candidates.jsonl"), encoding="utf-8")]
n = len(rows)
order = sorted(range(n), key=lambda i: len(rows[i]["so"]) + len(rows[i]["en"]))
sims = np.zeros(n, dtype=np.float32)
done_path = os.path.join(WORK, "parallel_labse.npy")
B = 256
t0 = time.time()
for s in range(0, n, B):
    idx = order[s:s + B]
    a = model.encode([rows[i]["so"] for i in idx], batch_size=B, convert_to_numpy=True, normalize_embeddings=True)
    b = model.encode([rows[i]["en"] for i in idx], batch_size=B, convert_to_numpy=True, normalize_embeddings=True)
    sims[idx] = (a * b).sum(axis=1)
    if (s // B) % 20 == 0:
        el = time.time() - t0
        print(f"{s+len(idx)}/{n} pairs  {el:.0f}s  eta {el/(s+len(idx))*(n-s-len(idx)):.0f}s", flush=True)
np.save(done_path, sims)
print("LABSE DONE", n, f"{time.time()-t0:.0f}s", flush=True)
