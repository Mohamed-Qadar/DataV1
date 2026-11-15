"""Raw text, steps 3.5 (exact + near dedup) and 3.6 (PII), then write final/v1.0 shards.

Usage: python3 -I 20_stage2_dedup_write.py <work_dir> <repo_dir>
Output: final/v1.0/raw_text/{train,validation,test}-NNNNN.parquet (id, text, source, license)
"""
import glob
import json
import os
import random
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ops import op5_dedup, op6_pii, op7_test_leakage  # noqa: E402

WORK, REPO = sys.argv[1], sys.argv[2]
FINAL = os.path.join(REPO, "final", "v1.0", "raw_text")
LICENSE = {"fineweb2": "odc-by-1.0", "madlad400": "odc-by-1.0", "hplt2": "cc0-1.0", "finepdfs": "odc-by-1.0"}
PRIORITY = {"fineweb2": 0, "finepdfs": 1, "madlad400": 2, "hplt2": 3}  # first seen wins on exact ties
SHARD_RAW_BYTES = 120_000_000
OUT_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string()), ("source", pa.string()),
                        ("license", pa.string())])
t0 = time.time()

files = sorted(glob.glob(os.path.join(WORK, "stage1", "*.parquet")),
               key=lambda p: (PRIORITY[os.path.basename(p)[4:-8]], p))
hs, nws, sigs, srcs = [], [], [], []
for f in files:
    tb = pq.read_table(f, columns=["h", "n_words", "sig", "source"])
    hs.append(tb.column("h").to_numpy())
    nws.append(tb.column("n_words").to_numpy())
    raw = b"".join(tb.column("sig").to_pylist())
    sigs.append(np.frombuffer(raw, dtype=np.uint64).reshape(-1, op5_dedup.K))
    srcs.append(np.full(tb.num_rows, PRIORITY[os.path.basename(f)[4:-8]], dtype=np.int8))
h = np.concatenate(hs); nw = np.concatenate(nws); sig = np.concatenate(sigs); src = np.concatenate(srcs)
N = len(h)
print("docs after steps 3.1-3.4:", N, f"t={time.time()-t0:.0f}s", flush=True)

# exact duplicates: first occurrence wins
_, first = np.unique(h, return_index=True)
status = np.ones(N, dtype=np.int8)          # 1 = exact duplicate
status[first] = 0                            # 0 = keep
n_exact = int((status == 1).sum())
print("exact duplicates removed:", n_exact, flush=True)

# near duplicates among survivors
alive = np.flatnonzero(status == 0)
keep = op5_dedup.near_duplicate_clusters(sig[alive], nw[alive])
status[alive[~keep]] = 2                     # 2 = near duplicate
n_near = int((status == 2).sum())
print("near duplicates removed:", n_near, f"t={time.time()-t0:.0f}s", flush=True)
del sig

# final write with PII masking and deterministic 96/2/2 split
os.makedirs(FINAL, exist_ok=True)
n_test = op7_test_leakage.load_belebele(os.path.join(REPO, "test_quarantine", "belebele", "data", "som_Latn.jsonl"))
print("test texts indexed for leakage guard:", n_test, flush=True)
rnd = random.Random(0)
names = {v: k for k, v in PRIORITY.items()}
stats = {"in": N, "rm_exact_dup": n_exact, "rm_near_dup": n_near, "rm_test_leak": 0, "rm_short_after_mask": 0, "out": 0, "pii_docs": 0,
         "emails_masked": 0, "phones_masked": 0, "by_source": {}, "by_split": {}, "by_license": {},
         "exact_dup_by_source": {}, "near_dup_by_source": {}}
for code, key in ((1, "exact_dup_by_source"), (2, "near_dup_by_source")):
    for p, name in names.items():
        stats[key][name] = int(((status == code) & (src == p)).sum())
removed = {"rm_exact_dup": [], "rm_near_dup": [], "rm_test_leak": [], "pii_masked_examples": []}
seen = dict.fromkeys(removed, 0)


def sample(key, item):
    seen[key] += 1
    if len(removed[key]) < 100:
        removed[key].append(item)
    else:
        j = rnd.randrange(seen[key])
        if j < 100:
            removed[key][j] = item


class Split:
    def __init__(self, name):
        self.name, self.idx, self.bytes, self.buf = name, 0, 0, {k: [] for k in OUT_SCHEMA.names}
        self.docs = self.words = 0

    def add(self, rec, nwords):
        for k, v in zip(OUT_SCHEMA.names, rec):
            self.buf[k].append(v)
        self.bytes += len(rec[1].encode("utf-8"))
        self.docs += 1
        self.words += nwords
        if self.bytes >= SHARD_RAW_BYTES:
            self.flush()

    def flush(self):
        if not self.buf["id"]:
            return
        path = os.path.join(FINAL, f"{self.name}-{self.idx:05d}.parquet")
        pq.write_table(pa.table(self.buf, schema=OUT_SCHEMA), path, compression="zstd", compression_level=12)
        self.idx += 1
        self.bytes = 0
        self.buf = {k: [] for k in OUT_SCHEMA.names}


splits = {n: Split(n) for n in ("train", "validation", "test")}
pos = 0
counter = 0
for f in files:
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=4000, columns=["text", "source", "n_words"]):
        texts = batch.column("text").to_pylist()
        sources = batch.column("source").to_pylist()
        words = batch.column("n_words").to_pylist()
        for text, source, nwords in zip(texts, sources, words):
            st = status[pos]
            pos += 1
            if st == 1:
                sample("rm_exact_dup", {"source": source, "text": text[:600]})
                continue
            if st == 2:
                sample("rm_near_dup", {"source": source, "text": text[:600]})
                continue
            if op7_test_leakage.contains_test_text(text):
                stats["rm_test_leak"] += 1
                sample("rm_test_leak", {"source": source, "text": text[:600]})
                continue
            masked, n_mail, n_phone = op6_pii.mask_pii(text)
            if n_mail or n_phone:
                nwords = len(masked.split())
                if nwords < 50:          # masking can push a borderline document under the limit
                    stats["rm_short_after_mask"] += 1
                    continue
                stats["pii_docs"] += 1
                stats["emails_masked"] += n_mail
                stats["phones_masked"] += n_phone
                i = masked.find("[EMAIL]") if n_mail else masked.find("[PHONE]")
                sample("pii_masked_examples", {"source": source, "text": masked[max(0, i - 200):i + 200]})
            counter += 1
            doc_id = f"so-raw-{counter:07d}"
            r = xxhash.xxh64_intdigest(doc_id.encode()) % 100
            split = "validation" if r < 2 else "test" if r < 4 else "train"
            lic = LICENSE[source]
            splits[split].add((doc_id, masked, source, lic), nwords)
            stats["out"] += 1
            bs = stats["by_source"].setdefault(source, {"docs": 0, "words": 0})
            bs["docs"] += 1; bs["words"] += nwords
            bl = stats["by_license"].setdefault(lic, {"docs": 0, "words": 0})
            bl["docs"] += 1; bl["words"] += nwords
    print("written", os.path.basename(f), stats["out"], f"t={time.time()-t0:.0f}s", flush=True)
assert pos == N, (pos, N)
for s in splits.values():
    s.flush()
    stats["by_split"][s.name] = {"docs": s.docs, "words": s.words, "files": s.idx}
json.dump(stats, open(os.path.join(WORK, "stage2_stats.json"), "w"), indent=1)
for k, v in removed.items():
    with open(os.path.join(WORK, f"removed_{k}.jsonl"), "w", encoding="utf-8") as fo:
        for item in v:
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
print("STAGE2 DONE", json.dumps(stats), flush=True)
