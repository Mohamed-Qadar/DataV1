"""Parallel pairs: cleaning steps 3.1-3.6 (cheap filters) -> candidate pool for LaBSE.

Usage: python3 -I 30_parallel_candidates.py <raw_dir> <work_dir> <repo_dir> <glotlid_model>
Human-translated sources (Infopankki, SMOL) are taken in full. NLLB (10.2M mined pairs)
is first cut to its top 10% by the provided mining score, cheaply filtered, then a
random pool of NLLB_POOL pairs goes on to the LaBSE check.
"""
import io
import json
import os
import random
import re
import sys
import zipfile

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ops import op1_normalize, op2_apostrophes, op3_language, op4_length_quality, op7_test_leakage  # noqa: E402

RAW, WORK, REPO, MODEL = sys.argv[1:5]
NLLB_POOL = 90_000
op3_language.load(MODEL)
op7_test_leakage.load_belebele(os.path.join(REPO, "test_quarantine", "belebele", "data", "som_Latn.jsonl"))
_URL = re.compile(r"https?://|www\.|\.com\b|\.net\b|\.org\b", re.I)
_NUM = re.compile(r"\d+")
_SPAM = op4_length_quality._SPAM
rnd = random.Random(0)
stats, removed, seen_rm = {}, {}, {}
seen_so, seen_en = set(), set()


def drop(source, reason, so, en):
    stats[source][reason] = stats[source].get(reason, 0) + 1
    key = reason
    seen_rm[key] = seen_rm.get(key, 0) + 1
    lst = removed.setdefault(key, [])
    item = {"source": source, "so": so[:300], "en": en[:300]}
    if len(lst) < 100:
        lst.append(item)
    else:
        j = rnd.randrange(seen_rm[key])
        if j < 100:
            lst[j] = item
    return None


def clean(source, so, en, sentence_level=True):
    st = stats.setdefault(source, {"in": 0, "out": 0})
    st["in"] += 1
    if not so or not en or not so.strip() or not en.strip():
        return drop(source, "rm_empty", so or "", en or "")
    so = op1_normalize.normalize(so)
    en = op1_normalize.normalize(en)
    if not so or not en:
        return drop(source, "rm_encoding", so or "", en or "")
    so = op2_apostrophes.fix_apostrophes(so).replace("\n", " ") if sentence_level else op2_apostrophes.fix_apostrophes(so)
    en = op2_apostrophes.fix_apostrophes(en).replace("\n", " ") if sentence_level else op2_apostrophes.fix_apostrophes(en)
    ws, we = len(so.split()), len(en.split())
    if ws < 4 or we < 4:
        return drop(source, "rm_short", so, en)
    if sentence_level and (ws > 80 or we > 80):
        return drop(source, "rm_long", so, en)
    ratio = len(so) / len(en)
    if ratio < 0.5 or ratio > 2.5:
        return drop(source, "rm_length_ratio", so, en)
    if so.lower() == en.lower():
        return drop(source, "rm_identical", so, en)
    if _URL.search(so) or _URL.search(en) or "@" in so or "@" in en:
        return drop(source, "rm_url_email", so, en)
    if sorted(_NUM.findall(so)) != sorted(_NUM.findall(en)):
        return drop(source, "rm_number_mismatch", so, en)
    for side in (so, en):
        letters = sum(c.isalpha() for c in side)
        if letters / len(side) < 0.6 or sum(c.isupper() for c in side) / max(1, letters) > 0.3:
            return drop(source, "rm_noise", so, en)
    if _SPAM.search(so) or _SPAM.search(en):
        return drop(source, "rm_spam", so, en)
    lab, p = op3_language.lang_of(so)
    if lab != "__label__som_Latn" or p < 0.5:
        return drop(source, "rm_language_so", so, en)
    lab, p = op3_language.lang_of(en)
    if lab != "__label__eng_Latn" or p < 0.5:
        return drop(source, "rm_language_en", so, en)
    ks, ke = so.lower(), en.lower()
    if ks in seen_so or ke in seen_en:
        return drop(source, "rm_duplicate", so, en)
    if op7_test_leakage.sentence_in_test(so):
        return drop(source, "rm_test_leak", so, en)
    seen_so.add(ks)
    seen_en.add(ke)
    st["out"] += 1
    return so, en


out = open(os.path.join(WORK, "parallel_candidates.jsonl"), "w", encoding="utf-8")


def emit(source, license_, so, en, level, score=None):
    out.write(json.dumps({"so": so, "en": en, "source": source, "license": license_, "level": level,
                          "mining_score": score}, ensure_ascii=False) + "\n")


# 1) Google SMOL (professional human translation) — sentences and documents
for ln in open(os.path.join(RAW, "smol/smolsent/en_so.jsonl"), encoding="utf-8"):
    r = json.loads(ln)
    c = clean("smol", r["trg"], r["src"])
    if c:
        emit("smol", "cc-by-4.0", c[0], c[1], "sentence")
for ln in open(os.path.join(RAW, "smol/smoldoc/en_so.jsonl"), encoding="utf-8"):
    r = json.loads(ln)
    srcs, trgs = r.get("srcs") or [], r.get("trgs") or []
    if len(srcs) != len(trgs) or not srcs:
        stats.setdefault("smol", {"in": 0, "out": 0})
        drop("smol", "rm_misaligned_doc", " ".join(trgs), " ".join(srcs))
        continue
    c = clean("smol", " ".join(trgs), " ".join(srcs), sentence_level=False)
    if c:
        emit("smol", "cc-by-4.0", c[0], c[1], "paragraph")

# 2) OPUS Infopankki (human-translated public information)
tb = pq.read_table(os.path.join(RAW, "infopankki/en-so/train-00000-of-00001.parquet"))
for r in tb.column("translation").to_pylist():
    c = clean("infopankki", r.get("so"), r.get("en"))
    if c:
        emit("infopankki", "cc-by-4.0", c[0], c[1], "sentence")
print("human sources:", json.dumps({k: v for k, v in stats.items()}), flush=True)

# 3) NLLB mined bitext — top 10% by mining score, cheap filters, random pool
z = zipfile.ZipFile(os.path.join(RAW, "nllb/en-so.txt.zip"))
name = {n.rsplit(".", 1)[-1]: n for n in z.namelist() if n.startswith("NLLB")}
scores = np.array(z.read(name["scores"]).split(), dtype=np.float32)
thr = float(np.quantile(scores, 0.90))
sel = scores >= thr
last = int(np.flatnonzero(sel).max())
print("nllb pairs:", len(scores), "score threshold (top 10%):", round(thr, 4), "selected:", int(sel.sum()),
      "last index:", last, flush=True)
stats["nllb"] = {"in_total": int(len(scores)), "rm_below_top10pct_score": int(len(scores) - sel.sum()),
                 "in": 0, "out": 0}
pool = []
f_so = io.TextIOWrapper(z.open(name["so"]), encoding="utf-8", errors="replace")
f_en = io.TextIOWrapper(z.open(name["en"]), encoding="utf-8", errors="replace")
for i, (so, en) in enumerate(zip(f_so, f_en)):
    if i > last:
        break
    if not sel[i]:
        continue
    c = clean("nllb", so.rstrip("\n"), en.rstrip("\n"))
    if c:
        pool.append((c[0], c[1], float(scores[i])))
    if i % 200000 == 0:
        print("  nllb line", i, "kept", len(pool), flush=True)
stats["nllb"]["passed_cheap_filters"] = len(pool)
rnd2 = random.Random(1)
rnd2.shuffle(pool)
for so, en, sc in pool[:NLLB_POOL]:
    emit("nllb", "odc-by-1.0", so, en, "sentence", sc)
stats["nllb"]["sent_to_labse"] = min(len(pool), NLLB_POOL)
stats["nllb"]["not_scored_reserve"] = max(0, len(pool) - NLLB_POOL)
out.close()
json.dump(stats, open(os.path.join(WORK, "parallel_stats.json"), "w"), indent=1)
for k, v in removed.items():
    with open(os.path.join(WORK, f"removed_parallel_{k}.jsonl"), "w", encoding="utf-8") as f:
        for item in v:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("CANDIDATES DONE", json.dumps(stats), flush=True)
