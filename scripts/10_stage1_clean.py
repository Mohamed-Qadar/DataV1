"""Raw text, steps 3.1-3.4 (+ dedup keys). One output shard per task, two workers.

Usage: python3 -I 10_stage1_clean.py <raw_dir> <work_dir> <glotlid_model>
Reads raw/ (never modifies it) and writes work/stage1/*.parquet + stage1_stats.json
"""
import glob
import gzip
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ops import op1_normalize, op2_apostrophes, op3_language, op4_length_quality, op5_dedup  # noqa: E402

RAW, WORK, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = os.path.join(WORK, "stage1")
LICENSE = {"fineweb2": "odc-by-1.0", "madlad400": "odc-by-1.0", "hplt2": "cc0-1.0", "finepdfs": "odc-by-1.0"}
SCHEMA = pa.schema([("text", pa.string()), ("source", pa.string()), ("url", pa.string()),
                    ("n_words", pa.int32()), ("h", pa.uint64()), ("sig", pa.binary(8 * op5_dedup.K))])


def tasks():
    t = []
    f = os.path.join(RAW, "fineweb2/data/som_Latn/train/000_00000.parquet")
    n = pq.ParquetFile(f).num_row_groups
    step = max(1, n // 16)
    for i in range(0, n, step):
        t.append(("fineweb2", "parquet", f, list(range(i, min(n, i + step))), "url"))
    shards = sorted(glob.glob(os.path.join(RAW, "madlad400/data-v1p5/so/*clean*.jsonl.gz")))
    for i in range(0, len(shards), 15):
        t.append(("madlad400", "gz", shards[i:i + 15], None, "url"))
    for f in sorted(glob.glob(os.path.join(RAW, "hplt2/som_Latn/*.parquet"))):
        n = pq.ParquetFile(f).num_row_groups
        half = max(1, (n + 1) // 2)
        for i in range(0, n, half):
            t.append(("hplt2", "parquet", f, list(range(i, min(n, i + half))), "u"))
    f = os.path.join(RAW, "finepdfs/data/som_Latn/train/000_00000.parquet")
    n = pq.ParquetFile(f).num_row_groups
    t.append(("finepdfs", "parquet", f, list(range(n)), "url"))
    return [(i,) + x for i, x in enumerate(t)]


def records(kind, path, groups, urlcol):
    if kind == "parquet":
        pf = pq.ParquetFile(path)
        for g in groups:
            tb = pf.read_row_group(g, columns=["text", urlcol])
            for text, url in zip(tb.column("text").to_pylist(), tb.column(urlcol).to_pylist()):
                yield text, url
    else:
        for p in path:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    try:
                        r = json.loads(ln)
                    except Exception:
                        continue
                    yield r.get("text"), r.get("url")


def init():
    op3_language.load(MODEL)


def run(task):
    tid, source, kind, path, groups, urlcol = task
    rnd = random.Random(tid)
    c = dict.fromkeys(["in", "rm_empty", "rm_encoding", "chg_normalize", "chg_apostrophe",
                       "rm_language", "rm_short", "rm_noise", "out", "words_out"], 0)
    removed = {k: [] for k in ("rm_encoding", "rm_language", "rm_short", "rm_noise")}
    seen = dict.fromkeys(removed, 0)

    def sample(reason, text):
        seen[reason] += 1
        item = {"source": source, "text": text[:600]}
        if len(removed[reason]) < 100:
            removed[reason].append(item)
        else:
            j = rnd.randrange(seen[reason])
            if j < 100:
                removed[reason][j] = item

    out_path = os.path.join(OUT, f"{tid:03d}_{source}.parquet")
    writer = pq.ParquetWriter(out_path + ".tmp", SCHEMA, compression="zstd", compression_level=3)
    buf = {k: [] for k in SCHEMA.names}

    def flush():
        if buf["text"]:
            writer.write_table(pa.table(buf, schema=SCHEMA))
            for k in buf:
                buf[k] = []

    for text, url in records(kind, path, groups, urlcol):
        c["in"] += 1
        if not isinstance(text, str) or not text.strip():
            c["rm_empty"] += 1
            continue
        t1 = op1_normalize.normalize(text)
        if t1 is None or not t1:
            c["rm_encoding"] += 1
            sample("rm_encoding", text)
            continue
        if t1 != text:
            c["chg_normalize"] += 1
        t2 = op2_apostrophes.fix_apostrophes(t1)
        if t2 != t1:
            c["chg_apostrophe"] += 1
        if not op3_language.is_somali(t2):
            c["rm_language"] += 1
            sample("rm_language", t2)
            continue
        nw = len(t2.split())
        if op4_length_quality.too_short(nw):
            c["rm_short"] += 1
            sample("rm_short", t2)
            continue
        if op4_length_quality.is_noise(t2):
            c["rm_noise"] += 1
            sample("rm_noise", t2)
            continue
        c["out"] += 1
        c["words_out"] += nw
        buf["text"].append(t2)
        buf["source"].append(source)
        buf["url"].append(url or "")
        buf["n_words"].append(nw)
        buf["h"].append(op5_dedup.exact_key(t2))
        buf["sig"].append(op5_dedup.signature(t2).tobytes())
        if len(buf["text"]) >= 4000:
            flush()
    flush()
    writer.close()
    os.replace(out_path + ".tmp", out_path)
    return tid, source, c, removed


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ts = tasks()
    print("tasks:", len(ts), flush=True)
    t0 = time.time()
    stats, removed_all = {}, {}
    with Pool(2, initializer=init) as pool:
        for tid, source, c, removed in pool.imap_unordered(run, ts):
            s = stats.setdefault(source, dict.fromkeys(c, 0))
            for k, v in c.items():
                s[k] += v
            for k, v in removed.items():
                removed_all.setdefault(k, []).extend(v)
            print(f"done {tid:03d} {source} in={c['in']} out={c['out']} t={time.time()-t0:.0f}s", flush=True)
    json.dump(stats, open(os.path.join(WORK, "stage1_stats.json"), "w"), indent=1)
    rnd = random.Random(0)
    for k, v in removed_all.items():
        rnd.shuffle(v)
        with open(os.path.join(WORK, f"removed_{k}.jsonl"), "w", encoding="utf-8") as f:
            for item in v[:100]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("STAGE1 DONE", json.dumps(stats), flush=True)
