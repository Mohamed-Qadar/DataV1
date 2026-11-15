"""Step 5 — validate every record of final/v1.0 against the schema.

Usage: python3 -I 40_validate.py <repo_dir> <work_dir>
"""
import glob
import json
import os
import sys

import pyarrow.parquet as pq

REPO, WORK = sys.argv[1], sys.argv[2]
FINAL = os.path.join(REPO, "final", "v1.0")
LICENSES = {"cc0-1.0", "odc-by-1.0", "cc-by-4.0"}
TEMPLATE_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "[INST]", "[/INST]", "<s>", "</s>",
                   "<|user|>", "<|assistant|>", "<|system|>")
report = {"translation": {}, "raw_text": {}, "errors": []}


def err(msg):
    if len(report["errors"]) < 50:
        report["errors"].append(msg)


ids = set()
for name in ("train", "validation", "test"):
    path = os.path.join(FINAL, "translation", f"{name}.jsonl")
    raw = open(path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        err(f"{path}: BOM found")
    raw.decode("utf-8")  # raises if not valid UTF-8
    n = bad = 0
    for i, ln in enumerate(raw.decode("utf-8").split("\n")):
        if not ln:
            continue
        n += 1
        try:
            r = json.loads(ln)
            assert set(r) == {"id", "messages", "source", "category", "language", "license"}, "fields"
            assert isinstance(r["id"], str) and r["id"] not in ids, "id"
            ids.add(r["id"])
            m = r["messages"]
            assert isinstance(m, list) and len(m) >= 2 and len(m) % 2 == 0, "messages length"
            for j, msg in enumerate(m):
                assert set(msg) == {"role", "content"}, "message keys"
                assert msg["role"] == ("user" if j % 2 == 0 else "assistant"), "role order"
                assert isinstance(msg["content"], str) and msg["content"].strip(), "empty content"
                assert not any(t in msg["content"] for t in TEMPLATE_TOKENS), "chat template token"
                assert "�" not in msg["content"], "replacement char"
            assert r["category"] == "translation" and r["language"] == "so-en", "category/language"
            assert r["license"] in LICENSES, "license"
            assert r["source"] in {"infopankki", "smol", "nllb"}, "source"
        except Exception as e:
            bad += 1
            err(f"{name}.jsonl line {i+1}: {e}")
    report["translation"][name] = {"records": n, "invalid": bad}

ids = set()
for name in ("train", "validation", "test"):
    files = sorted(glob.glob(os.path.join(FINAL, "raw_text", f"{name}-*.parquet")))
    n = bad = 0
    big = 0
    for p in files:
        size = os.path.getsize(p)
        big = max(big, size)
        tb = pq.read_table(p)
        if tb.schema.names != ["id", "text", "source", "license"]:
            err(f"{p}: schema {tb.schema.names}")
        for c in tb.schema.names:
            if tb.column(c).null_count:
                bad += tb.column(c).null_count
                err(f"{p}: nulls in {c}")
        col_ids = tb.column("id").to_pylist()
        dup = len(col_ids) - len(set(col_ids)) + len(ids.intersection(col_ids))
        if dup:
            bad += dup
            err(f"{p}: duplicate ids {dup}")
        ids.update(col_ids)
        lic = set(tb.column("license").to_pylist())
        if not lic <= LICENSES:
            bad += 1
            err(f"{p}: licenses {lic}")
        short = sum(1 for t in tb.column("text").to_pylist() if len(t.split()) < 50 or "�" in t)
        if short:
            bad += short
            err(f"{p}: {short} docs short or with replacement char")
        n += tb.num_rows
    report["raw_text"][name] = {"files": len(files), "records": n, "invalid": bad,
                                "largest_file_mb": round(big / 1e6, 1)}
json.dump(report, open(os.path.join(WORK, "validation_report.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
