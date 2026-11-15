"""Parallel pairs: apply the LaBSE thresholds, keep all passing pairs in processed/,
then build the translation instruction set in final/v1.0/.

Usage: python3 -I 32_parallel_build.py <work_dir> <repo_dir>

Calibration (see DATACARD): LaBSE scores correct professional Somali translations lower than
it does for high-resource languages (SMOL sentence median 0.74), so one threshold does not fit
both kinds of source.
  - human-translated sources (SMOL, Infopankki): keep >= 0.40 (removes real misalignments only)
  - mined source (NLLB): keep >= 0.75, and drop pairs whose Somali side copies lower-case
    English words from the English side (a sign of machine-translated Somali)

Directions: the Somali side of many NLLB pairs is machine-translated web text. It is therefore
never used as an answer. English->Somali examples come only from human-translated pairs;
NLLB pairs are used Somali->English only. Human pairs are used in both directions.
"""
import json
import os
import random
import re
import sys

import numpy as np
import xxhash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ops import op6_pii  # noqa: E402

WORK, REPO = sys.argv[1], sys.argv[2]
T_HUMAN, T_MINED = 0.40, 0.75
TARGET = 50_000
FINAL = os.path.join(REPO, "final", "v1.0", "translation")
PROC = os.path.join(REPO, "processed")
os.makedirs(FINAL, exist_ok=True)
os.makedirs(PROC, exist_ok=True)

# Somali user instructions. {t} is the text to translate.
TO_EN = ["Turjum af Ingiriisi: {t}",
         "Qoraalkan u turjum af Ingiriisi:\n{t}",
         "Fadlan jumladan af Ingiriisi u beddel: {t}",
         "Sidee af Ingiriisi loogu yiraahdaa: \"{t}\"?",
         "Af Ingiriisi ku qor qoraalka soo socda:\n{t}",
         "Ii turjum qoraalkan, kuna qor af Ingiriisi:\n{t}",
         "Waa maxay turjumaadda af Ingiriisi ee jumladan? {t}",
         "{t}\n\nKan af Ingiriisi iigu turjum."]
TO_SO = [s.replace("af Ingiriisi", "af Soomaali").replace("Af Ingiriisi", "Af Soomaali") for s in TO_EN]
assert not any("Ingiriisi" in s for s in TO_SO), "a Somali-target template still names English"
PARAGRAPH_OK = (1, 4, 5, 7)   # templates that also fit multi-sentence text

_WORD = re.compile(r"[A-Za-z]{4,}")
_MENU = re.compile(r"_|(Finnish|Swedish|English|Russian|Estonian|French|Somali|Arabic|Persian|Chinese|"
                   r"Turkish|Spanish|Kurdish|Thai|Vietnamese)\s*$")


def copied_english(so, en):
    e = {w.lower() for w in _WORD.findall(en)}
    return any(w[0].islower() and w.lower() in e for w in _WORD.findall(so))


rows = [json.loads(l) for l in open(os.path.join(WORK, "parallel_candidates.jsonl"), encoding="utf-8")]
sims = np.load(os.path.join(WORK, "parallel_labse.npy"))
assert len(rows) == len(sims)
stats = json.load(open(os.path.join(WORK, "parallel_stats.json")))
for k in list(stats):
    if k.startswith("_"):
        del stats[k]
for st in stats.values():
    for k in ("rm_low_labse", "rm_menu_artifact", "rm_copied_english", "passed_labse", "passed_all"):
        st.pop(k, None)
rnd = random.Random(0)
removed = {"rm_low_labse": [], "rm_menu_artifact": [], "rm_copied_english": []}
passed = []
for r, s in zip(rows, sims):
    r["labse"] = round(float(s), 4)
    st = stats[r["source"]]
    human = r["source"] != "nllb"
    reason = None
    if s < (T_HUMAN if human else T_MINED):
        reason = "rm_low_labse"
    elif human and (_MENU.search(r["en"]) or "_" in r["so"]):
        reason = "rm_menu_artifact"
    elif not human and copied_english(r["so"], r["en"]):
        reason = "rm_copied_english"
    if reason:
        st[reason] = st.get(reason, 0) + 1
        removed[reason].append({"source": r["source"], "labse": r["labse"], "so": r["so"][:300], "en": r["en"][:300]})
    else:
        st["passed_all"] = st.get("passed_all", 0) + 1
        passed.append(r)
for k, v in removed.items():
    rnd.shuffle(v)
    with open(os.path.join(WORK, f"removed_parallel_{k}.jsonl"), "w", encoding="utf-8") as f:
        for item in v[:100]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# every pair that passed cleaning stays available in processed/
with open(os.path.join(PROC, "parallel_pairs_clean.jsonl"), "w", encoding="utf-8") as f:
    for r in passed:
        f.write(json.dumps({k: r[k] for k in ("so", "en", "source", "license", "level", "labse")},
                           ensure_ascii=False) + "\n")

human = [r for r in passed if r["source"] != "nllb"]
mined = sorted((r for r in passed if r["source"] == "nllb"), key=lambda r: -r["labse"])
mined = mined[:max(0, TARGET - 2 * len(human))]

examples = []   # (pair_key, to_en, row)
for j, r in enumerate(human):
    examples.append((f"h{j}", True, r))
    examples.append((f"h{j}", False, r))
for j, r in enumerate(mined):
    examples.append((f"m{j}", True, r))
rnd.shuffle(examples)

sel = {"total": len(examples), "pairs_used": len(human) + len(mined), "human_pairs": len(human),
       "mined_pairs": len(mined), "by_source": {}, "by_direction": {"so->en": 0, "en->so": 0},
       "by_source_direction": {}, "by_license": {}, "by_split": {}, "phones_masked": 0,
       "labse_min_mined": round(min(r["labse"] for r in mined), 4) if mined else None,
       "labse_median_mined": round(float(np.median([r["labse"] for r in mined])), 4) if mined else None,
       "labse_median_human": round(float(np.median([r["labse"] for r in human])), 4)}
files = {n: open(os.path.join(FINAL, f"{n}.jsonl"), "w", encoding="utf-8") for n in ("train", "validation", "test")}
for i, (key, to_en, r) in enumerate(examples, 1):
    so, _, p1 = op6_pii.mask_pii(r["so"])
    en, _, p2 = op6_pii.mask_pii(r["en"])
    sel["phones_masked"] += p1 + p2
    templates = TO_EN if to_en else TO_SO
    k = rnd.choice(PARAGRAPH_OK) if r["level"] == "paragraph" else rnd.randrange(len(templates))
    rec = {"id": f"so-tr-{i:07d}",
           "messages": [{"role": "user", "content": templates[k].format(t=so if to_en else en)},
                        {"role": "assistant", "content": en if to_en else so}],
           "source": r["source"], "category": "translation", "language": "so-en", "license": r["license"]}
    h = xxhash.xxh64_intdigest(key.encode()) % 100     # both directions of a pair share a split
    split = "validation" if h < 2 else "test" if h < 4 else "train"
    files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
    d = "so->en" if to_en else "en->so"
    sel["by_split"][split] = sel["by_split"].get(split, 0) + 1
    sel["by_source"][r["source"]] = sel["by_source"].get(r["source"], 0) + 1
    sel["by_license"][r["license"]] = sel["by_license"].get(r["license"], 0) + 1
    sel["by_direction"][d] += 1
    sd = sel["by_source_direction"].setdefault(r["source"], {"so->en": 0, "en->so": 0})
    sd[d] += 1
for f in files.values():
    f.close()
stats["_thresholds"] = {"human": T_HUMAN, "mined": T_MINED}
stats["_passed_total"] = len(passed)
stats["_selected"] = sel
json.dump(stats, open(os.path.join(WORK, "parallel_stats.json"), "w"), indent=1)
print("BUILD DONE", json.dumps(stats), flush=True)
