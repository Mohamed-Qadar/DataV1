"""Question-answer set from the human-written Somali rows of the Aya Dataset.

Usage: python3 -I 35_qa_build.py <raw_dir> <work_dir> <repo_dir> <glotlid_model>

Only question-style prompts with short, self-contained answers go into the training
files. Religious and health questions are NOT put into the training files: they go to
needs_review.jsonl, because those answers must be checked by a scholar or a clinician
first. Nothing here has been fact-checked.
"""
import collections
import json
import os
import pickle
import random
import re
import sys

import pyarrow.parquet as pq
import xxhash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ops import op1_normalize, op2_apostrophes, op3_language, op6_pii, op7_test_leakage  # noqa: E402

RAW, WORK, REPO, MODEL = sys.argv[1:5]
FINAL = os.path.join(REPO, "final", "v1.0", "qa")
os.makedirs(FINAL, exist_ok=True)
op3_language.load(MODEL)
op7_test_leakage.load_belebele(os.path.join(REPO, "test_quarantine", "belebele", "data", "som_Latn.jsonl"))
MAX_ANSWER_WORDS = 200
MAX_OOV = 0.15            # share of words not found in the clean raw-text vocabulary
VOCAB_SHARDS = ("train-00000", "train-00001", "train-00002", "train-00003", "train-00012", "train-00020")

PREFIX = re.compile(r"^\s*(su[',]?aal(ta)?|jawaab(ta)?|jawab)\s*[:.\-–]?\s+", re.I)
QUESTION = re.compile(r"\?|^(maxay|maxaa|waa maxay|sidee|side|yaa|goorma|imisa|immisa|xaggee|halkee|ma |muxuu|maxuu|"
                      r"kee|intee|waa kuma|sheeg|qeex|sharax|qor|tirakoob|soo koob|bixi|magacaw|faahfaahi|waxaad sheeg)", re.I)
RELIGION = re.compile(r"\b(nabi|nebi|rasuul|qur'?aan|quraan|suura[dt]|saxaab|ilaah|alle|allaah|salaad|xadiis|islaam|"
                      r"muslim|malaa'?ig|janna|xaj|soon|sako|zako|masaajid|masjid|aayad|csw|scw|nnkh|sheekh|kitaab|"
                      r"uxud|badar|makka|madiina|mushrik|iimaan|shareeco|xaaraan|xalaal|duco|tukad|rakc|sujuud|anbiy|"
                      r"nuux|muuse|ciise|ibraahim|yuusuf|towxiid|tawxiid|sunn|faral|waajib|cibaad|diin|bede?r|bin|binu|bintu|"
                      r"khaliif|samad|aakhir|qiyaam|shahiid|gaal|kufr|jihaad|cadaab|naar|shaydaan|jin|asxaab|"
                      r"hijr|malag|caqiid|fiqh|imaam|wadaad|kacbad?|ramadaan)\w*", re.I)
HEALTH = re.compile(r"\b(cudur|daawo|daawey|caafimaad|xanuun|dhakhtar|dhaqtar|takhtar|isbitaal|uur|ilmo ?soo ?rid|"
                      r"kaadi|dhiig|wadn|kelya|beer|sambab|kansar|sonkor|macaan|tallaal|fayras|bakteeriy|qandho|"
                      r"qalliin|dhalmo|galmo|ferji|nuujin|nafaqo|fitamiin|vitamin|hurdo ?la'?aan|niyad ?jab|walwal)\w*", re.I)
URL = re.compile(r"https?://|www\.", re.I)
KEY = re.compile(r"[^a-z0-9']+")
TOK = re.compile(r"[a-z']{4,}")
WORD = re.compile(r"[a-z']+")


def clean(text):
    t = op1_normalize.normalize(text or "")
    if not t:
        return ""
    t = op2_apostrophes.fix_apostrophes(t)
    t = PREFIX.sub("", t).strip()
    return re.sub(r"\s+\d$", "", t) if len(t.split()) > 3 else t   # stray trailing list number


def load_vocab():
    """Words seen at least 5 times in ~100M words of the cleaned raw text (spelling reference)."""
    cache = os.path.join(WORK, "somali_vocab.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    cnt = collections.Counter()
    for name in VOCAB_SHARDS:
        path = os.path.join(REPO, "final", "v1.0", "raw_text", name + ".parquet")
        for t in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            cnt.update(WORD.findall(t.lower()))
    vocab = {w for w, c in cnt.items() if c >= 5}
    pickle.dump(vocab, open(cache, "wb"))
    return vocab


def oov_rate(text, vocab):
    ws = [w for w in WORD.findall(text.lower()) if len(w) >= 3]
    return sum(w not in vocab for w in ws) / len(ws) if ws else 0.0


def title_case(text, min_words=5):
    w = [x for x in text.split() if x[0].isalpha()]
    return len(w) >= min_words and sum(x[0].isupper() for x in w) / len(w) >= 0.8


def _unused(text):
    w = [x for x in text.split() if x[0].isalpha()]
    return len(w) >= 5 and sum(x[0].isupper() for x in w) / len(w) >= 0.8


def tokens(s):
    return set(TOK.findall(s.lower()))


stats = collections.OrderedDict(in_somali_rows=0)
removed = collections.defaultdict(list)


def drop(reason, p, a):
    stats[reason] = stats.get(reason, 0) + 1
    removed[reason].append({"prompt": p[:300], "answer": a[:300]})


rows = [r for r in pq.read_table(os.path.join(RAW, "aya/data/train-00000-of-00001.parquet")).to_pylist()
        if r["language"] == "Somali"]
stats["in_somali_rows"] = len(rows)
stats["annotators"] = len({r["user_id"] for r in rows})
groups = collections.OrderedDict()
for r in rows:
    p, a = clean(r["inputs"]), clean(r["targets"])
    if not p or not a:
        drop("rm_empty", p, a); continue
    if len(p.split()) < 2 or not QUESTION.search(p):
        drop("rm_not_a_question", p, a); continue
    groups.setdefault(KEY.sub(" ", p.lower()).strip(), []).append((p, a, r["annotation_type"]))
stats["question_rows"] = sum(len(v) for v in groups.values())
stats["unique_questions"] = len(groups)

VOCAB = load_vocab()
stats["spelling_vocabulary_words"] = len(VOCAB)
keep, review = [], []
for key, items in groups.items():
    stats["rm_duplicate_question"] = stats.get("rm_duplicate_question", 0) + len(items) - 1
    answers = list({a.lower(): (p, a, t) for p, a, t in items}.values())
    if len(answers) > 1:
        base = tokens(answers[0][1])
        agree = all(len(base & tokens(x[1])) / max(1, len(base | tokens(x[1]))) >= 0.5 for x in answers[1:])
        if not agree:
            drop("rm_conflicting_answers", answers[0][0], " || ".join(x[1][:120] for x in answers)); continue
    # prefer the re-annotated (edited) version, then the longest answer
    p, a, _ = max(answers, key=lambda x: (x[2] == "re-annotations", len(x[1])))
    na = len(a.split())
    if na > MAX_ANSWER_WORDS:
        drop("rm_long_answer", p, a); continue
    if URL.search(p) or URL.search(a):
        drop("rm_url", p, a); continue
    if title_case(a) or (title_case(p) and title_case(a, 3)):
        drop("rm_title_case_answer", p, a); continue
    if p.isupper() or a.isupper():
        drop("rm_all_caps", p, a); continue
    lab, pr = op3_language.lang_of(p)
    if len(p.split()) >= 4 and (lab != "__label__som_Latn" or pr < 0.5):
        drop("rm_question_not_somali", p, a); continue
    lab, pr = op3_language.lang_of(a)
    if na >= 4 and (lab != "__label__som_Latn" or pr < 0.5):
        drop("rm_answer_not_somali", p, a); continue
    if oov_rate(p + " " + a, VOCAB) > MAX_OOV:
        drop("rm_poor_spelling", p, a); continue
    if op7_test_leakage.sentence_in_test(p) or op7_test_leakage.sentence_in_test(a):
        drop("rm_test_leak", p, a); continue
    p, m1, f1 = op6_pii.mask_pii(p)
    a, m2, f2 = op6_pii.mask_pii(a)
    stats["pii_masked"] = stats.get("pii_masked", 0) + m1 + m2 + f1 + f2
    both = p + " " + a
    if RELIGION.search(both):
        review.append((p, a, "religion"))
    elif HEALTH.search(both):
        review.append((p, a, "health"))
    else:
        keep.append((p, a))

rnd = random.Random(0)
rnd.shuffle(keep)
split_count = collections.Counter()
files = {n: open(os.path.join(FINAL, f"{n}.jsonl"), "w", encoding="utf-8") for n in ("train", "validation", "test")}
for i, (p, a) in enumerate(keep, 1):
    rec = {"id": f"so-qa-{i:07d}",
           "messages": [{"role": "user", "content": p}, {"role": "assistant", "content": a}],
           "source": "aya", "category": "qa_general", "language": "so", "license": "apache-2.0"}
    h = xxhash.xxh64_intdigest(rec["id"].encode()) % 100
    split = "validation" if h < 2 else "test" if h < 4 else "train"
    split_count[split] += 1
    files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
for f in files.values():
    f.close()
with open(os.path.join(FINAL, "needs_review.jsonl"), "w", encoding="utf-8") as f:
    for i, (p, a, why) in enumerate(review, 1):
        f.write(json.dumps({"id": f"so-qa-review-{i:07d}",
                            "messages": [{"role": "user", "content": p}, {"role": "assistant", "content": a}],
                            "source": "aya", "category": f"qa_{why}", "language": "so", "license": "apache-2.0",
                            "review_reason": why}, ensure_ascii=False) + "\n")
stats["final_training_ready"] = len(keep)
stats["by_split"] = dict(split_count)
stats["needs_review"] = dict(collections.Counter(w for _, _, w in review))
stats["answer_words_median"] = sorted(len(a.split()) for _, a in keep)[len(keep) // 2] if keep else 0
json.dump(stats, open(os.path.join(WORK, "qa_stats.json"), "w"), indent=1)
for k, v in removed.items():
    rnd.shuffle(v)
    with open(os.path.join(WORK, f"removed_qa_{k}.jsonl"), "w", encoding="utf-8") as f:
        for item in v[:100]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
print("QA DONE", json.dumps(stats), flush=True)
