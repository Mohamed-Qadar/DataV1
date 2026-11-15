"""Step 5 — write final/v1.0/DATACARD.md from the pipeline statistics.

Usage: python3 -I 50_datacard.py <work_dir> <repo_dir>
"""
import glob
import json
import os
import sys

WORK, REPO = sys.argv[1], sys.argv[2]
s1 = json.load(open(os.path.join(WORK, "stage1_stats.json")))
s2 = json.load(open(os.path.join(WORK, "stage2_stats.json")))
par = json.load(open(os.path.join(WORK, "parallel_stats.json")))
val = json.load(open(os.path.join(WORK, "validation_report.json")))
tok = json.load(open(os.path.join(WORK, "token_ratio.json"))) if os.path.exists(os.path.join(WORK, "token_ratio.json")) else None
NAMES = {"fineweb2": "FineWeb-2 (som_Latn)", "hplt2": "HPLT v2 cleaned (som_Latn)",
         "madlad400": "MADLAD-400 v1.5 clean (so)", "finepdfs": "FinePDFs (som_Latn)",
         "infopankki": "OPUS Infopankki (en-so)", "smol": "Google SMOL (en-so)", "nllb": "NLLB v1 mined (en-so)"}
ORDER = ["fineweb2", "hplt2", "madlad400", "finepdfs"]


def n(x):
    return f"{x:,}"


def tot(key):
    return sum(s1[s].get(key, 0) for s in ORDER)


def du(path):
    return sum(os.path.getsize(p) for p in glob.glob(path, recursive=True) if os.path.isfile(p))


L = []
A = L.append
A("# DATACARD — Somali AI dataset v1.0\n")
A("Built 2026-11-15. Training-ready Somali data in two sets: raw text for language training and "
  "translation instructions for instruction training. Every record carries its source and licence.\n")
A("## 1. What is in this version\n")
A("| Set | Format | Records | Words | Files | Size |")
A("|---|---|---:|---:|---:|---:|")
rw = sum(v["words"] for v in s2["by_split"].values())
rt_size = du(os.path.join(REPO, "final/v1.0/raw_text/*.parquet"))
tr_size = du(os.path.join(REPO, "final/v1.0/translation/*.jsonl"))
A(f"| `raw_text/` | Parquet (`id`, `text`, `source`, `license`) | {n(s2['out'])} | {n(rw)} | "
  f"{sum(v['files'] for v in s2['by_split'].values())} | {rt_size/1e9:.2f} GB |")
sel = par["_selected"]
A(f"| `translation/` | JSONL chat `messages` | {n(sel['total'])} | — | 3 | {tr_size/1e6:.1f} MB |\n")
if tok:
    A(f"Token estimate for `raw_text/`: about {n(int(rw * tok['tokens_per_word']))} tokens with the "
      f"`{tok['tokenizer']}` tokenizer ({tok['tokens_per_word']:.2f} tokens per word, measured on "
      f"{n(tok['docs'])} random documents).\n")
A("### Splits (96% / 2% / 2%, assigned by a hash of the record id)\n")
A("| Split | Raw text docs | Raw text words | Raw text files | Translation records |")
A("|---|---:|---:|---:|---:|")
for sp in ("train", "validation", "test"):
    b = s2["by_split"][sp]
    A(f"| {sp} | {n(b['docs'])} | {n(b['words'])} | {b['files']} | {n(sel['by_split'].get(sp, 0))} |")
A("")
A("## 2. Raw text: what each cleaning step removed\n")
A("| Step | " + " | ".join(NAMES[s] for s in ORDER) + " | Total |")
A("|---|" + "---:|" * (len(ORDER) + 1))
rows = [("Records read", "in"), ("Removed: empty", "rm_empty"),
        ("3.1 Removed: unrepairable encoding", "rm_encoding"),
        ("3.1 Changed by normalisation / ftfy (kept)", "chg_normalize"),
        ("3.2 Apostrophes standardised (kept)", "chg_apostrophe"),
        ("3.3 Removed: not Somali", "rm_language"),
        ("3.4 Removed: under 50 words", "rm_short"),
        ("3.4 Removed: noise / spam", "rm_noise"),
        ("Left after steps 3.1–3.4", "out")]
for label, key in rows:
    A(f"| {label} | " + " | ".join(n(s1[s].get(key, 0)) for s in ORDER) + f" | {n(tot(key))} |")
A("| 3.5 Removed: exact duplicates | " + " | ".join(n(s2["exact_dup_by_source"][s]) for s in ORDER)
  + f" | {n(s2['rm_exact_dup'])} |")
A("| 3.5 Removed: near duplicates (MinHash) | " + " | ".join(n(s2["near_dup_by_source"][s]) for s in ORDER)
  + f" | {n(s2['rm_near_dup'])} |")
A(f"| Removed: contains test-set text | | | | | {n(s2['rm_test_leak'])} |")
A(f"| Removed: under 50 words after PII masking | | | | | {n(s2['rm_short_after_mask'])} |")
A("| **Final documents** | " + " | ".join(n(s2["by_source"].get(s, {}).get("docs", 0)) for s in ORDER)
  + f" | **{n(s2['out'])}** |")
A("| Final words | " + " | ".join(n(s2["by_source"].get(s, {}).get("words", 0)) for s in ORDER)
  + f" | {n(rw)} |\n")
A(f"Step 3.6 masked {n(s2['emails_masked'])} e-mail addresses and {n(s2['phones_masked'])} phone numbers "
  f"in {n(s2['pii_docs'])} documents (`[EMAIL]`, `[PHONE]`).\n")
A("Exact duplicates keep the first copy in the order FineWeb-2, FinePDFs, MADLAD-400, HPLT v2, so a page "
  "present in several sources is counted under the earlier one. Near duplicates keep the longest copy.\n")
A("### Licence distribution (raw text)\n")
A("| Licence | Documents | Words |")
A("|---|---:|---:|")
for lic, v in sorted(s2["by_license"].items()):
    A(f"| {lic} | {n(v['docs'])} | {n(v['words'])} |")
A("")
A("## 3. Translation set: what each step removed\n")
keys = ["rm_empty", "rm_encoding", "rm_misaligned_doc", "rm_short", "rm_long", "rm_length_ratio", "rm_identical",
        "rm_url_email", "rm_number_mismatch", "rm_noise", "rm_spam", "rm_language_so", "rm_language_en",
        "rm_duplicate", "rm_test_leak"]
labels = {"rm_empty": "empty", "rm_encoding": "unrepairable encoding", "rm_misaligned_doc": "misaligned document",
          "rm_short": "under 4 words", "rm_long": "over 80 words", "rm_length_ratio": "length ratio",
          "rm_identical": "both sides identical", "rm_url_email": "URL or e-mail", "rm_number_mismatch": "numbers differ",
          "rm_noise": "noise", "rm_spam": "spam", "rm_language_so": "Somali side not Somali",
          "rm_language_en": "English side not English", "rm_duplicate": "duplicate", "rm_test_leak": "test-set text"}
P = ["smol", "infopankki", "nllb"]
A("| Step | " + " | ".join(NAMES[s] for s in P) + " |")
A("|---|---:|---:|---:|")
A(f"| Pairs in source | {n(par['smol']['in'])} | {n(par['infopankki']['in'])} | {n(par['nllb']['in_total'])} |")
A(f"| Below top 10% mining score (not processed) | — | — | {n(par['nllb']['rm_below_top10pct_score'])} |")
for k in keys:
    if any(par[s].get(k) for s in P):
        A(f"| Removed: {labels[k]} | " + " | ".join(n(par[s].get(k, 0)) for s in P) + " |")
A(f"| Passed cheap filters | {n(par['smol']['out'])} | {n(par['infopankki']['out'])} | {n(par['nllb']['passed_cheap_filters'])} |")
A(f"| Scored with LaBSE | {n(par['smol']['out'])} | {n(par['infopankki']['out'])} | {n(par['nllb']['sent_to_labse'])} (random sample) |")
th = par["_thresholds"]
A(f"| 3.7 Removed: LaBSE similarity too low (human < {th['human']}, mined < {th['mined']}) | " + " | ".join(n(par[s].get('rm_low_labse', 0)) for s in P) + " |")
A("| Removed: menu / link-label fragments | " + " | ".join(n(par[s].get('rm_menu_artifact', 0)) for s in P) + " |")
A("| Removed: Somali side copies English words (machine-translation sign) | — | — | " + n(par['nllb'].get('rm_copied_english', 0)) + " |")
A("| Passed all cleaning (kept in `processed/parallel_pairs_clean.jsonl`) | " + " | ".join(n(par[s].get('passed_all', 0)) for s in P) + " |")
A("| **Pairs used in final set** | " + " | ".join(f"**{n(sel['by_source_direction'].get(s, {}).get('so->en', 0))}**" for s in P) + " |")
A("| Examples Somali→English | " + " | ".join(n(sel['by_source_direction'].get(s, {}).get('so->en', 0)) for s in P) + " |")
A("| Examples English→Somali | " + " | ".join(n(sel['by_source_direction'].get(s, {}).get('en->so', 0)) for s in P) + " |\n")
A(f"Final set: {n(sel['total'])} examples built from {n(sel['pairs_used'])} pairs "
  f"({n(sel['by_direction']['so->en'])} Somali→English, {n(sel['by_direction']['en->so'])} English→Somali). "
  "Licences: " + ", ".join(f"{k} {n(v)}" for k, v in sorted(sel["by_license"].items())) + ".\n")
A("Two decisions that differ from the original plan, both made to protect answer quality:\n")
A("1. **LaBSE threshold calibrated per source.** LaBSE scores correct professional Somali translations "
  f"low (SMOL sentences: median 0.74, a quarter under 0.67), so a single 0.75 cut-off would have discarded "
  f"about a third of the human translations. Human-translated sources keep pairs at {th['human']} or above, "
  f"which removes real misalignments only; mined NLLB pairs must reach {th['mined']} "
  f"(lowest NLLB score in the final set: {sel['labse_min_mined']}).")
A("2. **Not half/half by direction.** Many NLLB Somali sentences are machine-translated web text. Using "
  "them as answers would teach the model unnatural Somali, so NLLB pairs are used Somali→English only. "
  "English→Somali examples come only from human translations, and each human pair is used in both "
  "directions (both directions of a pair always fall in the same split).\n")
A(f"NLLB reserve: {n(par['nllb'].get('not_scored_reserve', 0))} pairs passed the cheap filters but were not "
  "scored with LaBSE in this build (CPU-only run, about 90 minutes for 102,063 pairs). At the observed pass "
  "rate they would yield well over 100,000 more Somali→English pairs.\n")
A("## 4. Validation\n")
A("| Set | Split | Records | Invalid | Largest file |")
A("|---|---|---:|---:|---:|")
for sp in ("train", "validation", "test"):
    v = val["raw_text"][sp]
    A(f"| raw_text | {sp} | {n(v['records'])} | {v['invalid']} | {v['largest_file_mb']} MB |")
for sp in ("train", "validation", "test"):
    v = val["translation"][sp]
    A(f"| translation | {sp} | {n(v['records'])} | {v['invalid']} | — |")
A("")
A("Checks: UTF-8 without BOM, required fields, unique ids, user/assistant order, no empty content, "
  "no chat-template tokens, allowed licences, no document under 50 words.\n")
A("## 5. Known limitations\n")
A("1. Not included pending a licence decision: SomaliWeb v1 (CC-BY-SA, partly CC100) and Somali Wikipedia (CC-BY-SA).")
A("2. Dataset licences cover the compilations, not the rights in the original web pages. Get legal review before commercial training.")
A("3. PII masking is regex only (e-mails, phone numbers). Names and addresses are not masked.")
A("4. No toxicity filter was applied. Web text can contain offensive or clan-related hate content.")
A("5. Machine-translated Somali web pages are still present in the raw text; only spam and structural noise were filtered.")
A("5b. Af-Maay text is removed by the language filter (it is identified as a different language). This version is Standard Somali only.")
A("6. NLLB pairs are automatically mined. LaBSE removes most mismatches but not all, and their Somali side is often machine-translated; they appear only as Somali→English examples. Review a sample before relying on them.")
A("7. The Somali instruction templates in the translation set were written by an AI model and should be checked by a native-speaker editor.")
A("8. The test-leakage guard used Belebele (FLORES-200 passages). FLORES+ itself is gated and was not downloaded; repeat the check once it is in `test_quarantine/`.")
A("9. This version has no question–answer or chat data. That is the next step.")
A("10. Quality was measured on samples and by automatic filters; no human editor has reviewed this version.\n")
open(os.path.join(REPO, "final", "v1.0", "DATACARD.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
