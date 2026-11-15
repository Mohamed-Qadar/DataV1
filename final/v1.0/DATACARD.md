# DATACARD — Somali AI dataset v1.0

Built 2026-11-15. Training-ready Somali data in three sets: raw text for language training, and translation instructions and question-answer pairs for instruction training. Every record carries its source and licence. Where each source comes from is declared in the repository README.

## 1. What is in this version

| Set | Format | Records | Words | Files | Size |
|---|---|---:|---:|---:|---:|
| `raw_text/` | Parquet (`id`, `text`, `source`, `license`) | 1,505,050 | 520,429,948 | 30 | 1.09 GB |
| `translation/` | JSONL chat `messages` | 50,000 | — | 3 | 23.9 MB |
| `qa/` | JSONL chat `messages` | 1,115 training-ready (+ 1,036 held for review) | — | 4 | 1.0 MB |

Token estimate for `raw_text/`: about 1,363,459,117 tokens with the `Qwen/Qwen3-8B` tokenizer (2.62 tokens per word, measured on 2,000 random documents).

### Splits (96% / 2% / 2%, assigned by a hash of the record id)

| Split | Raw text docs | Raw text words | Raw text files | Translation records | QA records |
|---|---:|---:|---:|---:|---:|
| train | 1,444,653 | 499,553,778 | 28 | 47,979 | 1,067 |
| validation | 30,121 | 10,542,693 | 1 | 1,014 | 22 |
| test | 30,276 | 10,333,477 | 1 | 1,007 | 26 |

## 2. Raw text: what each cleaning step removed

| Step | FineWeb-2 (som_Latn) | HPLT v2 cleaned (som_Latn) | MADLAD-400 v1.5 clean (so) | FinePDFs (som_Latn) | Total |
|---|---:|---:|---:|---:|---:|
| Records read | 1,070,384 | 966,507 | 212,235 | 21,781 | 2,270,907 |
| Removed: empty | 0 | 0 | 0 | 0 | 0 |
| 3.1 Removed: unrepairable encoding | 134 | 213 | 392 | 0 | 739 |
| 3.1 Changed by normalisation / ftfy (kept) | 19,935 | 11,633 | 15,083 | 12,210 | 58,861 |
| 3.2 Apostrophes standardised (kept) | 451,167 | 461,413 | 105,288 | 2,888 | 1,020,756 |
| 3.3 Removed: not Somali | 15,607 | 6,317 | 206 | 184 | 22,314 |
| 3.4 Removed: under 50 words | 1,486 | 1 | 15 | 350 | 1,852 |
| 3.4 Removed: noise / spam | 5,786 | 25,387 | 1,148 | 649 | 32,970 |
| Left after steps 3.1–3.4 | 1,047,371 | 934,589 | 210,474 | 20,598 | 2,213,032 |
| 3.5 Removed: exact duplicates | 5 | 375,423 | 509 | 1 | 375,938 |
| 3.5 Removed: near duplicates (MinHash) | 175,002 | 136,441 | 20,367 | 229 | 332,039 |
| Removed: contains test-set text | | | | | 0 |
| Removed: under 50 words after PII masking | | | | | 5 |
| **Final documents** | 872,360 | 422,725 | 189,598 | 20,367 | **1,505,050** |
| Final words | 247,064,709 | 159,209,728 | 90,729,437 | 23,426,074 | 520,429,948 |

Step 3.6 masked 172,158 e-mail addresses and 29,795 phone numbers in 138,572 documents (`[EMAIL]`, `[PHONE]`).

Exact duplicates keep the first copy in the order FineWeb-2, FinePDFs, MADLAD-400, HPLT v2, so a page present in several sources is counted under the earlier one. Near duplicates keep the longest copy.

### Licence distribution (raw text)

| Licence | Documents | Words |
|---|---:|---:|
| cc0-1.0 | 422,725 | 159,209,728 |
| odc-by-1.0 | 1,082,325 | 361,220,220 |

## 3. Translation set: what each step removed

| Step | Google SMOL (en-so) | OPUS Infopankki (en-so) | NLLB v1 mined (en-so) |
|---|---:|---:|---:|
| Pairs in source | 1,201 | 47,220 | 10,229,073 |
| Below top 10% mining score (not processed) | — | — | 9,206,158 |
| Removed: unrepairable encoding | 0 | 0 | 374 |
| Removed: under 4 words | 2 | 8,607 | 49,274 |
| Removed: over 80 words | 0 | 0 | 51 |
| Removed: length ratio | 0 | 181 | 45,575 |
| Removed: both sides identical | 0 | 248 | 12 |
| Removed: URL or e-mail | 0 | 11 | 2,014 |
| Removed: numbers differ | 2 | 647 | 54,757 |
| Removed: noise | 0 | 355 | 8,411 |
| Removed: spam | 1 | 9 | 3,645 |
| Removed: Somali side not Somali | 1 | 206 | 48,003 |
| Removed: English side not English | 1 | 1,637 | 2,111 |
| Removed: duplicate | 0 | 24,450 | 326,480 |
| Removed: test-set text | 0 | 0 | 2 |
| Passed cheap filters | 1,194 | 10,869 | 482,206 |
| Scored with LaBSE | 1,194 | 10,869 | 90,000 (random sample) |
| 3.7 Removed: LaBSE similarity too low (human < 0.4, mined < 0.75) | 14 | 126 | 53,694 |
| Removed: menu / link-label fragments | 1 | 991 | 0 |
| Removed: Somali side copies English words (machine-translation sign) | — | — | 4,544 |
| Passed all cleaning (kept in `processed/parallel_pairs_clean.jsonl`) | 1,179 | 9,752 | 31,762 |
| **Pairs used in final set** | **1,179** | **9,752** | **28,138** |
| Examples Somali→English | 1,179 | 9,752 | 28,138 |
| Examples English→Somali | 1,179 | 9,752 | 0 |

Final set: 50,000 examples built from 39,069 pairs (39,069 Somali→English, 10,931 English→Somali). Licences: cc-by-4.0 21,862, odc-by-1.0 28,138.

Two decisions that differ from the original plan, both made to protect answer quality:

1. **LaBSE threshold calibrated per source.** LaBSE scores correct professional Somali translations low (SMOL sentences: median 0.74, a quarter under 0.67), so a single 0.75 cut-off would have discarded about a third of the human translations. Human-translated sources keep pairs at 0.4 or above, which removes real misalignments only; mined NLLB pairs must reach 0.75 (lowest NLLB score in the final set: 0.7819).
2. **Not half/half by direction.** Many NLLB Somali sentences are machine-translated web text. Using them as answers would teach the model unnatural Somali, so NLLB pairs are used Somali→English only. English→Somali examples come only from human translations, and each human pair is used in both directions (both directions of a pair always fall in the same split).

NLLB reserve: 392,206 pairs passed the cheap filters but were not scored with LaBSE in this build (CPU-only run, about 90 minutes for 102,063 pairs). At the observed pass rate they would yield well over 100,000 more Somali→English pairs.

## 4. Question-answer set

Source: the Somali rows of the Aya Dataset (Cohere Labs community, Apache-2.0), written by 77 volunteer annotators. It is the only human-written Somali question-answer source found with a licence that allows commercial use. Machine-translated and synthetic sets were inspected and left out (see `licenses/SOURCES.md`).

| Step | Rows |
|---|---:|
| Somali rows in Aya (original + re-annotated) | 7,704 |
| Removed: empty | 5 |
| Removed: prompt is not a question (titles, essays, poems, song lyrics) | 1,276 |
| Removed: repeated question (other versions of the same question) | 2,728 |
| Unique questions | 3,695 |
| Removed: versions of the answer disagree | 204 |
| Removed: answer over 200 words (pasted articles) | 515 |
| Removed: URL in text | 6 |
| Removed: Every Word Capitalised style | 465 |
| Removed: all capitals | 14 |
| Removed: question not Somali | 27 |
| Removed: answer not Somali | 70 |
| Removed: poor spelling (over 15% of words unknown to the clean corpus) | 243 |
| Held for expert review: religion (`needs_review.jsonl`) | 883 |
| Held for expert review: health (`needs_review.jsonl`) | 153 |
| **Training-ready (`train` / `validation` / `test`)** | **1,115** |

Median answer length: 9 words. The spelling check compares each pair with a vocabulary of 312,710 words seen at least five times in about 100 million words of the cleaned raw text.

Religion and health questions are detected by keyword and kept out of the training files. Those answers need a scholar or a clinician before a model learns from them. The keyword match is not perfect: some such questions remain in the training files, and some general questions are in the review file.

## 5. Validation

| Set | Split | Records | Invalid | Largest file |
|---|---|---:|---:|---:|
| raw_text | train | 1,444,653 | 0 | 39.1 MB |
| raw_text | validation | 30,121 | 0 | 22.3 MB |
| raw_text | test | 30,276 | 0 | 21.9 MB |
| translation | train | 47,979 | 0 | — |
| translation | validation | 1,014 | 0 | — |
| translation | test | 1,007 | 0 | — |
| qa | train | 1,067 | 0 | — |
| qa | validation | 22 | 0 | — |
| qa | test | 26 | 0 | — |
| qa | needs_review | 1,036 | 0 | — |

Checks: UTF-8 without BOM, required fields, unique ids, user/assistant order, no empty content, no chat-template tokens, allowed licences, no document under 50 words.

## 6. Known limitations

1. Not included pending a licence decision: SomaliWeb v1 (CC-BY-SA, partly CC100) and Somali Wikipedia (CC-BY-SA).
2. Dataset licences cover the compilations, not the rights in the original web pages. Get legal review before commercial training.
3. PII masking is regex only (e-mails, phone numbers). Names and addresses are not masked.
4. No toxicity filter was applied. Web text can contain offensive or clan-related hate content.
5. Machine-translated Somali web pages are still present in the raw text; only spam and structural noise were filtered.
5b. Af-Maay text is removed by the language filter (it is identified as a different language). This version is Standard Somali only.
6. NLLB pairs are automatically mined. LaBSE removes most mismatches but not all, and their Somali side is often machine-translated; they appear only as Somali→English examples. Review a sample before relying on them.
7. The Somali instruction templates in the translation set were written by an AI model and should be checked by a native-speaker editor.
8. The test-leakage guard used Belebele (FLORES-200 passages). FLORES+ itself is gated and was not downloaded; repeat the check once it is in `test_quarantine/`.
9. The question-answer set is small and unedited. Answers were written by volunteers and have not been fact-checked; spelling and phrasing are uneven even after filtering. An editor should read it before training.
9b. There is no multi-turn chat data yet, and the instruction data is far below the 100,000-example goal.
10. Quality was measured on samples and by automatic filters; no human editor has reviewed this version.
