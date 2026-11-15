# SOURCES.md — Step 2 inspection results

Inspected: 2026-11-15. Place this file in `licenses/SOURCES.md`.

Method: each dataset was inspected remotely before any download to the project folder.
About 500 random records per dataset were read (random Parquet row groups, random shards
or random byte offsets). Language was identified with `langdetect`; "broken" means
mojibake or U+FFFD detected by `ftfy`; "noise" is a heuristic flag (spam keywords, high
upper-case ratio, low letter ratio, repeated lines, URL density). Sample sizes are small,
so percentages are estimates (roughly ±3 points).

Licenses below are what each dataset card states for the compilation. They do not settle
the rights in the underlying web pages. Have a lawyer confirm before commercial training.

## A. Raw text (language training)

| Dataset (Hugging Face ID) | License | Docs | Size | Somali | Broken | Dup in sample | Noise | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HPLT v2 — `HPLT/HPLT2.0_cleaned`, `som_Latn` | CC0-1.0 | 966,507 | 1.69 GB | 100% | 0% | 0% | 5.0% | **DOWNLOAD** — largest CC0 source. 20% of its URLs repeat inside the set itself, so dedup matters. |
| FineWeb-2 — `HuggingFaceFW/fineweb-2`, `som_Latn/train` | ODC-BY | 1,070,384 | 1.00 GB | 99.2% | 0% | 0% | 1.2% | **DOWNLOAD** — cleanest large web set; 72% of its URLs are not in HPLT. Skip the `som_Latn_removed` folder (1.79 GB of rejected text). |
| MADLAD-400 — `allenai/MADLAD-400`, `data-v1p5/so` clean shards | ODC-BY | 212,235 | 0.28 GB | 100% | 0.8% | 0% | 1.0% | **DOWNLOAD** — clean, long documents (106M words). 63% of URLs already in FineWeb-2, so about one third is new. Skip the `noisy` shards. |
| FinePDFs — `HuggingFaceFW/finepdfs`, `som_Latn/train` | ODC-BY | 21,781 | 0.10 GB | 99.8% | 0% | 0% | 6.2% | **DOWNLOAD** — government, NGO and school documents (median 555 words). No overlap with the web sets. Formal register. |
| GlotCC-V1 — `cis-lmu/GlotCC-V1`, `som-Latn` | CC0-1.0 | 43,188 | 0.10 GB | 100% | 0.6% | 3.8% | 8.6% | **SKIP** — noisiest of the web sets and mostly already covered (46% of URLs in FineWeb-2, 20% in HPLT). |
| SomaliWeb v1 — `khaledyusuf44/somaliweb-v1` | CC-BY-SA-4.0 | 819,322 | 0.51 GB | 100% | 0% | 0% | 1.2% | **YOUR DECISION** — best quality, but share-alike licence, and 28.5% of it comes from CC100 whose card lists the licence as "unknown". 71% of it is HPLT v2, which is available above under CC0. |
| Somali Wikipedia — `wikimedia/wikipedia`, `20231101.so` | CC-BY-SA-3.0 / GFDL | 9,021 | 8 MB | 93.4% | 0% | 0% | 12.5% | **YOUR DECISION** — same share-alike question. Small gain: 44% of articles are under 50 words. |
| FineTranslations — `HuggingFaceFW/finetranslations`, `som_Latn` | ODC-BY | 1,066,618 | 5.04 GB | 99.2% | 0% | 0% | 1.0% | **SKIP** — same Somali documents as FineWeb-2 plus machine-made English. |
| `maanka2/somali-web-corpus` | MIT (uploader's label) | n/a | 1.35 GB | 82.2% | 0% | 0% | 0% | **SKIP** — card says it was scraped from BBC Somali, Shabeelle and SONNA with no permission recorded; 89% of records are under 50 words; contains English. |
| `castorini/afriberta-corpus`, `somali` | Apache-2.0 (label) | n/a | 77 MB | not sampled | | | | **SKIP** — card says the text is mostly from the BBC website. |
| `csebuetnlp/xlsum`, Somali | CC-BY-NC-SA-4.0 | n/a | 6 MB | not sampled | | | | **SKIP** — non-commercial licence. |
| CC100 — `statmt/cc100` | "unknown" | n/a | n/a | not sampled | | | | **SKIP** — licence unclear. |
| OSCAR — `oscar-corpus/OSCAR-2301`, `community-oscar` | CC0-1.0 | n/a | n/a | not inspected | | | | **LISTED, NOT DOWNLOADED** — gated; needs a Hugging Face login and manual approval. Common Crawl based like the sets above. |
| `somast/somali-nlp-corpus` | "other" | n/a | 2.66 GB | not inspected | | | | **LISTED, NOT DOWNLOADED** — gated; has a separate non-commercial part. |

### Overlap

How much of each sample is already contained in SomaliWeb v1 (15-word window match
against all 819,322 SomaliWeb documents; SomaliWeb's own sample scored 100% as a check):

| Sample | Already in SomaliWeb v1 |
|---|---:|
| HPLT v2 | 89.4% |
| Wikipedia so | 53.6% |
| MADLAD-400 clean | 45.6% |
| GlotCC | 43.0% |
| FineWeb-2 | 38.7% (51.5% in a second sample of the same documents) |
| FinePDFs | 2.4% |

URL overlap between the web sets (exact URL match, full sets):

| | in HPLT v2 | in FineWeb-2 | in MADLAD |
|---|---:|---:|---:|
| HPLT v2 URLs | — | 37.7% | 8.9% |
| FineWeb-2 URLs | 28.3% | — | 13.0% |
| MADLAD URLs | 32.4% | 62.8% | — |
| GlotCC URLs | 20.4% | 46.4% | 5.5% |
| FinePDFs URLs | 0.1% | 0.0% | 0.0% |

Distinct URLs across HPLT v2, FineWeb-2, MADLAD, GlotCC and FinePDFs: 1,603,004
(before cleaning; the same text under two URLs is not counted as overlap here, so Step 3
dedup will remove more).

## B. Parallel pairs (translation set)

| Dataset | License | Pairs | Size | Somali side | Dup in sample | Decision |
|---|---|---:|---:|---:|---:|---|
| OPUS Infopankki — `Helsinki-NLP/opus_infopankki`, `en-so` | CC-BY-4.0 | 47,220 | 2.3 MB | 85.2% overall; 99% for sentences of 8+ words | 19.4% | **DOWNLOAD** — human-translated public information from Finland. The low overall figure comes from one- and two-word menu fragments, which Step 3 removes. Expect roughly 20,000–28,000 usable pairs. |
| Google SMOL — `google/smol`, `smolsent/en_so` + `smoldoc/en_so` | CC-BY-4.0 | 862 sentences + 339 documents | 1.1 MB | not sampled (tiny) | | **DOWNLOAD** — professional human translation. |
| NLLB v1 mined bitext — OPUS `NLLB` `en-so` (`https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-so.txt.zip`) | ODC-BY | 10,229,073 | 0.66 GB zip (1.6 GB unpacked) | 98.8% | 0% | **DOWNLOAD, FILTER HARD** — automatically aligned. Of 14 pairs read by hand, about half were mismatched, and some Somali is machine-translated web text. Only LaBSE-filtered pairs may enter the final set. |
| CCMatrix — `sentence-transformers/parallel-sentences-ccmatrix`, `en-so` | none stated on card | 222,793 | 30 MB | 100% | 7.6% | **LISTED, NOT DOWNLOADED** — licence not stated; NLLB v1 on OPUS already includes CCMatrix data for pairs it did not update. |

Note: `allenai/nllb` on Hugging Face is a loader script only and cannot be streamed; the
same data is taken from OPUS instead. Human-translated pairs available under a clear
licence total about 48,000 before cleaning, so reaching 50,000 good pairs depends on how
many NLLB pairs survive the LaBSE filter.

## C. Test sets (go to `test_quarantine/`, never into training)

| Dataset | License | Size | Decision |
|---|---|---:|---|
| Belebele — `facebook/belebele`, `som_Latn` | CC-BY-SA-4.0 | 900 questions, 0.9 MB | **DOWNLOAD to quarantine** |
| FLORES+ — `openlanguagedata/flores_plus`, `som_Latn` (dev + devtest) | CC-BY-SA-4.0 | 0.9 MB | **DOWNLOAD to quarantine** — gated: needs a Hugging Face login and accepting the terms. |

## D. Found but not part of this step (question–answer data, for the next step)

| Dataset | License | Content | Note |
|---|---|---|---|
| `CohereLabs/aya_dataset`, Somali rows | Apache-2.0 | 7,704 human-written prompt/answer pairs (3,734 original, 3,970 re-annotated) | Strong candidate for the instruction set. |
| `CohereLabs/aya_collection_language_split`, `somali` | Apache-2.0 | 3,582,111 rows, 1.13 GB | Machine-translated English tasks; translation errors visible. Needs heavy filtering. |
| `yacdev/somali-100k-native-conversations` | Apache-2.0 | 10,000 synthetic dialogues, 5.4 MB | Template-like openers; needs editor review. |

## Download total if approved

About 3.7 GB to download (5 GB with NLLB unpacked). SomaliWeb v1 and Wikipedia would
add 0.52 GB if approved.

## Outcome for v1.0 (2026-11-15)

Downloaded and used: HPLT v2, FineWeb-2, MADLAD-400 clean v1.5, FinePDFs, NLLB v1 (OPUS),
OPUS Infopankki, Google SMOL. Belebele went to `test_quarantine/`.

Not used: SomaliWeb v1 and Somali Wikipedia (share-alike licence question still open), FLORES+
(gated; needs a Hugging Face login), and every dataset marked SKIP or LISTED above.

Correction to the estimate above: OPUS Infopankki yielded far fewer usable pairs than expected,
because most of its sentences repeat across page versions. Actual counts are in
`final/v1.0/DATACARD.md`.

## E. Question–answer and instruction candidates (inspected 2026-11-15)

Rule applied: human-written Somali, licence that allows commercial use.

| Dataset | Licence | Content | Decision |
|---|---|---|---|
| `CohereLabs/aya_dataset`, Somali rows | Apache-2.0 | 7,704 prompt/answer rows written by 77 volunteer annotators (3,734 original, 3,970 re-annotated); 3,695 unique questions | **USED** — the only human-written source with a clear commercial licence. After cleaning: 1,115 training-ready pairs, 1,036 held for expert review (religion, health). |
| `CohereLabs/aya_collection_language_split`, `somali` | Apache-2.0 | 3,582,111 machine-translated task rows | **NOT USED** — machine translation; errors visible in samples. |
| `akoksal/muri-it-language-split`, `som` | Apache-2.0 | Instructions produced by a model and machine-translated, attached to web and Wikipedia text | **NOT USED** — instructions are not human-written; underlying text has mixed licences. |
| `Zyroxx66/Somali-Reasoning-Dataset`, `Zyroxx66/Somali-Somlish-Instruct-2K-Dataset` | Apache-2.0 | Synthetic and translated instruction data, partly mixed Somali–English | **NOT USED** — synthetic, no human review recorded. |
| `yacdev/somali-100k-native-conversations` and related | Apache-2.0 | Synthetic multi-turn dialogues | **NOT USED** — template-like wording; needs editor review first. |
| `saillab/alpaca-somali-cleaned`, `burtugeey/*alpaca*` | none stated | Machine translation of Alpaca | **NOT USED** — no licence stated; the original Alpaca data is non-commercial. |
| `alexandrainst/multi-wiki-qa` | CC-BY-NC-SA-4.0 | Model-generated questions on Wikipedia | **NOT USED** — non-commercial. |
| `Atnafu/Afri-MCQA` | CC-BY-NC-4.0 | Cultural multiple-choice benchmark | **NOT USED** — non-commercial; it is an evaluation set. |
| `unkadlabs/qor-af-soomaali` | CC-BY-SA-4.0 | Community-written, validated Somali sentences | **NOT USED** — share-alike; same open question as SomaliWeb v1. Worth revisiting after the legal decision. |
| `bltlab/lr-sum`, `som` | CC-BY-4.0 | News summarisation, test split only | **NOT USED** — evaluation data, and not question–answer. |
| `engabdalla/somali-qa`, `planwise-data/somali-stem-dataset` | CC-BY-4.0 | Empty repository; 100-row terminology sample | **NOT USED** — no usable content. |
