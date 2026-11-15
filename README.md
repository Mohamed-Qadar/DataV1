# Somali AI — DataV1

Training-ready Somali data, version 1.0 (built 2026-11-15).

| Folder | Content |
|---|---|
| `final/v1.0/raw_text/` | Cleaned, deduplicated Somali text for language training. Parquet shards with `id`, `text`, `source`, `license`. Splits: `train-*`, `validation-*`, `test-*`. |
| `final/v1.0/translation/` | Somali↔English translation instructions in chat `messages` format. |
| `final/v1.0/qa/` | Human-written Somali question–answer pairs in chat `messages` format. `needs_review.jsonl` holds religion and health questions that must be checked by an expert before use. |
| `final/v1.0/DATACARD.md` | Counts for every cleaning step, licence distribution, known limitations. Read this first. |
| `processed/parallel_pairs_clean.jsonl` | Every parallel pair that passed cleaning, with its LaBSE score. |
| `test_quarantine/` | Test sets. Never use these for training. |
| `licenses/` | `SOURCES.md` (every dataset inspected and the decision) and `ATTRIBUTION.md` (required credits). |
| `logs/` | Up to 100 random examples of what each cleaning step removed, and the raw statistics. |
| `scripts/` | The pipeline that produced this version. |

## Data provenance declaration

All data in this repository was taken from the public datasets listed below, downloaded on
2026-11-15. This project scraped no website, generated no data with an AI model, and used no
private or unpublished source. The only text written for this project is the set of 16 short
Somali instruction templates ("Turjum af Ingiriisi: …") in the translation set, which an AI
model wrote.

The source datasets themselves are not free of machine-made text: web pages and NLLB sentence
pairs include machine-translated Somali. That is why NLLB is never used as a Somali answer.

Each record names its origin in the `source` field and its licence in the `license` field.

### Sources used

| `source` | Dataset and publisher | Where the text originally comes from | Licence | Downloaded from (revision) | In this version |
|---|---|---|---|---|---|
| `fineweb2` | FineWeb-2, Somali subset `som_Latn` — Hugging Face | Public web pages collected by Common Crawl, 2013–2024 | ODC-BY 1.0 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 (`af9c13333eb9`) | 872,360 documents in `raw_text/` |
| `hplt2` | HPLT v2 cleaned, `som_Latn` — HPLT project | Public web pages, mostly from the Internet Archive, some from Common Crawl | CC0 1.0 | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned (`d1324a5283f7`) | 422,725 documents in `raw_text/` |
| `madlad400` | MADLAD-400 v1.5, Somali clean split — Google, hosted by Allen AI | Public web pages collected by Common Crawl up to August 2022 | ODC-BY 1.0 | https://huggingface.co/datasets/allenai/MADLAD-400 (`9d886a76bd8f`) | 189,598 documents in `raw_text/` |
| `finepdfs` | FinePDFs, `som_Latn` — Hugging Face | Public PDF documents found by Common Crawl, 2013–2025, and re-fetched from the web (for example government, NGO and school documents) | ODC-BY 1.0 | https://huggingface.co/datasets/HuggingFaceFW/finepdfs (`220bac3acbf0`) | 20,367 documents in `raw_text/` |
| `nllb` | NLLB v1 mined bitext, English–Somali — Meta AI, distributed by OPUS | Sentence pairs matched automatically from Common Crawl and ParaCrawl web text | ODC-BY 1.0 | https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-so.txt.zip (SHA-256 `95c2f2f9e0dc4a42…`) | 28,138 examples in `translation/`, Somali→English only |
| `infopankki` | OPUS Infopankki v1, English–Somali — City of Helsinki, distributed by OPUS | Human translations of infopankki.fi, a public information site for immigrants in Finland | CC-BY 4.0 | https://huggingface.co/datasets/Helsinki-NLP/opus_infopankki (`52339513e7ee`) | 9,752 pairs, used in both directions (19,504 examples) in `translation/` |
| `smol` | SMOL (SmolSent, SmolDoc), English→Somali — Google | English texts translated into Somali by professional translators | CC-BY 4.0 | https://huggingface.co/datasets/google/smol (`fdaff3a1a019`) | 1,179 pairs, used in both directions (2,358 examples) in `translation/` |
| `aya` | Aya Dataset, Somali rows — Cohere Labs and the Aya open-science community | Questions and answers written by 77 volunteer Somali speakers on the Aya Annotation Platform | Apache 2.0 | https://huggingface.co/datasets/CohereLabs/aya_dataset (`f9ea04583f02`) | 1,115 pairs in `qa/` + 1,036 in `qa/needs_review.jsonl` |
| — (test only) | Belebele, `som_Latn` — Meta AI | Reading-comprehension questions on FLORES-200 passages | CC-BY-SA 4.0 | https://huggingface.co/datasets/facebook/belebele (`7899cdfa4e1e`) | 900 questions in `test_quarantine/` |

### What was done to the data

Text was not rewritten. The pipeline only removed records and made mechanical repairs:
encoding repair, Unicode normalisation, one apostrophe character, removal of "Su'aal:" / "Jawaab:"
labels, language filtering, removal of
short and noisy documents, duplicate removal, masking of e-mail addresses and phone numbers, and
removal of misaligned translation pairs. Exact counts per step are in `final/v1.0/DATACARD.md`.

### Inspected and deliberately not used

| Dataset | Reason |
|---|---|
| SomaliWeb v1 | Share-alike licence (CC-BY-SA), and 28.5% of it comes from CC100, whose licence is unclear. Awaiting a legal decision. |
| Somali Wikipedia | Share-alike licence (CC-BY-SA). Awaiting the same decision. |
| `maanka2/somali-web-corpus`, AfriBERTa corpus, XL-Sum | Text scraped from BBC Somali and other news sites, or non-commercial licence. |
| CCMatrix, CC100, OSCAR, `somast/somali-nlp-corpus` | Licence not stated, or access is gated. |
| Aya Collection (Somali), Alpaca-Somali sets, `Zyroxx66/Somali-Reasoning-Dataset`, MURI-IT | Machine-translated or synthetic instructions, not human-written. |
| `yacdev` Somali conversation sets | Synthetic dialogues with template-like wording. |
| Multi-Wiki-QA, Afri-MCQA | Non-commercial licence. |
| `unkadlabs/qor-af-soomaali` | Share-alike licence (CC-BY-SA). |

Full inspection results are in `licenses/SOURCES.md`.

### Limits of this declaration

The licences above are those stated by each dataset's publisher for the compilation. The web
datasets contain text written by many third parties, and the dataset licences do not settle the
rights in those original pages. Get legal review before training a commercial model.

## Load

```python
from datasets import load_dataset

raw = load_dataset("parquet", data_files={
    "train": "final/v1.0/raw_text/train-*.parquet",
    "validation": "final/v1.0/raw_text/validation-*.parquet",
    "test": "final/v1.0/raw_text/test-*.parquet"})

tr = load_dataset("json", data_files={
    "train": "final/v1.0/translation/train.jsonl",
    "validation": "final/v1.0/translation/validation.jsonl",
    "test": "final/v1.0/translation/test.jsonl"})

qa = load_dataset("json", data_files={
    "train": "final/v1.0/qa/train.jsonl",
    "validation": "final/v1.0/qa/validation.jsonl",
    "test": "final/v1.0/qa/test.jsonl"})
```

The instruction records hold plain `messages`. The chat template of the base model is applied
by the training framework, not stored in the data.

## Rebuild

```
python3 scripts/00_download.py raw test_quarantine
python3 scripts/10_stage1_clean.py raw work work/glotlid/model.bin
python3 scripts/20_stage2_dedup_write.py work .
python3 scripts/30_parallel_candidates.py raw work . work/glotlid/model.bin
python3 scripts/31_parallel_labse.py work work/labse
python3 scripts/32_parallel_build.py work .
python3 scripts/35_qa_build.py raw work . work/glotlid/model.bin
python3 scripts/40_validate.py . work
python3 scripts/50_datacard.py work .
```

Needs: `datasets huggingface_hub pyarrow ftfy xxhash fasttext sentence-transformers`, the GlotLID
model (`cis-lmu/glotlid`, `model.bin`) and LaBSE (`sentence-transformers/LaBSE`). The raw downloads
(about 3.9 GB) are not stored in this repository.

## Before using this for a commercial model

Read section 6 of the DATACARD. In short: licences were checked at dataset level only, no human
editor has reviewed the text, the question–answer set is small and not fact-checked, and there is
no multi-turn chat data yet.
