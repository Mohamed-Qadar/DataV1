# Somali AI — DataV1

Training-ready Somali data, version 1.0 (built 2026-11-15).

| Folder | Content |
|---|---|
| `final/v1.0/raw_text/` | Cleaned, deduplicated Somali text for language training. Parquet shards with `id`, `text`, `source`, `license`. Splits: `train-*`, `validation-*`, `test-*`. |
| `final/v1.0/translation/` | Somali↔English translation instructions in chat `messages` format. `train.jsonl`, `validation.jsonl`, `test.jsonl`. |
| `final/v1.0/DATACARD.md` | Counts for every cleaning step, source and licence distribution, known limitations. Read this first. |
| `processed/parallel_pairs_clean.jsonl` | Every parallel pair that passed cleaning, with its LaBSE score. |
| `test_quarantine/` | Test sets. Never use these for training. |
| `licenses/` | `SOURCES.md` (inspection results and decisions) and `ATTRIBUTION.md` (required credits). |
| `logs/` | Up to 100 random examples of what each cleaning step removed. Read them to judge the filters. |
| `scripts/` | The pipeline that produced this version. |

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
```

The translation records hold plain `messages`. The chat template of the base model is applied
by the training framework, not stored in the data.

## Rebuild

```
python3 scripts/00_download.py raw test_quarantine
python3 scripts/10_stage1_clean.py raw work work/glotlid/model.bin
python3 scripts/20_stage2_dedup_write.py work .
python3 scripts/30_parallel_candidates.py raw work . work/glotlid/model.bin
python3 scripts/31_parallel_labse.py work work/labse
python3 scripts/32_parallel_build.py work .
python3 scripts/40_validate.py . work
python3 scripts/50_datacard.py work .
```

Needs: `datasets huggingface_hub pyarrow ftfy xxhash fasttext sentence-transformers`, the GlotLID
model (`cis-lmu/glotlid`, `model.bin`) and LaBSE (`sentence-transformers/LaBSE`). The raw downloads
(about 3.7 GB) are not stored in this repository.

## Before using this for a commercial model

Read section 5 of the DATACARD. In short: licences were checked at dataset level only, no human
editor has reviewed the text, and there is no question–answer data in this version yet.
