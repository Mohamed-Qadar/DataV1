"""Download only the approved Somali subsets into raw/ (never modified afterwards).

Usage: python3 -I 00_download.py <raw_dir> <test_quarantine_dir>
"""
import os
import sys
import urllib.request

from huggingface_hub import HfApi, hf_hub_download

RAW, QUAR = sys.argv[1], sys.argv[2]
api = HfApi()


def get(repo, files, sub):
    out = os.path.join(RAW, sub)
    os.makedirs(out, exist_ok=True)
    for f in files:
        hf_hub_download(repo, f, repo_type="dataset", local_dir=out)
    print("ok", sub, len(files), flush=True)


get("HuggingFaceFW/finepdfs", ["data/som_Latn/train/000_00000.parquet"], "finepdfs")
get("Helsinki-NLP/opus_infopankki", ["en-so/train-00000-of-00001.parquet"], "infopankki")
get("google/smol", ["smolsent/en_so.jsonl", "smoldoc/en_so.jsonl"], "smol")
get("HuggingFaceFW/fineweb-2", ["data/som_Latn/train/000_00000.parquet"], "fineweb2")
clean = sorted(i.path for i in api.list_repo_tree("allenai/MADLAD-400", path_in_repo="data-v1p5/so",
                                                  repo_type="dataset") if "clean" in i.path)
get("allenai/MADLAD-400", clean, "madlad400")
get("HPLT/HPLT2.0_cleaned", [f"som_Latn/train-0000{i}-of-00007.parquet" for i in range(7)], "hplt2")

os.makedirs(QUAR, exist_ok=True)
hf_hub_download("facebook/belebele", "data/som_Latn.jsonl", repo_type="dataset",
                local_dir=os.path.join(QUAR, "belebele"))
print("ok belebele", flush=True)

nllb = os.path.join(RAW, "nllb")
os.makedirs(nllb, exist_ok=True)
dst = os.path.join(nllb, "en-so.txt.zip")
if not os.path.exists(dst):
    urllib.request.urlretrieve("https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-so.txt.zip", dst)
print("ok nllb", os.path.getsize(dst), flush=True)
print("DONE", flush=True)
