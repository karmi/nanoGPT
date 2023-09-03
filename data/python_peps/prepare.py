"""
Prepare the dataset for character-level language modeling from the
list of Python PEP documents (https://peps.python.org/pep-0001/).
"""
import os
import re
from multiprocessing.pool import ThreadPool

import pickle
import requests
from tqdm import tqdm
import tiktoken
import numpy as np


def fetch_pep_list_from_github():
    url = "https://api.github.com/repos/python/peps/contents"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from GitHub API: {response.status_code}")

    files = response.json()
    pep_files = [
        f["name"]
        for f in files
        if f["name"].startswith("pep-")
        and (f["name"].endswith(".txt") or f["name"].endswith(".rst"))
    ]

    return pep_files


def download_pep(pep_filename):
    # print(f"Downloading [{pep_filename}]")
    base_url = "https://raw.githubusercontent.com/python/peps/main/"
    url = f"{base_url}{pep_filename}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to download {pep_filename}: {response.status_code}")

    return response.text


def download_peps_in_parallel(pep_list, num_threads=16):
    with ThreadPool(num_threads) as pool:
        pep_contents = list(
            tqdm(pool.imap(download_pep, pep_list), total=len(pep_list))
        )
    return dict(zip(pep_list, pep_contents))


input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

if not os.path.exists(input_file_path):
    pep_list = fetch_pep_list_from_github()

    print(f"downloading {len(pep_list)} PEP documents...")
    pep_contents = download_peps_in_parallel(pep_list)

    with open(input_file_path, "w") as f:
        f.write("\n\n".join(pep_contents.values()))

# -------------------------------------------------------

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# train has 2,906,859 tokens
# val has 322,886 tokens
