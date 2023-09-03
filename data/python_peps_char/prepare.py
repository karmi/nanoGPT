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
            tqdm(
                pool.imap(download_pep, pep_list), total=len(pep_list), desc="Download"
            )
        )
    return dict(zip(pep_list, pep_contents))


input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

if not os.path.exists(input_file_path):
    pep_list = fetch_pep_list_from_github()

    ignore_peps = [
        "pep-8001.rst",
        "pep-8001.rst",
        "pep-8103.rst",
        "pep-8104.rst",
    ]

    print(f"downloading {len(pep_list)-len(ignore_peps)} PEP documents...")
    pep_contents = download_peps_in_parallel(
        [item for item in pep_list if not item in ignore_peps]
    )

    ignore_lines = [
        re.compile("^::"),
        re.compile("^.. "),
        re.compile("Copyright\n=+^", re.MULTILINE),
    ]

    with open(input_file_path, "w") as f:
        f.write("\n\n".join(pep_contents.values()))

# -------------------------------------------------------

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


# create the train and test splits
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# length of dataset in characters: 11,303,894
# train has 10,173,504 tokens
# val has 1,130,390 tokens
