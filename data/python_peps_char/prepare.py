"""
Prepare the dataset for character-level language modeling from the
list of Python PEP documents (https://peps.python.org/pep-0001/).
"""
import os
import re
import tempfile
import zipfile

import pickle
import requests
from tqdm import tqdm
import numpy as np


def download_zip(url, dest):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        tqdm_bar = tqdm(total=total_size, desc="Download", unit="iB", leave=False)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=500 * 1024):  # 500 KB chunks
                tqdm_bar.update(len(chunk))
                f.write(chunk)

        tqdm_bar.close()


def extract_zip(zip_file_path):
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir


def clean_content(content):
    patterns = [
        # Remove metadata from preamble
        (
            re.compile(
                "^(Author:|Status:|Type:|Content-Type:|Created:|Post-History:|Version:|Last-Modified:)(.*?)(?=(^[\w-]+:|^\n))",
                flags=re.MULTILINE | re.DOTALL,
            ),
            "",
        ),
        # Remove image blocks
        (
            re.compile(
                r".. image::.*?(\n(?=^[^\s])|\Z)", flags=re.MULTILINE | re.DOTALL
            ),
            "",
        ),
        # Remove all '::' lines
        (
            re.compile(r"^\s*::\s*$", flags=re.MULTILINE),
            "",
        ),
        # Remove metadata from footer
        (
            re.compile("^\.\.\s+Local Variables:.*End:", re.MULTILINE | re.DOTALL),
            "",
        ),
        # Remove copyright notices
        (
            re.compile(
                "Copyright\n=+\n*$\s*This document.+\.$", flags=re.MULTILINE | re.DOTALL
            ),
            "",
        ),
        # Remove consecutive newlines
        (re.compile("^\n{2,}", flags=re.MULTILINE), "\n"),
        (re.compile("\n{2,}$", flags=re.MULTILINE), "\n"),
    ]

    cleaned_string = content
    for pat, repl in patterns:
        cleaned_string = pat.sub(repl, cleaned_string)
    return cleaned_string.strip()


input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

if not os.path.exists(input_file_path):
    zip_url = "https://codeload.github.com/python/peps/zip/refs/heads/main"
    zip_dest = tempfile.NamedTemporaryFile(delete=False).name
    download_zip(zip_url, zip_dest)
    print(f"Downloaded ZIP to {zip_dest}")

    repo_path = extract_zip(zip_dest)
    print(f"Extracted content to {repo_path}")

    contents_path = os.path.join(repo_path, "peps-main")
    pep_files = sorted(
        [
            f
            for f in os.listdir(contents_path)
            if f.startswith("pep-")
            and (f.endswith(".txt") or f.endswith(".rst"))
            and not f.startswith("pep-8")
        ]
    )

    with tqdm(total=len(pep_files), desc="Process", leave=False) as pbar:
        for pep_file in pep_files:
            with open(os.path.join(contents_path, pep_file), "r") as in_file:
                content = clean_content(in_file.read())
                with open(input_file_path, "a") as out_file:
                    out_file.write(content)
            pbar.update(1)

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
