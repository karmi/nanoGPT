"""
Prepare the Jane Austen dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
from io import BytesIO
from zipfile import ZipFile

import pickle
import requests
import numpy as np

# download the jane austen dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

start_marker = "THE WORKS OF JANE AUSTEN"
end_marker = "End of the Project Gutenberg EBook of The Complete Works of Jane Austen"
inside_text = False

if not os.path.exists(input_file_path):
    zip_url = "https://www.gutenberg.org/files/31100/31100.zip"
    response = requests.get(zip_url)
    zip_file = ZipFile(BytesIO(response.content))

    with zip_file.open("31100.txt") as text:
        with open(input_file_path, "w") as f:
            for line in text:
                line = line.decode("latin-1")
                if line.startswith(start_marker):
                    inside_text = True
                    f.write(start_marker)
                    f.write(line)
                    continue
                if line.startswith(end_marker):
                    inside_text = False
                    break
                if inside_text:
                    f.write(line)

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

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
