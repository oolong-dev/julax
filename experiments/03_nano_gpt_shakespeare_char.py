# /// script
# dependencies = [
#   "julax",
#   "requests",
#   "numpy",
# ]
#
# [tool.uv.sources]
# julax = { path = "../", editable = true }
# ///

import grain
import requests
import os
import numpy as np


class ShakespeareChar(grain.sources.RandomAccessDataSource):
    def __init__(self, seq_len: int = 256) -> None:
        self.seq_len = seq_len
        input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
        if not os.path.exists(input_file_path):
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(input_file_path, "w") as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, "r") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.data = np.array([self.stoi[c] for c in text], dtype=np.int32)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, index: int):
        x = self.data[index : index + self.seq_len]
        y = self.data[index + 1 : index + self.seq_len + 1]
        return {
            "input_ids": x,
            "target_labels": y,
        }
