# /// script
# dependencies = [
#   "julax",
# ]
#
# [tool.uv.sources]
# julax = { path = "../", editable = true }
# ///

import grain
import numpy as np


class FakeSource(grain.sources.RandomAccessDataSource):
    def __init__(self, seq_len: int = 256) -> None:
        self._seq_len = seq_len
        self._data = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 1024
        )

    def __getitem__(self, index: int):
        return {
            "input_ids": self._data[index : index + self._seq_len],
            "target_labels": self._data[index + 1 : index + 1 + self._seq_len],
        }

    def __len__(self) -> int:
        return len(self._data) - self._seq_len


dataset = grain.MapDataset.source(FakeSource()).shuffle(seed=10).batch(batch_size=2)
