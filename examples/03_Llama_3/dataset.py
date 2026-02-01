import os
import grain
import jax.numpy as jnp


import json
import gzip
from transformers import AutoTokenizer
from grain._src.core.sharding import ShardByJaxProcess, even_split
from grain import IterDataset, DatasetIterator


class _TextDatasetIterator(DatasetIterator):
    def __init__(self, file_path: str, tokenizer, batch_size: int, seq_len: int):
        super().__init__()
        self._file_path = file_path
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._seq_len = seq_len

        self._line = 0
        self._token_buffer = []
        self._file = self._seek_line(file_path, self._line)

    def _seek_line(self, file_path: str, line: int):
        file = gzip.open(file_path, "rt", encoding="utf-8")
        for _ in range(line):
            next(file)
        return file

    def __next__(self):
        B, T = self._batch_size, self._seq_len
        while len(self._token_buffer) <= B * T:
            line = next(self._file)
            data = json.loads(line)
            text = data["text"]
            tokens = self._tokenizer.encode(text)
            self._token_buffer.extend(tokens)
            self._line += 1
        batch = self._token_buffer[: B * T + 1]
        self._token_buffer = self._token_buffer[B * T :]
        return {
            "inputs": jnp.array(batch[:-1]).reshape(B, T),
            "target_labels": jnp.array(batch[1:]).reshape(B, T),
        }

    def get_state(self) -> dict:
        return {
            "line": self._line,
            "token_buffer": self._token_buffer,
        }

    def set_state(self, state: dict):
        if hasattr(self, "_file") and self._file:
            self._file.close()
        self._line = state["line"]
        self._token_buffer = state["token_buffer"]
        self._file = self._seek_line(self._file_path, self._line)
        self.start_prefetch()

    def close(self) -> None:
        if hasattr(self, "_file") and self._file:
            self._file.close()


class TextIterDataset(IterDataset):
    def __init__(
        self, file_path: str, tokenizer_dir: str, batch_size: int, seq_len: int
    ):
        super().__init__()
        self._file_path = file_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self._batch_size = batch_size
        self._seq_len = seq_len

    def __iter__(self):
        return _TextDatasetIterator(
            self._file_path, self._tokenizer, self._batch_size, self._seq_len
        )


def create_dataset(
    batch_size: int,
    seq_len: int,
    data_dir: str,
    tokenizer_dir: str,
    seed: int = 2026,
    n_open_files: int = 4,
    n_prefetch_per_file: int = 4,
) -> IterDataset:
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])

    ds = grain.experimental.InterleaveIterDataset(
        grain.MapDataset.source(files)
        .shuffle(seed=seed)
        .slice(slice(*even_split(len(files), ShardByJaxProcess(drop_remainder=True))))
        .map(
            lambda file_path: TextIterDataset(
                file_path, tokenizer_dir, batch_size, seq_len
            )
        ),  # pyright: ignore[reportArgumentType]
        cycle_length=n_open_files,
    ).mp_prefetch(
        grain.MultiprocessingOptions(
            num_workers=n_open_files, per_worker_buffer_size=n_prefetch_per_file
        )
    )

    return ds
