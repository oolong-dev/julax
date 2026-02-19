import grain
import numpy as np

import io
import json
import gzip
from transformers import AutoTokenizer
from grain._src.core.sharding import ShardByJaxProcess, even_split
from grain import IterDataset, DatasetIterator
from etils.epath import Path


class JsonlDatasetIterator(DatasetIterator):
    def __init__(self, file_path: Path):
        super().__init__()
        self._file_path = file_path
        self._file = None
        self._raw_ctx = None
        self._line = 0

    def _seek(self, line: int = 0):
        self.close()
        # epath.Path.open() returns a context manager, so we must __enter__ it
        self._raw_ctx = self._file_path.open("rb")
        raw_file = self._raw_ctx.__enter__()
        self._file = io.TextIOWrapper(gzip.GzipFile(fileobj=raw_file), encoding="utf-8")
        for _ in range(line):
            next(self._file)
        return self._file

    def __next__(self):
        line = next(self._file or self._seek())
        data = json.loads(line)
        text = data["text"]
        self._line += 1
        return text

    def get_state(self) -> dict:
        return {"line": self._line}

    def set_state(self, state: dict):
        self._line = state["line"]
        self._seek(self._line)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        if self._raw_ctx:
            self._raw_ctx.__exit__(None, None, None)
            self._raw_ctx = None


class JsonlIterDataset(IterDataset):
    def __init__(self, file_path: Path):
        super().__init__()
        self._file_path = file_path

    def __iter__(self):
        return JsonlDatasetIterator(self._file_path)


class VanillaBatchIterator(DatasetIterator):
    def __init__(self, ds_iter: DatasetIterator, batch_size: int, seq_len: int):
        super().__init__()
        self._inner_iter = iter(ds_iter)
        self._batch_size = batch_size
        self._seq_len = seq_len

        self._token_buffer = []

    def __next__(self):
        B, T = self._batch_size, self._seq_len
        while len(self._token_buffer) <= B * T:
            tokens = next(self._inner_iter)
            self._token_buffer.extend(tokens)
        batch = np.array(self._token_buffer[: B * T + 1])
        self._token_buffer = self._token_buffer[B * T :]
        return {
            "inputs": {
                "token_ids": batch[:-1].reshape(B, T),
            },
            "target_labels": batch[1:].reshape(B, T),
        }

    def get_state(self) -> dict:
        return {
            "inner_iter": self._inner_iter.get_state(),
            "token_buffer": self._token_buffer,
        }

    def set_state(self, state: dict):
        self._inner_iter.set_state(state["inner_iter"])
        self._token_buffer = state["token_buffer"]

    def close(self) -> None:
        self._inner_iter.close()


class VanillaIterDataset(IterDataset):
    def __init__(self, file_path: Path, tokenizer_dir, batch_size, seq_len):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._ds = JsonlIterDataset(file_path).map(lambda x: self._tokenizer.encode(x))

    def __iter__(self) -> DatasetIterator:
        return VanillaBatchIterator(iter(self._ds), self._batch_size, self._seq_len)


def create_dataset(
    batch_size: int,
    seq_len: int,
    tokenizer_dir: str,
    data_dir: str,
    split_pattern: str = "c4-train.*",
    seed: int = 2026,
    n_open_files: int = 4,
    n_prefetch_per_file: int = 4,
) -> IterDataset:
    files = [p for p in Path(data_dir).iterdir() if p.match(split_pattern)]

    ds = grain.experimental.InterleaveIterDataset(
        grain.MapDataset.source(files)
        .shuffle(seed=seed)
        .slice(slice(*even_split(len(files), ShardByJaxProcess(drop_remainder=True))))
        .map(
            lambda file_path: VanillaIterDataset(
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
