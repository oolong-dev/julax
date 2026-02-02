import grain
import os
import pickle
import numpy as np


from grain._src.core.sharding import ShardByJaxProcess, even_split
from grain.experimental import FlatMapIterDataset, FlatMapTransform, ParquetIterDataset


class Tokenize(FlatMapTransform):
    def __init__(self, tokenizer_path: str) -> None:
        super().__init__()
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
            self.bos_token_id = self.tokenizer.encode_single_token("<|bos|>")

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id] + self.tokenizer.encode_ordinary(text)

    def flat_map(self, element):
        return self.encode(element)

    def get_segment_ids(self, tokens: np.ndarray):
        assert tokens.ndim == 2
        bos_mask = tokens == self.bos_token_id
        bos_mask[:, 0] = False
        segment_ids = np.cumsum(bos_mask, axis=1)

        return segment_ids


def create_dataset(
    batch_size: int,
    seq_len: int,
    data_dir: str,
    tokenizer_path: str,
    split: str = "train",
    seed: int = 2025,
) -> grain.IterDataset:
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
    if split == "train":
        files = files[:-1]
    else:
        # TODO:
        raise ValueError("Unsupported yet")

    tokenize = Tokenize(tokenizer_path)

    # TODO: window shuffle?
    # TODO: prefetch to device?
    ds = grain.experimental.InterleaveIterDataset(
        grain.MapDataset.source(files)
        .shuffle(seed=seed)
        .slice(slice(*even_split(len(files), ShardByJaxProcess(drop_remainder=True))))
        .map(
            lambda file_path: FlatMapIterDataset(
                ParquetIterDataset(file_path).map(lambda x: x["text"]), tokenize
            )
            .batch(batch_size * seq_len + 1)
            .map(np.array)
            .map(
                lambda x: {
                    "inputs": {
                        "token_ids": x[:-1].reshape(batch_size, seq_len),
                        "segment_ids": tokenize.get_segment_ids(
                            x[:-1].reshape(batch_size, seq_len)
                        ),
                    },
                    "target_labels": x[1:].reshape(batch_size, seq_len),
                }
            )
        ),  # pyright: ignore[reportArgumentType]
        cycle_length=4,
    )

    return ds.mp_prefetch(
        grain.MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1)
    )
