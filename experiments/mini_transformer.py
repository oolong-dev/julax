# /// script
# dependencies = [
#   "julax",
# ]
#
# [tool.uv.sources]
# julax = { path = "../", editable = true }
# ///

import grain
import jax
import numpy as np
import optax
from julax.core import Learner, Trainer
from julax.experiment import Experiment
from julax.layers import (
    Chain,
    Linear,
    LayerNorm,
    Repeated,
    SkipConnection,
    Embedding,
    Unembedding,
)
from julax.observers import default_observer


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


def main(
    seed: int = 5,
    seq_len: int = 256,
    global_batch_size: int = 128,
    num_steps: int = 1000,
    num_vocab: int = 10,
    dim: int = 768,
    num_heads: int = 12,
    head_dim: int = 64,
    num_layers: int = 2,
):
    return Experiment(
        name="mini_transformer",
        trainer=Trainer(
            learner=Learner(
                feature_name="input_ids",
                label_name="target_labels",
                model=Chain(
                    emb=Embedding(in_dim=num_vocab, out_dim=dim),
                    blocks=Repeated(
                        n=num_layers,
                        layer=Chain(
                            attn=SkipConnection(
                                layer=Chain(
                                    norm_attn=LayerNorm(dim=dim), attn=lambda x: x
                                )
                            ),
                            mlp=SkipConnection(
                                layer=Chain(
                                    norm_mlp=LayerNorm(dim=dim),
                                    act=jax.nn.gelu,
                                    mlp=Chain(
                                        up=Linear(
                                            in_dim=dim, out_dim=4 * dim, b_init=None
                                        ),
                                        down=Linear(
                                            in_dim=4 * dim, out_dim=dim, b_init=None
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ),
                    unemb=Unembedding(in_dim=dim, out_dim=num_vocab),
                ),
                loss_fn=optax.softmax_cross_entropy_with_integer_labels,
            ),
            optimizer=optax.sgd(0.01),
        ),
        dataset=(
            grain.MapDataset.source(FakeSource(seq_len))
            .shuffle(seed=seed)
            .repeat()
            .batch(batch_size=global_batch_size)
            .slice(slice(num_steps))
            .to_iter_dataset()
        ),
        observer=default_observer(),
    )


x = main()
x.run()
x.close()
