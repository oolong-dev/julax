# Reproduce https://sdbuchanan.com/blog/jax-2/

from functools import partial

import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.nn.initializers import truncated_normal

from julax.experiment import Experiment, run
from julax.layers import (
    Chain,
    Embedding,
    LayerNorm,
    Linear,
    Parallel,
    Repeat,
    Residual,
    RotaryEmbedding,
    Learner,
    Trainer,
    Rearrange,
)
from julax.utils import identity


import logging
from absl import logging as absl_logging

logging.root.setLevel(logging.INFO)
absl_logging.use_python_logging()


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
    param_std: float = 0.02,
):
    return Experiment(
        name="mini_transformer",
        trainer=Trainer(
            learner=Learner(
                feature_name="input_ids",
                label_name="target_labels",
                model=Chain(
                    emb=Embedding(
                        in_dim=num_vocab,
                        out_dim=dim,
                        w_init=truncated_normal(stddev=param_std),
                    ),
                    blocks=Repeat(
                        n=num_layers,
                        layer=Chain(
                            attn=Residual(
                                processor=Chain(
                                    norm_attn=LayerNorm(dim=dim),
                                    attn=Chain(
                                        # qkv projection
                                        Linear(
                                            in_dim=dim,
                                            out_dim=3 * dim,
                                            w_init=truncated_normal(stddev=param_std),
                                            b_init=None,
                                        ),
                                        Rearrange(
                                            "B T (qkv N H) -> B T (qkv N) H",
                                            B=global_batch_size,
                                            T=seq_len,
                                            qkv=3,
                                            N=num_heads,
                                            H=head_dim,
                                        ),
                                        partial(
                                            jnp.split, indices_or_sections=3, axis=2
                                        ),
                                        Parallel(
                                            query=RotaryEmbedding(
                                                embedding_dims=head_dim,
                                                fprop_dtype=jnp.float32,
                                            ),
                                            key=RotaryEmbedding(
                                                embedding_dims=head_dim,
                                                fprop_dtype=jnp.float32,
                                            ),
                                            value=identity,
                                            reduce=lambda qkv: jax.nn.dot_product_attention(
                                                qkv["query"],
                                                qkv["key"],
                                                qkv["value"],
                                                is_causal=True,
                                            ),
                                        ),
                                        Rearrange(
                                            "B T N H -> B T (N H)",
                                            B=global_batch_size,
                                            T=seq_len,
                                            N=num_heads,
                                            H=head_dim,
                                        ),
                                        Linear(
                                            in_dim=dim,
                                            out_dim=dim,
                                            w_init=truncated_normal(stddev=param_std),
                                            b_init=None,
                                        ),
                                    ),
                                )
                            ),
                            mlp=Residual(
                                processor=Chain(
                                    norm_mlp=LayerNorm(dim=dim),
                                    mlp=Chain(
                                        up=Linear(
                                            in_dim=dim,
                                            out_dim=4 * dim,
                                            w_init=truncated_normal(stddev=param_std),
                                            b_init=None,
                                        ),
                                        act=jax.nn.gelu,
                                        down=Linear(
                                            in_dim=4 * dim,
                                            out_dim=dim,
                                            w_init=truncated_normal(stddev=param_std),
                                            b_init=None,
                                        ),
                                    ),
                                )
                            ),
                        ),
                    ),
                    unemb=Linear(
                        in_dim=dim,
                        out_dim=num_vocab,
                        w_init=truncated_normal(stddev=param_std),
                    ),
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
    )


x = main()
run(x)
