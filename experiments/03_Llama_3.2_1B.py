# /// script
# dependencies = [
#   "julax",
#   "pyarrow",
# ]
#
# [tool.uv.sources]
# julax = { path = "../", editable = true }
# ///

import os
import pickle

import grain
import jax
import jax.numpy as jnp
import numpy as np
from grain._src.core.sharding import ShardByJaxProcess, even_split
from grain.experimental import FlatMapIterDataset, FlatMapTransform, ParquetIterDataset
from jax import Array

from julax.base import Dtype
from julax.core import LayerBase, Param, State
from julax.einops import Rearrange
from julax.layers import (
    Branch,
    Chain,
    Embedding,
    Linear,
    Parallel,
    Repeat,
    Residual,
    RMSNorm,
    Select,
)
from julax.utils import identity


# Adapted from:
# https://github.com/AI-Hypercomputer/maxtext/blob/9204d6bbbf8bb19a05ebed72a55cfec687e0e044/src/MaxText/layers/embeddings.py#L486-L622
# TODO: The real and imaginary part are interleaved. benchmark with the HF
# transformer style (first half as real,  second half as imaginary).
def apply_rotary_emb(
    inputs: jax.Array,
    timescale: jax.Array,
    position: None | jax.Array = None,
    fprop_dtype: Dtype | None = jnp.bfloat16,
) -> jax.Array:
    """Applies LLaMA variant of rotary position embedding.

    Args:
        inputs: The input sequence on which to apply the Rotary position
            embedding. It is assumed of shape [B, S, N, H].
        position: Optional position array [B, S]. Only needed when the sequence
            is packed.

    Returns:
        A jax.Array of shape [B, S, N, H] with rotary position embeddings applied.
    """
    # Ensure input is 4D
    if len(inputs.shape) != 4:
        raise ValueError(
            "Input is assumed to be a rank 4 tensor of shape [B, S, N, H]."
        )
    embedding_dims = inputs.shape[3]

    # Shift the inputs left and right as per LLaMA's specific behavior
    inputs_shifted_left = jnp.concatenate([inputs[..., 1:], inputs[..., :1]], axis=-1)
    inputs_shifted_right = jnp.concatenate(
        [inputs[..., -1:], inputs[..., :-1]], axis=-1
    )
    inputs_shifted = jax.lax.select(
        jnp.tile(
            jnp.mod(jnp.arange(embedding_dims, dtype=jnp.int32), 2),
            inputs.shape[:-1] + (1,),
        ),
        inputs_shifted_right,
        inputs_shifted_left,
    )

    # Determine positions if not provided
    if position is None:
        seq_length = inputs.shape[1]
        position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]

    # Calculate sinusoidal input
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / timescale

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    # Apply alternating sign
    sign = jnp.tile(jnp.array([-1, 1]), embedding_dims // 2)

    # Combine original inputs with sinusoidal information
    outputs = inputs * cos + inputs_shifted * sin * sign

    if fprop_dtype:
        outputs = outputs.astype(fprop_dtype)

    # sin, cos = jnp.split(sinusoidal_pos, 2, axis=-1)
    # sin_pos = jnp.reshape(jnp.stack([sin, sin], axis=-1), sinusoidal_pos.shape)
    # cos_pos = jnp.reshape(jnp.stack([cos, cos], axis=-1), sinusoidal_pos.shape)
    # rotate_half_inputs = jnp.reshape(
    #     jnp.stack([-inputs[..., 1::2], inputs[..., ::2]], axis=-1), inputs.shape
    # )
    # output = inputs * cos_pos + rotate_half_inputs * sin_pos

    return outputs


class LLaMARotaryEmbedding(LayerBase):
    embedding_dims: int
    min_timescale: int
    max_timescale: int
    cast_as_fprop_dtype: bool = True
    fprop_dtype: Dtype = jnp.bfloat16

    scaling_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def _apply_scaling_factor(self, freq):
        """apply scaling factor to rotary position embedding."""
        low_freq_wavelen = self.original_max_position_embeddings / self.low_freq_factor
        high_freq_wavelen = (
            self.original_max_position_embeddings / self.high_freq_factor
        )
        wavelen = 2 * jnp.pi / freq

        def lower_wavelen(freq):
            return freq

        def bigger_or_equal_wavelen(freq):
            def bigger_wavelen(freq):
                return freq / self.scaling_factor

            def equal_wavelen(freq):
                smooth = (
                    self.original_max_position_embeddings / wavelen
                    - self.low_freq_factor
                ) / (self.high_freq_factor - self.low_freq_factor)
                return (1 - smooth) * freq / self.scaling_factor + smooth * freq

            bigger_wavelen_cond = wavelen > low_freq_wavelen
            return jax.lax.cond(
                bigger_wavelen_cond, bigger_wavelen, equal_wavelen, freq
            )

        lower_wavelen_cond = wavelen < high_freq_wavelen
        return jax.lax.cond(
            lower_wavelen_cond, lower_wavelen, bigger_or_equal_wavelen, freq
        )

    def state(self, rng) -> State:
        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
        fraction = jnp.repeat(fraction, 2)
        timescale = (
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        )
        timescale = 1.0 / jax.vmap(self._apply_scaling_factor)(1.0 / timescale)
        return State(timescale=timescale)

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        return apply_rotary_emb(
            x,
            s["timescale"],
            position=None,
            fprop_dtype=self.fprop_dtype if self.cast_as_fprop_dtype else None,
        ), s


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


def attention(qkv, timescale):
    q, k, v = qkv
    q = apply_rotary_emb(q, timescale)
    k = apply_rotary_emb(k, timescale)
    o = jax.nn.dot_product_attention(q, k, v, is_causal=True)
    return o


def transformer_block(
    batch_size, seq_len, dim, num_q_heads, num_kv_heads, head_dim, ffn_hidden_dim
):
    Branch(  # {hidden(in), timescale} => {hidden(out), timescale}
        hidden=Chain(
            attn=Residual(
                processor=Chain(
                    Parallel(
                        qkv=Chain(
                            norm=RMSNorm(dim=dim),
                            qkv_proj=Branch(
                                q=Chain(
                                    Linear(in_dim=dim, out_dim=num_q_heads * head_dim),
                                    Rearrange(
                                        "B T (N H) -> B T N H",
                                        B=batch_size,
                                        T=seq_len,
                                        N=num_q_heads,
                                        H=head_dim,
                                    ),
                                ),
                                k=Chain(
                                    Linear(in_dim=dim, out_dim=num_kv_heads * head_dim),
                                    Rearrange(
                                        "B S (K H) -> B S K H",
                                        B=batch_size,
                                        S=seq_len,
                                        K=num_kv_heads,
                                        H=head_dim,
                                    ),
                                ),
                                v=Chain(
                                    Linear(in_dim=dim, out_dim=num_kv_heads * head_dim),
                                    Rearrange(
                                        "B S (K H) -> B S K H",
                                        B=batch_size,
                                        S=seq_len,
                                        K=num_kv_heads,
                                        H=head_dim,
                                    ),
                                ),
                            ),
                        ),
                        timescale=identity,
                        reduce=attention,
                    ),
                    Rearrange(
                        "B T N H -> B T (N H)",
                        B=batch_size,
                        T=seq_len,
                        N=num_q_heads,
                        H=head_dim,
                    ),
                    Linear(in_dim=dim, out_dim=dim),
                ),
                skip_through=Select(key="hidden"),
            ),
            ffn=Residual(
                Chain(
                    norm=RMSNorm(dim=dim),
                    up=Branch(
                        up_proj=Linear(in_dim=dim, out_dim=ffn_hidden_dim),
                        gate_proj=Chain(
                            proj=Linear(in_dim=dim, out_dim=ffn_hidden_dim),
                            activation=jax.nn.silu,
                        ),
                        reduce=jnp.multiply,
                    ),
                    down=Linear(in_dim=ffn_hidden_dim, out_dim=dim),
                )
            ),
        ),
        timescale=Select(key="timescale"),
    )


class Transformer(LayerBase):
    emb: Embedding
    rope: LLaMARotaryEmbedding
    blocks: Repeat
    out_norm: RMSNorm

    def forward(self, x: dict, p: Param, s: State) -> tuple[Array, State]:
        S = State(rope=s["rope"])

        h = x["token_ids"]
        h, S["emb"] = self.emb(h, p["emb"], s["emb"])
        h, S["blocks"] = self.blocks(
            {"hidden": h, "timescale": s["rope"]["timescale"]}, p["blocks"], s["blocks"]
        )
        h, S["out_norm"] = self.out_norm(h, p["out_norm"], s["out_norm"])

        o = self.emb.attend(h, p["emb"])

        return o, S
