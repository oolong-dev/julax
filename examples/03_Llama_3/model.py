import jax
import jax.numpy as jnp
from julax.base import Array, Dtype, Param, State, PRNG
from julax.layers import (
    LayerBase,
    Branch,
    Chain,
    Embedding,
    Linear,
    Parallel,
    Repeat,
    Residual,
    RMSNorm,
    Select,
    Rearrange,
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
    # Determine positions if not provided
    if position is None:
        seq_length = inputs.shape[1]
        position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]

    # Calculate sinusoidal input
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / timescale

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    r, i = jnp.split(inputs, 2, axis=-1)
    pos_r = cos * r - sin * i
    pos_i = sin * r + cos * i
    outputs = jnp.concatenate([pos_r, pos_i], axis=-1)

    if fprop_dtype:
        outputs = outputs.astype(fprop_dtype)

    return outputs


class LLaMARotaryEmbedding(LayerBase):
    embedding_dims: int
    min_timescale: int = 1
    max_timescale: int = 10_000
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


def attention(inputs):
    q = inputs["hidden"]["q"]
    k = inputs["hidden"]["k"]
    v = inputs["hidden"]["v"]
    q = apply_rotary_emb(q, inputs["timescale"])
    k = apply_rotary_emb(k, inputs["timescale"])
    o = jax.nn.dot_product_attention(q, k, v, is_causal=True)
    return o


class CachedAttention(LayerBase):
    batch_size: int
    cache_size: int
    num_kv_heads: int
    head_dim: int
    dtype: Dtype = jnp.bfloat16

    def state(self, rng: PRNG) -> State:
        return State(
            k=jnp.zeros(
                (self.batch_size, self.cache_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
            ),
            v=jnp.zeros(
                (self.batch_size, self.cache_size, self.num_kv_heads, self.head_dim),
                dtype=self.dtype,
            ),
            end_index=jnp.zeros(1, dtype=jnp.int32),
        )

    def state_length(self) -> int:
        return (
            2 * self.batch_size * self.cache_size * self.num_kv_heads * self.head_dim
            + 1
        )

    def forward(self, inputs: dict, p: Param, s: Param) -> tuple[Array, State]:
        q = inputs["hidden"]["q"]
        k = inputs["hidden"]["k"]
        v = inputs["hidden"]["v"]
        seq_len = q.shape[1]

        timescale = inputs["timescale"]
        position = inputs["position"]

        q = apply_rotary_emb(q, timescale, position)
        k = apply_rotary_emb(k, timescale, position)

        slice_indices = (0, s["end_index"][0], 0, 0)
        k = jax.lax.dynamic_update_slice(s["k"], k, slice_indices)
        v = jax.lax.dynamic_update_slice(s["v"], v, slice_indices)
        # o = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        query_positions = jnp.arange(seq_len) + s["end_index"][0]
        key_positions = jnp.arange(self.cache_size)
        attention_mask = key_positions[None, :] <= query_positions[:, None]
        o = jax.nn.dot_product_attention(q, k, v, mask=attention_mask[None, None, :, :])

        S = State(
            k=k,
            v=v,
            end_index=s["end_index"] + seq_len,
        )
        return o, S


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
            {
                "hidden": h,
                "timescale": s["rope"]["timescale"],
                "position": x.get("position", None),
            },
            p["blocks"],
            s["blocks"],
        )
        h, S["out_norm"] = self.out_norm(h["hidden"], p["out_norm"], s["out_norm"])

        o = self.emb.attend(h, p["emb"])

        return o, S


def create_model(
    model="llama_3.2_1b", is_training: bool = True, cache_size=None, batch_size=1
):
    if is_training is False:
        assert cache_size is not None, "cache_size must be provided for inference."

    match model:
        case "llama_3.2_1b":
            dim = 2048
            num_q_heads = 32
            num_kv_heads = 8
            head_dim = 64
            ffn_hidden_dim = 8192
            vocab_size = 128256

            return Transformer(
                emb=Embedding(in_dim=vocab_size, out_dim=dim, param_dtype=jnp.bfloat16),
                rope=LLaMARotaryEmbedding(
                    embedding_dims=head_dim,
                    min_timescale=1,
                    max_timescale=500_000,
                ),
                out_norm=RMSNorm(dim=dim, eps=1e-05, scale_dtype=jnp.bfloat16),
                blocks=Repeat(
                    n=16,
                    layer=Branch(
                        hidden=Chain(
                            attn=Residual(
                                Chain(
                                    Parallel(
                                        hidden=Chain(
                                            norm=RMSNorm(dim=dim, eps=1e-05),
                                            qkv_proj=Branch(
                                                q=Chain(
                                                    Linear(
                                                        in_dim=dim,
                                                        out_dim=num_q_heads * head_dim,
                                                        param_dtype=jnp.bfloat16,
                                                    ),
                                                    Rearrange(
                                                        "B T (N H) -> B T N H",
                                                        N=num_q_heads,
                                                        H=head_dim,
                                                    ),
                                                ),
                                                k=Chain(
                                                    Linear(
                                                        in_dim=dim,
                                                        out_dim=num_kv_heads * head_dim,
                                                        param_dtype=jnp.bfloat16,
                                                    ),
                                                    Rearrange(
                                                        "B S (K H) -> B S K H",
                                                        K=num_kv_heads,
                                                        H=head_dim,
                                                    ),
                                                ),
                                                v=Chain(
                                                    Linear(
                                                        in_dim=dim,
                                                        out_dim=num_kv_heads * head_dim,
                                                        param_dtype=jnp.bfloat16,
                                                    ),
                                                    Rearrange(
                                                        "B S (K H) -> B S K H",
                                                        K=num_kv_heads,
                                                        H=head_dim,
                                                    ),
                                                ),
                                            ),
                                        ),
                                        timescale=identity,
                                        position=identity,
                                        reduce=attention
                                        if cache_size is None
                                        else CachedAttention(
                                            batch_size=batch_size,
                                            cache_size=cache_size,
                                            num_kv_heads=num_kv_heads,
                                            head_dim=head_dim,
                                            dtype=jnp.bfloat16,
                                        ),
                                    ),
                                    Rearrange(
                                        "B T N H -> B T (N H)",
                                        N=num_q_heads,
                                        H=head_dim,
                                    ),
                                    Linear(
                                        in_dim=dim,
                                        out_dim=dim,
                                        param_dtype=jnp.bfloat16,
                                    ),
                                ),
                                skip_through=Select(key="hidden"),
                            ),
                            ffn=Residual(
                                Chain(
                                    norm=RMSNorm(dim=dim, eps=1e-05),
                                    up=Branch(
                                        up_proj=Linear(
                                            in_dim=dim,
                                            out_dim=ffn_hidden_dim,
                                            param_dtype=jnp.bfloat16,
                                        ),
                                        gate_proj=Chain(
                                            proj=Linear(
                                                in_dim=dim,
                                                out_dim=ffn_hidden_dim,
                                                param_dtype=jnp.bfloat16,
                                            ),
                                            activation=jax.nn.silu,
                                        ),
                                        reduce=lambda x: x["up_proj"] * x["gate_proj"],
                                    ),
                                    down=Linear(
                                        in_dim=ffn_hidden_dim,
                                        out_dim=dim,
                                        param_dtype=jnp.bfloat16,
                                    ),
                                )
                            ),
                        ),
                        timescale=Select(key="timescale"),
                        position=Select(key="position"),
                    ),
                ),
            )

        case _:
            raise ValueError(f"Unknown model: {model}")
