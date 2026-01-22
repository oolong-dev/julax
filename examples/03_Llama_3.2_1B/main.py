import os
import pickle
import optax
from safetensors import safe_open

import grain
import jax
import jax.numpy as jnp
import numpy as np
from grain._src.core.sharding import ShardByJaxProcess, even_split
from grain.experimental import FlatMapIterDataset, FlatMapTransform, ParquetIterDataset
from jax import Array

from julax.base import Dtype, Param, State, PRNG
from julax.experiment.experiment import Experiment
from julax.layers import (
    LayerBase,
    Learner,
    Trainer,
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
from julax.experiment.observers import default_observer
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


def create_transformer(
    batch_size=1,
    seq_len=10,
    dim=2048,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=64,
    ffn_hidden_dim=8192,
    vocab_size=128256,
    cache_size=None,
) -> Transformer:
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
                                                B=batch_size,
                                                T=seq_len,
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
                                                B=batch_size,
                                                S=seq_len,
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
                                                B=batch_size,
                                                S=seq_len,
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
                                B=batch_size,
                                T=seq_len,
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


def from_hf(p, s, model_path=None):
    # Allow overriding the default model path via argument or environment variable.
    if model_path is None:
        model_path = os.getenv(
            "LLAMA_MODEL_PATH", "models/Llama-3.2-1B-Instruct/model.safetensors"
        )

    tensors = {}
    with safe_open(model_path, framework="flax", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    w_ln1 = []
    w_q = []
    w_k = []
    w_v = []
    w_o = []
    w_ln2 = []
    w_up = []
    w_gate = []
    w_down = []

    for i in range(16):
        w_ln1.append(tensors[f"model.layers.{i}.input_layernorm.weight"])
        w_q.append(tensors[f"model.layers.{i}.self_attn.q_proj.weight"].T)
        w_k.append(tensors[f"model.layers.{i}.self_attn.k_proj.weight"].T)
        w_v.append(tensors[f"model.layers.{i}.self_attn.v_proj.weight"].T)
        w_o.append(tensors[f"model.layers.{i}.self_attn.o_proj.weight"].T)

        w_ln2.append(tensors[f"model.layers.{i}.post_attention_layernorm.weight"])
        w_up.append(tensors[f"model.layers.{i}.mlp.up_proj.weight"].T)
        w_gate.append(tensors[f"model.layers.{i}.mlp.gate_proj.weight"].T)
        w_down.append(tensors[f"model.layers.{i}.mlp.down_proj.weight"].T)

    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["norm"]["scale"] = (
        jnp.stack(w_ln1, axis=0)
    )
    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["qkv_proj"]["q"]["#0"][
        "w"
    ] = jnp.stack(w_q, axis=0)
    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["qkv_proj"]["k"]["#0"][
        "w"
    ] = jnp.stack(w_k, axis=0)
    p["blocks"]["hidden"]["attn"]["processor"]["#0"]["hidden"]["qkv_proj"]["v"]["#0"][
        "w"
    ] = jnp.stack(w_v, axis=0)
    p["blocks"]["hidden"]["attn"]["processor"]["#2"]["w"] = jnp.stack(w_o, axis=0)

    p["blocks"]["hidden"]["ffn"]["processor"]["norm"]["scale"] = jnp.stack(
        w_ln2, axis=0
    )
    p["blocks"]["hidden"]["ffn"]["processor"]["up"]["up_proj"]["w"] = jnp.stack(
        w_up, axis=0
    )
    p["blocks"]["hidden"]["ffn"]["processor"]["up"]["gate_proj"]["proj"]["w"] = (
        jnp.stack(w_gate, axis=0)
    )
    p["blocks"]["hidden"]["ffn"]["processor"]["down"]["w"] = jnp.stack(w_down, axis=0)

    p["emb"]["w"] = tensors["model.embed_tokens.weight"]
    p["out_norm"]["scale"] = tensors["model.norm.weight"]

    return p, s


def verify(max_seq_len=30, model_path=None):
    tokens = [128000, 791, 6367, 311, 28915, 264, 1695, 19692, 374, 220]
    input_ids = jnp.array([tokens])
    m_prefill = create_transformer(seq_len=input_ids.shape[1], cache_size=max_seq_len)
    m_decode = create_transformer(seq_len=1, cache_size=max_seq_len)
    p, s = from_hf(*m_prefill.init(), model_path=model_path)
    o, s_cached = m_prefill(
        {
            "token_ids": input_ids,
            "position": jnp.arange(input_ids.shape[1]).reshape(1, -1),
        },
        p,
        s,
    )

    for _ in range(max_seq_len - 1 - input_ids.shape[1]):
        new_token = int(o[0][-1].argmax())
        position = jnp.array([[len(tokens)]])
        tokens.append(new_token)

        input_ids = jnp.array([[new_token]])

        o, s_cached = m_decode(
            {"token_ids": input_ids, "position": position}, p, s_cached
        )
    print("Generated tokens:", tokens)


def create_experiment():
    return Experiment(
        name="llama_3.2_1b",
        max_steps=1000,
        trainer=Trainer(
            learner=Learner(
                feature_name="inputs",
                label_name="target_labels",
                model=create_transformer(),
                loss_fn=optax.softmax_cross_entropy_with_integer_labels,
            ),
            optimizer=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(
                    b1=0.9,
                    b2=0.95,
                    eps=1e-8,
                ),
                optax.add_decayed_weights(0.1),
                optax.scale_by_schedule(
                    optax.warmup_cosine_decay_schedule(
                        init_value=0.0,
                        peak_value=0.0005,
                        warmup_steps=2_000,
                        decay_steps=30_000,
                        end_value=5.0e-05,
                    )
                ),
                optax.scale(-1.0),
            ),
        ),
        dataset=create_dataset(
            batch_size=4,
            seq_len=4096,
            data_dir="data/wikipedia_parquet",
            tokenizer_path="models/llama_tokenizer.pkl",
        ),
        observer=default_observer(),
    )
