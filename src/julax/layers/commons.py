import jax
import jax.numpy as jnp
from jax import Array
from jax.nn.initializers import Initializer, lecun_normal, ones, variance_scaling, zeros
from jax.sharding import PartitionSpec as P

from julax.base import Dtype, OutShardingType

from julax.base import PRNG, Param, State
from .base import LayerBase


class Linear(LayerBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = lecun_normal()
    b_init: Initializer | None = None

    param_dtype: Dtype | None = None
    param_sharding: OutShardingType = None
    out_sharding: OutShardingType = None

    def param(self, rng: PRNG) -> Param:
        p = Param()
        rng_w, rng_b = jax.random.split(rng)
        p["w"] = self.w_init(
            rng_w,
            (self.in_dim, self.out_dim),
            dtype=self.param_dtype,
            out_sharding=self.param_sharding,
        )
        if self.b_init:
            p["b"] = self.b_init(
                rng_b,
                (self.out_dim,),
                dtype=self.param_dtype,
                out_sharding=(
                    None if self.param_sharding is None else P(self.param_sharding[-1])
                ),
            )
        return p

    def param_length(self) -> int:
        return self.in_dim * self.out_dim + (self.out_dim if self.b_init else 0)

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        o = jnp.einsum("...d,dh->...h", x, p["w"], out_sharding=self.out_sharding)
        if self.b_init is not None:
            o += p["b"]
        return o, s


class Dropout(LayerBase):
    rate: float

    def state(self, rng: PRNG) -> State:
        return State(rng=rng, is_training=True)

    def state_length(self) -> int:
        return 4  # typically 32 bits?

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        rng, s["rng"] = jax.random.split(s["rng"])
        if s["is_training"] and self.rate > 0:
            mask = jax.random.bernoulli(rng, self.rate, x.shape)
            o = jnp.where(mask, 0, x) / (1 - self.rate)
        else:
            o = x
        return o, s


def _update_mode(s: State, key: str, val):
    return jax.tree.map_with_path(
        lambda path, x: (
            val if jax.tree_util.keystr(path[-1:], simple=True) == key else True
        ),
        s,
    )


def train_mode(s: State):
    return _update_mode(s, "is_training", True)


def test_mode(s: State):
    return _update_mode(s, "is_training", False)


#####


class Embedding(LayerBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = variance_scaling(1.0, "fan_out", "normal")

    param_dtype: Dtype | None = None
    param_sharding: OutShardingType = None
    out_sharding: OutShardingType = None

    def param(self, rng: PRNG) -> Param:
        return Param(
            w=self.w_init(
                rng,
                (self.in_dim, self.out_dim),
                dtype=self.param_dtype,
                out_sharding=self.param_sharding,
            )
        )

    def param_length(self) -> int:
        return self.in_dim * self.out_dim

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        return p["w"].at[x].get(out_sharding=self.out_sharding), s

    def attend(self, x: Array, p: Param) -> Array:
        return jnp.einsum("...ld,nd->...ln", x, p["w"], out_sharding=self.out_sharding)


class RotaryEmbedding(LayerBase):
    """Rotary Position Embedding."""

    # Adapted from https://github.com/AI-Hypercomputer/maxtext/blob/9204d6bbbf8bb19a05ebed72a55cfec687e0e044/src/MaxText/layers/embeddings.py#L271C11-L356C17
    embedding_dims: int
    min_timescale: int = 1
    max_timescale: int = 10000
    cast_as_fprop_dtype: bool = True
    fprop_dtype: Dtype = jnp.bfloat16
    rope_linear_scaling_factor: float = 1.0

    def state(self, rng: PRNG) -> State:
        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
        timescale = (
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        )
        if self.rope_linear_scaling_factor != 1.0:
            timescale = timescale * self.rope_linear_scaling_factor
        return State(timescale=timescale)

    def state_length(self) -> int:
        return self.embedding_dims // 2

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        seq_length = x.shape[1]
        position = jnp.arange(seq_length, dtype=jnp.float32)[
            jnp.newaxis, :, jnp.newaxis, jnp.newaxis
        ]
        sinusoid_inp = position / s["timescale"]
        sin = jnp.sin(sinusoid_inp).astype(x.dtype)
        cos = jnp.cos(sinusoid_inp).astype(x.dtype)
        first_half, second_half = jnp.split(x, 2, axis=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        if self.cast_as_fprop_dtype:
            first_part = first_part.astype(self.fprop_dtype)
            second_part = second_part.astype(self.fprop_dtype)
        x_out = jnp.concatenate((first_part, second_part), axis=-1)
        return x_out, s


class LayerNorm(LayerBase):
    dim: int
    epsilon: float = 1e-5
    w_init: Initializer = ones
    b_init: Initializer = zeros
    compute_dtype: Dtype | None = None

    param_dtype: Dtype | None = None
    param_sharding: OutShardingType = None
    out_sharding: OutShardingType = None

    def param(self, rng: PRNG) -> Param:
        w_rng, b_rng = jax.random.split(rng)
        return Param(
            w=self.w_init(
                w_rng,
                (self.dim,),
                dtype=self.param_dtype,
                out_sharding=self.param_sharding,
            ),
            b=self.b_init(
                b_rng,
                (self.dim,),
                dtype=self.param_dtype,
                out_sharding=(
                    None if self.param_sharding is None else P(self.param_sharding[-1])
                ),
            ),
        )

    def param_length(self) -> int:
        return 2 * self.dim

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        x_std = jax.nn.standardize(
            x.astype(self.compute_dtype), epsilon=self.epsilon
        ).astype(self.param_dtype)
        o = x_std * p["w"] + p["b"]
        if self.out_sharding is not None:
            o = jax.lax.with_sharding_constraint(o, self.out_sharding)
        return o, s


class RMSNorm(LayerBase):
    dim: int
    eps: float = 1e-8
    zero_center: bool = False
    scale_init: Initializer | None = ones
    scale_dtype: Dtype | None = None
    scale_sharding: OutShardingType = None

    dtype: Dtype = jnp.float32
    param_sharding: OutShardingType = None
    out_sharding: OutShardingType = None

    def param(self, rng: PRNG) -> Param:
        if self.scale_init is None:
            return Param()
        else:
            return Param(
                scale=self.scale_init(
                    rng,
                    (self.dim,),
                    dtype=self.scale_dtype,
                    out_sharding=(
                        None
                        if self.param_sharding is None
                        else P(self.param_sharding[-1])
                    ),
                )
            )

    def param_length(self) -> int:
        if self.scale_init is None:
            return 0
        else:
            return self.dim

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        x_dtype = x.dtype

        x = x.astype(self.dtype)
        rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)

        if self.zero_center:
            x = x - x.mean(axis=-1, keepdims=True)

        o = x * rms

        if self.scale_init is not None:
            o = o * p["scale"]

        o = o.astype(x_dtype)

        if self.out_sharding is not None:
            o = jax.lax.with_sharding_constraint(o, self.out_sharding)
        return o, s
