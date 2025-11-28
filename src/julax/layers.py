from typing import Callable

import jax
from jax import Array
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from jax.nn.initializers import (
    Initializer,
    lecun_normal,
    ones,
    zeros,
    variance_scaling,
)

from .core import PRNG, LayerBase, LayerLike, PyTree, Param, State, dispatch


class F(LayerBase):
    f: Callable

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.f(x), s


@dispatch
def to_layer(x: Callable):
    return F(f=x)


class SkipConnection(LayerBase):
    layer: LayerLike
    connection: Callable = jnp.add

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        S = State()
        o, S["layer"] = self.layer(x, p["layer"], s["layer"])
        return self.connection(o, x), S


class Repeated(LayerBase):
    n: int
    layer: LayerLike

    def sublayers(self) -> dict:
        return {f"layer_{i}": self.layer for i in range(self.n)}

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        S = State()
        o = x
        for i in range(self.n):
            o, S[f"layer_{i}"] = self.layer(o, p[f"layer_{i}"], s[f"layer_{i}"])
        return o, S


class NamedLayers(LayerBase):
    names: tuple[str, ...]
    layers: tuple[LayerLike, ...]

    def __init__(self, *args, **kwargs):
        names = tuple(f"layer_{i}" for i in range(len(args))) + tuple(kwargs.keys())
        layers = tuple(args) + tuple(kwargs.values())
        super().__init__(names=names, layers=layers)

    def sublayers(self) -> dict:
        return {k: v for k, v in zip(self.names, self.layers)}


class Chain(NamedLayers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        h = x
        S = State()
        for name, layer in zip(self.names, self.layers):
            h, S[name] = layer(h, p[name], s[name])
        return h, S


class Branch(NamedLayers):
    """1 -> N"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        O = {}
        S = State()
        for name, layer in zip(self.names, self.layers):
            O[name], S[name] = layer(x, p[name], s[name])
        # ??? return dict?
        return tuple(O.values()), S


class Parallel(NamedLayers):
    """N -> N"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        assert len(x) == len(self.layers)
        O = {}
        S = State()
        for name, layer, xᵢ in zip(self.names, self.layers, x):
            O[name], S[name] = layer(xᵢ, p[name], s[name])
        # ??? return dict?
        return tuple(O.values()), S


#####


class Linear(LayerBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = lecun_normal()
    b_init: None | Initializer = zeros

    def param(self, rng: PRNG) -> Param:
        rng_w, rng_b = jax.random.split(rng)
        return Param(
            w=self.w_init(
                rng_w,
                (self.in_dim, self.out_dim),
                dtype=self.param_dtype,
                out_sharding=self.param_sharding,
            ),
            b=(
                self.b_init(
                    rng_b,
                    (self.out_dim,),
                    dtype=self.param_dtype,
                    out_sharding=(
                        None
                        if self.param_sharding is None
                        else P(self.param_sharding[-1])
                    ),
                )
                if self.b_init
                else None
            ),
        )

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        o = jnp.einsum("...d,dh->...h", x, p["w"], out_sharding=self.out_sharding)
        if p["b"] is not None:
            o += p["b"]
        return o, s


class Dropout(LayerBase):
    rate: float

    def state(self, rng: PRNG) -> State:
        return State(rng=rng, is_training=True)

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
    w_init: Initializer = variance_scaling(1.0, "fan_in", "normal", out_axis=0)

    def param(self, rng: PRNG) -> Param:
        return Param(
            w=self.w_init(
                rng,
                (self.in_dim, self.out_dim),
                dtype=self.param_dtype,
                out_sharding=self.param_sharding,
            )
        )

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        return p["w"].at[x].get(out_sharding=self.out_sharding), s


class Unembedding(Embedding):
    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        return jnp.einsum("bld,dn->bln", x, p["w"], out_sharding=self.out_sharding), s


class LayerNorm(LayerBase):
    dim: int
    epsilon: float = 1e-5
    w_init: Initializer = ones
    b_init: Initializer = zeros
    compute_dtype: jnp.dtype | None = None

    def param(self, rng: PRNG) -> Param:
        w_rng, b_rng = jax.random.split(rng)
        return Param(
            w=self.w_init(
                w_rng,
                (self.dim,),
                dtype=self.param_dtype,
                out_sharding=self.out_sharding,
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

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        x_std = jax.nn.standardize(
            x.astype(self.compute_dtype), epsilon=self.epsilon
        ).astype(self.param_dtype)
        return x_std * p["w"] + p["b"], s
