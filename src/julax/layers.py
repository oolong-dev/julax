from typing import Callable

from jax import Array
import jax
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
    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        h = x
        S = {}
        for name, layer in zip(self.names, self.layers):
            h, S[name] = layer(h, p[name], s[name])
        return h, State(**S)


class Branch(NamedLayers):
    """1 -> N"""

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        O = {}
        S = {}
        for name, layer in zip(self.names, self.layers):
            O[name], S[name] = layer(x, p[name], s[name])
        # ??? return dict?
        return tuple(O.values()), State(**S)


class Parallel(NamedLayers):
    """N -> N"""

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        assert len(x) == len(self.layers)
        O = {}
        S = {}
        for name, layer, xᵢ in zip(self.names, self.layers, x):
            O[name], S[name] = layer(xᵢ, p[name], s[name])
        # ??? return dict?
        return tuple(O.values()), State(**S)


#####


class Linear(LayerBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = lecun_normal()
    b_init: None | Initializer = zeros

    def param(self, rng: PRNG) -> Param:
        rng_w, rng_b = jax.random.split(rng)
        return Param(
            w=self.w_init(rng_w, (self.in_dim, self.out_dim)),
            b=self.b_init(rng_b, (self.out_dim,)) if self.b_init else None,
        )

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        o = jnp.einsum("...d,dh->...h", x, p["w"])
        if "b" in p:
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
        return Param(w=self.w_init(rng, (self.in_dim, self.out_dim)))

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        return p["w"][x], s


class LayerNorm(LayerBase):
    dim: int
    ϵ: float = 1e-5
    w_init: Initializer = ones
    b_init: Initializer = zeros

    def param(self, rng: PRNG) -> Param:
        w_rng, b_rng = jax.random.split(rng)
        return Param(
            w=self.w_init(w_rng, (self.dim,)), b=self.b_init(b_rng, (self.dim,))
        )

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        x_mean = x.mean(axis=-1, keepdims=True)
        x -= x_mean
        var = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.ϵ)
        # TODO: cast dtype
        return x * p["w"] + p["b"], s
