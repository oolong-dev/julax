from typing import Callable

from jax import Array
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, ones, zeros, variance_scaling

from .core import PRNG, LayerBase, LayerLike, PyTree, Param, State, dispatch


class F(LayerBase):
    f: Callable

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.f(x), s


@dispatch
def to_layer(x: Callable):
    return F(f=x)


class Chain(LayerBase):
    layers: tuple[LayerLike, ...]

    def __init__(self, *args):
        super().__init__(layers=args)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        h = x
        S = ()
        for layer, p, s in zip(self.layers, p["layers"], s["layers"]):
            h, sᵢ = layer(h, p, s)
            S += (sᵢ,)
        return h, State(layers=S)


#####


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
    w_init: Initializer = variance_scaling(
        2.0,
        "fan_out",
        "truncated_normal",
        in_axis=-2,
        out_axis=-1,
        batch_axis=(),
    )

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
