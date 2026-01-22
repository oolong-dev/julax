import jax
import jax.numpy as jnp

from typing import Callable, Annotated
from julax.base import PRNG, PyTree, Param, State
from julax.utils import identity
from pydantic import Field

from .base import LayerBase, LayerLike, dispatch


class F(LayerBase):
    f: Callable

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.f(x), s


@dispatch
def to_layer(x: Callable):
    return F(f=x)


# TODO: generalize to select subtree
class Select(LayerBase):
    key: int | str

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        match self.key:
            case int(k):
                return x[k], s
            case str(k) if k.startswith("."):
                return getattr(x, k), s
            case str(k):
                return x[k], s
            case _:
                raise ValueError(f"Unsupported key type: {type(self.key)}")


class Repeat(LayerBase):
    n: int
    layer: LayerLike

    def sublayers(self) -> dict:
        return {f"#{i}": self.layer for i in range(self.n)}

    @dispatch
    def init(self, rng: PRNG) -> tuple[Param, State]:
        def scan_init(carry, rng):
            p, s = self.layer.init(rng)
            return carry, (p, s)

        rngs = jax.random.split(rng, self.n)
        _, (P, S) = jax.lax.scan(scan_init, None, rngs)
        return P, S

    def __getitem__(self, key) -> LayerBase:
        return self.layer

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        def scan_forward(x, ps):
            return self.layer(x, *ps)

        o, s = jax.lax.scan(scan_forward, x, (p, s))

        return o, s


class NamedLayers(LayerBase):
    names: Annotated[tuple[str, ...], Field(repr=False)]
    layers: Annotated[tuple[LayerLike, ...], Field(repr=False)]

    def sublayers(self) -> dict:
        return {k: v for k, v in zip(self.names, self.layers)}


class Chain(NamedLayers):
    def __init__(self, *args, **kwargs):
        names = tuple(f"#{i}" for i in range(len(args))) + tuple(kwargs.keys())
        layers = tuple(args) + tuple(kwargs.values())
        super().__init__(names=names, layers=layers)

    def __getitem__(self, key: str | int) -> LayerBase:
        if isinstance(key, int):
            return self.layers[key]
        else:
            return self.sublayers()[key]

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        h = x
        S = State()
        for name, layer in zip(self.names, self.layers):
            h, S[name] = layer(h, p[name], s[name])
        return h, S


class Branch(Chain):
    """1 -> N"""

    # place holder to bypass lint check
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        O = {}
        S = State()
        reduce = None
        for name, layer in zip(self.names, self.layers):
            if "reduce" == name:
                reduce = layer
                continue
            O[name], S[name] = layer(x, p[name], s[name])

        if reduce is None:
            return O, S
        else:
            O, S["reduce"] = reduce(O, p["reduce"], s["reduce"])
            return O, S


class Residual(Branch):
    def __init__(self, processor, *, skip_through=identity, reduce: Callable = jnp.add):
        super().__init__(
            processor=processor,
            skip_through=skip_through,
            reduce=lambda x: reduce(x["processor"], x["skip_through"]),
        )


class Parallel(Branch):
    """N -> N

    !!! WARNING
        JAX's JIT compilation reconstructs dictionaries with sorted keys. Do not
        rely on the order of keys in the input dictionary.
    """

    # place holder to bypass lint check
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        reduce = None
        if "reduce" in self.names:
            assert len(x) == len(self.layers) - 1, (
                "Number of inputs must match number of layers."
            )
            assert self.names.index("reduce") == len(self.layers) - 1, (
                "`reduce` layer must be the last layer."
            )
            names = self.names[:-1]
            layers = self.layers[:-1]
            reduce = self.layers[-1]
        else:
            names = self.names
            layers = self.layers
            assert len(x) == len(self.layers), (
                "Number of inputs must match number of layers."
            )

        if isinstance(x, dict):
            inputs = [x[name] for name in names]
        elif isinstance(x, (list, tuple)):
            inputs = x
        else:
            raise ValueError("Input to Parallel must be a dict, list, or tuple.")

        O = {}
        S = State()
        for name, layer, xᵢ in zip(names, layers, inputs, strict=True):
            O[name], S[name] = layer(xᵢ, p[name], s[name])

        if reduce is None:
            return O, S
        else:
            O, S["reduce"] = reduce(O, p["reduce"], s["reduce"])
            return O, S
