from typing import Any, Optional
from einops.layers import RearrangeMixin, ReduceMixin
from einops.layers._einmix import _EinmixMixin

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from pydantic import computed_field

from julax.base import FrozenDict, Param, PyTree, State, PRNG
from .base import LayerBase


class Reduce(LayerBase):
    pattern: str
    reduction: str
    sizes: FrozenDict

    def __init__(self, pattern: str, reduction: str, **kwargs):
        super().__init__(pattern=pattern, reduction=reduction, sizes=kwargs)

    def model_post_init(self, context: Any) -> None:
        self._reducer = ReduceMixin(self.pattern, self.reduction, **self.sizes)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self._reducer._apply_recipe(x), s


class Rearrange(LayerBase):
    pattern: str
    sizes: FrozenDict

    def __init__(self, pattern: str, **kwargs):
        super().__init__(pattern=pattern, sizes=kwargs)

    def model_post_init(self, context: Any) -> None:
        self._rearranger = RearrangeMixin(self.pattern, **self.sizes)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self._rearranger._apply_recipe(x), s


class _Einmix(_EinmixMixin):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self._w_shape = weight_shape
        self._b_shape = bias_shape

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern: Optional[str],
        pre_reshape_lengths: Optional[dict],
        post_reshape_pattern: Optional[str],
        post_reshape_lengths: Optional[dict],
    ):
        self._pre_rearrange = None
        if pre_reshape_pattern is not None:
            self._pre_rearrange = Rearrange(pre_reshape_pattern, **pre_reshape_lengths)

        self._post_rearrange = None
        if post_reshape_pattern is not None:
            self._post_rearrange = Rearrange(
                post_reshape_pattern, **post_reshape_lengths
            )


class EinMix(LayerBase):
    pattern: str
    w_shape: str
    w_init: Initializer
    b_shape: str | None
    b_init: Initializer | None
    sizes: FrozenDict

    @computed_field
    @property
    def einsum_pattern(self) -> str:
        return self._einmix.einsum_pattern

    def __init__(
        self,
        pattern: str,
        *,
        w_shape: str,
        w_init: Initializer,
        b_shape: str | None = None,
        b_init: Initializer | None = None,
        **kwargs,
    ):
        super().__init__(
            pattern=pattern,
            w_shape=w_shape,
            w_init=w_init,
            b_shape=b_shape,
            b_init=b_init,
            sizes=kwargs,
        )

    def model_post_init(self, context: Any) -> None:
        self._einmix = _Einmix(
            pattern=self.pattern,
            weight_shape=self.w_shape,
            bias_shape=self.b_shape,
            **self.sizes,
        )

    def param(self, rng: PRNG) -> Param:
        if self.b_init and self._einmix._b_shape:
            rng_w, rng_b = jax.random.split(rng)
            return Param(
                w=self.w_init(rng_w, self._einmix._w_shape),
                b=self.b_init(rng_b, self._einmix._b_shape),
            )
        else:
            return Param(w=self.w_init(rng, self._einmix._w_shape))

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        if self._einmix._pre_rearrange is not None:
            x, _ = self._einmix._pre_rearrange(x, p, s)

        o = jnp.einsum(self._einmix.einsum_pattern, x, p["w"])

        if "b" in p:
            o += p["b"]

        if self._einmix._post_rearrange is not None:
            o, _ = self._einmix._post_rearrange(o, p, s)

        return o, s
