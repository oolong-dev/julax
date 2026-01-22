import jax
import plum
import optax

from abc import ABC, abstractmethod
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError
from typing import Annotated
from functools import partial

from julax.base import PRNG, PyTree, dispatch, Param, State

from jax import jit


class LayerBase(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        ignored_types=(
            jax.stages.Wrapped,
            plum.function.Function,
            optax.GradientTransformation,
        ),
    )

    def sublayers(self) -> dict:
        attrs_flatten, treedef = jax.tree.flatten(
            dict(self), is_leaf=lambda x: isinstance(x, LayerBase)
        )
        masked_sublayers = jax.tree.unflatten(
            treedef, [x if isinstance(x, LayerBase) else None for x in attrs_flatten]
        )

        res = {}
        for k, v in masked_sublayers.items():
            if jax.tree.reduce(
                lambda x, y: x or y,
                v,
                None,
                is_leaf=lambda x: isinstance(x, LayerBase),
            ):
                res[k] = v
        return res

    def __getitem__(self, key: str) -> "LayerBase":
        return self.sublayers()[key]

    def _ipython_display_(self):
        from .pprint import pprint

        pprint(self)

    def param(self, rng: PRNG) -> Param:
        return Param()

    def param_length(self) -> int:
        return 0

    def state(self, rng: PRNG) -> State:
        return State()

    def state_length(self) -> int:
        return 0

    def numel(self) -> tuple[int, int]:
        num_params = self.param_length()
        num_states = self.state_length()

        for sublayer in self.sublayers().values():
            p, s = sublayer.numel()
            num_params += p
            num_states += s

        return num_params, num_states

    @dispatch
    def init(self, seed: int = 0) -> tuple[Param, State]:
        return self.init(jax.random.key(seed))

    @dispatch
    def init(self, rng: PRNG) -> tuple[Param, State]:
        sublayers, treedef = jax.tree.flatten(
            self.sublayers(), is_leaf=lambda x: isinstance(x, LayerBase)
        )

        sublayer_params_flatten, sublayer_stats_flatten = [], []

        for layer in sublayers:
            if layer is None:
                sublayer_params_flatten.append(None)
                sublayer_stats_flatten.append(None)
            else:
                rng, _rng = jax.random.split(rng)
                p, s = layer.init(_rng)
                sublayer_params_flatten.append(p)
                sublayer_stats_flatten.append(s)

        sublayer_params = Param(**jax.tree.unflatten(treedef, sublayer_params_flatten))
        sublayer_states = State(**jax.tree.unflatten(treedef, sublayer_stats_flatten))

        rng_p, rng_s = jax.random.split(rng)
        layer_params = self.param(rng_p)
        layer_states = self.state(rng_s)
        return self.init(layer_params, layer_states, sublayer_params, sublayer_states)

    @dispatch
    def init(
        self, layer_params, layer_states, sublayer_params, sublayer_states
    ) -> tuple[Param, State]:
        assert len(layer_params.keys() & sublayer_params.keys()) == 0
        assert len(layer_states.keys() & sublayer_states.keys()) == 0

        return (
            sublayer_params | layer_params,
            sublayer_states | layer_states,
        )

    @abstractmethod
    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]: ...

    @partial(jit, static_argnums=0, donate_argnames=("p", "s"))
    def jit_forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.forward(x, p, s)

    @dispatch
    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        # TODO: make sure treedef of the returned state is not changed compared to `s`?
        return self.jit_forward(x, p, s)

    @dispatch
    def __call__(self, x: PyTree) -> tuple[PyTree, State]:
        return self.__call__(x, *self.init())


#####


@dispatch(precedence=1)
def to_layer(x: LayerBase):
    return x


@dispatch
def to_layer(x):
    raise ValidationError(f"Failed to convert to LayerBase: {x}")


LayerLike = Annotated[LayerBase, BeforeValidator(to_layer)]
