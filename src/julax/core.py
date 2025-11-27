from abc import ABC, abstractmethod
import plum
from functools import partial
from typing import Annotated, Callable, TypeAlias, Any

import optax
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError


#####

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

#####

from julax.base import PRNG, OutShardingType, PyTree, dispatch

# TODO: use RootModel[dict] for better customization
# Or maybe SimpleNamespace?
Param: TypeAlias = dict
State: TypeAlias = dict

#####


class LayerBase(BaseModel, ABC):
    param_dtype: jnp.dtype | None = None
    param_sharding: OutShardingType = None
    out_sharding: OutShardingType = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        ignored_types=(
            jax.stages.Wrapped,
            plum.function.Function,
            optax.GradientTransformation,
        ),
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # TODO: respect `FieldInfo`
        jax.tree_util.register_dataclass(
            cls, data_fields=list(cls.model_fields.keys()), meta_fields=[]
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

    def param(self, rng: PRNG) -> Param:
        return Param()

    def state(self, rng: PRNG) -> State:
        return State()

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

        return sublayer_params | layer_params, sublayer_states | layer_states

    @abstractmethod
    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]: ...

    @dispatch
    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        # TODO: make sure treedef of the returned state is not changed compared to `s`?
        return self.forward(x, p, s)

    @dispatch
    def __call__(self, x: PyTree) -> tuple[PyTree, State]:
        return self.__call__(x, *self.init())


@dispatch(precedence=1)
def to_layer(x: LayerBase):
    return x


@dispatch
def to_layer(x):
    raise ValidationError(f"Failed to convert to LayerBase: {x}")


LayerLike = Annotated[LayerBase, BeforeValidator(to_layer)]


class Learner(LayerBase):
    loss_fn: Callable[[PyTree, PyTree], Any]
    model: LayerBase
    agg: Callable = jnp.mean
    feature_name: str = "feature"
    label_name: str = "label"

    def forward(self, input: dict, p: Param, s: State) -> tuple[PyTree, State]:
        x = input[self.feature_name]
        y = input[self.label_name]
        ŷ, s["model"] = self.model(x, p["model"], s["model"])
        losses = self.loss_fn(ŷ, y)
        loss = self.agg(losses)
        return loss, s


class Trainer(LayerBase):
    learner: Learner
    optimizer: Any

    def state(self, rng: PRNG) -> State:
        return State(optimizer=None, step=0, loss=0.0)

    @dispatch
    def init(
        self, layer_params, layer_states, sublayer_params, sublayer_states
    ) -> tuple[Param, State]:
        layer_states["optimizer"] = self.optimizer.init(sublayer_params["learner"])
        return sublayer_params | layer_params, sublayer_states | layer_states

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        loss, state = self.learner(x, p["learner"], s["learner"])
        return loss, State(
            learner=state, optimizer=s["optimizer"], step=s["step"] + 1, loss=loss
        )

    @partial(jit, static_argnums=0)
    def forward_and_backward(
        self, x: PyTree, p: Param, s: State
    ) -> tuple[Param, State]:
        (_, S), grads = value_and_grad(self.forward, argnums=1, has_aux=True)(x, p, s)
        updates, S["optimizer"] = self.optimizer.update(grads, S["optimizer"])
        P = optax.apply_updates(p, updates)
        return P, S

    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[Param, State]:
        return self.forward_and_backward(x, p, s)
