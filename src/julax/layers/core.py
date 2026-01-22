from functools import partial
from jax import jit, value_and_grad
from julax.base import PRNG, PyTree, dispatch
from .base import Param, State, LayerBase
from typing import Callable, Any
import jax.numpy as jnp
import optax


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
        return State(optimizer=None, loss=0.0)

    @dispatch
    def init(
        self, layer_params, layer_states, sublayer_params, sublayer_states
    ) -> tuple[Param, State]:
        layer_states["optimizer"] = self.optimizer.init(sublayer_params["learner"])
        return (
            sublayer_params | layer_params,
            sublayer_states | layer_states,
        )

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        loss, state = self.learner(x, p["learner"], s["learner"])
        return loss, State(learner=state, optimizer=s["optimizer"], loss=loss)

    @partial(jit, static_argnums=0, donate_argnames=("p", "s"))
    def forward_and_backward(
        self, x: PyTree, p: Param, s: State
    ) -> tuple[Param, State]:
        (_, S), grads = value_and_grad(self.forward, argnums=1, has_aux=True)(x, p, s)
        updates, S["optimizer"] = self.optimizer.update(
            grads["learner"], S["optimizer"]
        )
        P = optax.apply_updates(p["learner"], updates)
        return {"learner": P}, S

    @dispatch
    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[Param, State]:
        return self.forward_and_backward(x, p, s)
