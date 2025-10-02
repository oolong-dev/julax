import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, Array
from jax.nn.initializers import Initializer, truncated_normal
from jax.tree_util import (
    register_dataclass,
    register_pytree_with_keys_class,
    GetAttrKey,
)
from typing import Iterable
from typing import Callable, TypeAlias, Any

PRNGKey: TypeAlias = Array
PyTree: TypeAlias = Any

from jsonargparse import auto_cli, auto_parser

# from jsonargparse._common import not_subclass_type_selectors

# # not_subclass_type_selectors.pop("dataclass")
# not_subclass_type_selectors.pop("pydantic")

import optax

import tensorflow_datasets as tfds
import tensorflow as tf

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict

from functools import partial

import plum

dispatch = plum.Dispatcher(warn_redefinition=True)

Param: TypeAlias = dict
State: TypeAlias = dict


class LayerBase(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        ignored_types=(jax.stages.Wrapped, plum.function.Function),
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # TODO: respect `FieldInfo`
        jax.tree_util.register_dataclass(
            cls, data_fields=list(cls.model_fields.keys()), meta_fields=[]
        )

    def sublayers(self) -> dict:
        attrs = {f: getattr(self, f) for f in self.model_fields_set}
        attrs_flatten, treedef = jax.tree.flatten(
            attrs,
            is_leaf=lambda x: isinstance(x, LayerBase),
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

    def param(self, rng: PRNGKey) -> Param:
        return Param()

    def state(self, rng: PRNGKey) -> State:
        return State()

    @dispatch
    def init(self, seed: int) -> tuple[Param, State]:
        return self.init(jax.random.key(seed))

    @dispatch
    def init(self, rng: PRNGKey) -> tuple[Param, State]:
        sublayers, treedef = jax.tree.flatten(
            self.sublayers(), is_leaf=lambda x: isinstance(x, LayerBase)
        )

        sublayer_params_flatten, sublayer_stats_flatten = [], []

        for l in sublayers:
            if l is None:
                sublayer_params_flatten.append(None)
                sublayer_stats_flatten.append(None)
            else:
                rng, _rng = jax.random.split(rng)
                p, s = l.init(_rng)
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

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        raise NotImplementedError

    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.forward(x, p, s)


#####


class Dense(LayerBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = truncated_normal()
    b_init: Initializer = truncated_normal()
    activation: Callable | None = None

    def param(self, rng: PRNGKey) -> Param:
        rng_w, rng_b = jax.random.split(rng)
        p = Param(
            w=self.w_init(rng_w, (self.in_dim, self.out_dim)),
            b=self.b_init(rng_b, (self.out_dim,)),
        )
        return p

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        h = jnp.einsum("...d,dh->...h", x, p["w"])
        o = h + p["b"]
        if self.activation:
            o = self.activation(o)
        return o, s


class Chain(LayerBase):
    layers: tuple[LayerBase, ...]

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        h = x
        S = ()
        for l, p, s in zip(self.layers, p["layers"], s["layers"]):
            h, sᵢ = l(h, p, s)
            S += (sᵢ,)
        return h, State(layers=S)


class Learner(LayerBase):
    loss_fn: Callable
    model: LayerBase
    agg: Callable = jnp.mean
    feature_name: str = "feature"
    label_name: str = "label"

    def forward(self, input: dict, p: Param, s: State) -> tuple[PyTree, State]:
        x = input[self.feature_name]
        y = input[self.label_name]
        ŷ, S = self.model(x, p["model"], s["model"])
        losses = self.loss_fn(ŷ, y)
        l = self.agg(losses)
        return l, State(model=S)


class Trainer(LayerBase):

    learner: Learner
    optimizer: Any

    def state(self, rng: PRNGKey) -> State:
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
    def forward_and_backward(self, x, p, s):
        (loss, S), grads = value_and_grad(self.forward, argnums=1, has_aux=True)(
            x, p, s
        )
        updates, S["optimizer"] = self.optimizer.update(grads, S["optimizer"])
        P = optax.apply_updates(p, updates)
        return P, S

    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[Param, State]:
        return self.forward_and_backward(x, p, s)


class CheckpointManager:

    def load(self) -> tuple[Param, State] | tuple[None, None]:
        return None, None

    def save(self, model: LayerBase, p: Param, s: State):
        pass


class Experiment(LayerBase):
    name: str = "mnist"

    seed: int = 0
    checkpoint_manager: CheckpointManager = CheckpointManager()

    trainer: Trainer
    dataset_factory: Callable[[], Iterable]

    observer: Callable

    def state(self, rng: PRNGKey) -> State:
        return State(input=self.dataset_factory())

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        P, S = self.trainer(x, p["trainer"], s["trainer"])
        return Param(trainer=P), State(trainer=S, input=s["input"])

    def run(self):
        p, s = self.checkpoint_manager.load()
        if (p, s) == (None, None):
            p, s = self.init(self.seed)

        self.observer(self, p, s)

        for x in s["input"]:
            p, s = self(x, p, s)

            self.checkpoint_manager.save(self, p, s)
            self.observer(self, p, s)

        return p, s


def observer(x: Experiment, p: Param, s: State):
    if s["trainer"]["step"] % 100 == 0:
        dataset = (
            tfds.load("mnist", split="test")
            .batch(32, drop_remainder=True)
            .map(
                lambda x: {
                    "feature": tf.reshape(x["image"], (32, -1)),
                    "label": x["label"],
                }
            )
            .take(1000)
            .as_numpy_iterator()
        )
        model = x.trainer.learner.model
        param = p["trainer"]["learner"]["model"]
        state = s["trainer"]["learner"]["model"]  # TODO: convert to test mode
        n_correct, n_total = 0, 0
        for batch in dataset:
            ŷ, _ = model(batch["feature"], param, state)
            n_correct += (ŷ.argmax(axis=1) == batch["label"]).sum().item()
            n_total += 32
        acc = n_correct / n_total

        logging.info(f"Accuracy at step {s['trainer']['step']}: {acc}")


def dataset_factory():
    return (
        tfds.load("mnist", split="train")
        .repeat()
        .shuffle(1024, seed=123)
        .batch(32, drop_remainder=True)
        .map(
            lambda x: {
                "feature": tf.reshape(x["image"], (32, -1)),
                "label": x["label"],
            }
        )
        .take(1000)
        .as_numpy_iterator()
    )


X = Experiment(
    name="mnist",
    trainer=Trainer(
        learner=Learner(
            model=Chain(
                layers=(
                    Dense(in_dim=784, out_dim=512, activation=jax.nn.relu),
                    Dense(in_dim=512, out_dim=512, activation=jax.nn.relu),
                    Dense(in_dim=512, out_dim=10),
                )
            ),
            loss_fn=optax.softmax_cross_entropy_with_integer_labels,
        ),
        optimizer=optax.sgd(0.01),
    ),
    dataset_factory=dataset_factory,
    observer=observer,
)

def run(X: Experiment = X):
    p, s = X.checkpoint_manager.load()
    if (p, s) == (None, None):
        p, s = X.init(X.seed)

    X.observer(X, p, s)

    for x in s["input"]:
        p, s = X(x, p, s)

        X.checkpoint_manager.save(X, p, s)
        X.observer(X, p, s)

    return p, s


# if __name__ == "__main__":
#     auto_cli(run)
