import abc
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import grain
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, Array
from jax.nn.initializers import Initializer, truncated_normal
from typing import Callable, TypeAlias, Any

PRNGKey: TypeAlias = Array
PyTree: TypeAlias = Any

import optax

import tensorflow_datasets as tfds
import tensorflow as tf


from pydantic import BaseModel, ConfigDict

from functools import partial

from datetime import datetime


import plum

dispatch = plum.Dispatcher(warn_redefinition=True)

Param: TypeAlias = dict
State: TypeAlias = dict

import orbax.checkpoint as ocp


class LayerBase(BaseModel, abc.ABC):
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

    def param(self, rng: PRNGKey) -> Param:
        return Param()

    def state(self, rng: PRNGKey) -> State:
        return State()

    @dispatch
    def init(self, seed: int = 0) -> tuple[Param, State]:
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

    @abc.abstractmethod
    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        ...

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

    def __init__(self, manager: ocp.CheckpointManager) -> None:
        self._manager = manager

    def save(self, p: Param, s: State):
        self._manager.save(
            s["trainer"]["step"],
            args=ocp.args.Composite(
                param=ocp.args.PyTreeSave(item=p),
                state_trainer=ocp.args.PyTreeSave(item=s["trainer"]),
                state_dataset_iter=grain.checkpoint.CheckpointSave(item=s["input"]),
            ),
        )

    def load(self, p: Param, s: State) -> tuple[Param, State]:
        try:
            restored = self._manager.restore(
                step=None,
                args=ocp.args.Composite(
                    param=ocp.args.PyTreeRestore(
                        item=p,
                        restore_args=ocp.checkpoint_utils.construct_restore_args(p),
                    ),
                    state_trainer=ocp.args.PyTreeRestore(
                        item=s["trainer"],
                        restore_args=ocp.checkpoint_utils.construct_restore_args(
                            s["trainer"]
                        ),
                    ),
                    state_dataset_iter=grain.checkpoint.CheckpointRestore(
                        item=s["input"]
                    ),
                ),
            )
            param = restored["param"]
            state_trainer = restored["state_trainer"]
            state_dataset_iter = restored["state_dataset_iter"]
            return param, State(input=state_dataset_iter, trainer=state_trainer)
        except FileNotFoundError:
            return p, s


class Experiment(LayerBase):
    name: str = "mnist"

    seed: int = 0
    checkpoint_manager: CheckpointManager = CheckpointManager(
        ocp.CheckpointManager("./checkpoints")
    )

    trainer: Trainer
    dataset: grain.IterDataset

    observer: Callable

    def state(self, rng: PRNGKey) -> State:
        return State(input=iter(self.dataset))

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        P, S = self.trainer(x, p["trainer"], s["trainer"])
        return Param(trainer=P), State(trainer=S, input=s["input"])

    def run(self):
        p, s = self.init(self.seed)
        p, s = self.checkpoint_manager.load(p, s)
        self.observer(self, p, s)

        for x in s["input"]:
            p, s = self(x, p, s)

            self.checkpoint_manager.save(p, s)
            self.observer(self, p, s)

        return p, s


def observer(x: Experiment, p: Param, s: State):
    if s["trainer"]["step"] % 100 == 0:
        dataset = (
            grain.MapDataset.source(tfds.data_source("mnist", split="test"))
            .batch(32, drop_remainder=True)
            .map(
                lambda x: {
                    "feature": x["image"].reshape(32, -1),
                    "label": x["label"],
                }
            )
            .to_iter_dataset()
        )
        model = x.trainer.learner.model
        param = p["trainer"]["learner"]["model"]
        state = s["trainer"]["learner"]["model"]  # TODO: convert to test mode
        n_correct, n_total = 0, 0
        for batch in iter(dataset):
            ŷ, _ = model(batch["feature"], param, state)
            n_correct += (ŷ.argmax(axis=1) == batch["label"]).sum().item()
            n_total += 32
        acc = n_correct / n_total

        logger.info(f"Accuracy at step {s['trainer']['step']}: {acc}")


E = Experiment(
    name="mnist",
    checkpoint_manager=CheckpointManager(
        ocp.CheckpointManager(
            directory=os.path.join(
                os.getcwd(), "checkpoints", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
            options=ocp.CheckpointManagerOptions(save_interval_steps=100),
        )
    ),
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
    dataset=grain.MapDataset.source(tfds.data_source("mnist", split="train"))
    .seed(seed=45)
    .shuffle()
    .batch(32, drop_remainder=True)
    .map(
        lambda x: {
            "feature": x["image"].reshape(32, -1),
            "label": x["label"],
        }
    )
    .slice(slice(1000))
    .to_iter_dataset(),
    observer=observer,
)
