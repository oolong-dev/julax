import abc
from copy import deepcopy
import os

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import grain
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, Array
from jax.nn.initializers import (
    Initializer,
    truncated_normal,
    variance_scaling,
    ones,
    zeros,
)
from typing import Annotated, Callable, TypeAlias, Any

PRNGKey: TypeAlias = Array
PyTree: TypeAlias = Any

import optax

import tensorflow_datasets as tfds
import tensorflow as tf


from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

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
    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]: ...

    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.forward(x, p, s)


#####
# layers
#####


class Linear(LayerBase):
    in_dim: int
    out_dim: int
    w_init: Initializer
    b_init: None | Initializer = None

    def param(self, rng: PRNGKey) -> Param:
        rng_w, rng_b = jax.random.split(rng)
        return Param(
            w=self.w_init(rng_w, (self.in_dim, self.out_dim)),
            b=self.b_init(rng_b, (self.out_dim,)) if self.b_init else None,
        )

    def forward(self, x: Array, p: Param, st: State) -> tuple[Array, State]:
        o = jnp.einsum("...d,dh->...h", x, p["w"])
        if "b" in p:
            o += p["b"]
        return o, st


@dispatch(precedence=1)
def to_layer(x: LayerBase):
    return x


@dispatch
def to_layer(x):
    raise ValidationError(f"Failed to convert to LayerBase: {x}")


LayerLike = Annotated[LayerBase, BeforeValidator(to_layer)]


class Chain(LayerBase):
    layers: tuple[LayerLike, ...]

    def __init__(self, *args):
        super().__init__(layers=args)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        h = x
        S = ()
        for l, p, s in zip(self.layers, p["layers"], s["layers"]):
            h, sᵢ = l(h, p, s)
            S += (sᵢ,)
        return h, State(layers=S)


class F(LayerBase):
    f: Callable

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.f(x), s


@dispatch
def to_layer(x: Callable):
    return F(f=x)


class Dropout(LayerBase):
    rate: float

    def state(self, rng: PRNGKey) -> State:
        return State(rng=rng, is_training=True)

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        rng, _rng = jax.random.split(s["rng"])
        if s["is_training"] and self.rate > 0:
            mask = jax.random.bernoulli(rng, self.rate, x.shape)
            o = jnp.where(mask, 0, x) / (1 - self.rate)
        else:
            o = x
        return o, State(rng=_rng, is_training=s["is_training"])


def update_mode(s: State, key: str, val):
    return jax.tree.map_with_path(
        lambda path, x: (
            val if jax.tree_util.keystr([path[-1]], simple=True) == key else True
        ),
        s,
    )


def train_mode(s: State):
    return update_mode(s, "is_training", True)


def test_mode(s: State):
    return update_mode(s, "is_training", False)


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

    def param(self, rng: PRNGKey) -> Param:
        return Param(w=self.w_init(rng, (self.in_dim, self.out_dim)))

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        return p["w"][x], s


class LayerNorm(LayerBase):
    dim: int
    ϵ: float = 1e-5
    w_init: Initializer = ones
    b_init: Initializer = zeros

    def param(self, rng: PRNGKey) -> Param:
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
    checkpoint_manager: CheckpointManager
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


# #####
# Experiments
# #####


class ShakespeareChar(grain.sources.RandomAccessDataSource):

    def __init__(
        self,
        dir: str = "data",
        dataset: str = "shakespeare_char",
        split="train",
        block_size=256,
    ):
        self._block_size = block_size
        self._data = np.memmap(
            os.path.join(dir, dataset, f"{split}.bin"), dtype=np.uint16, mode="r"
        )

    def __getitem__(self, i):
        return {
            "feature": self._data[i : i + self._block_size],
            "label": self._data[i + 1 : i + self._block_size + 1],
        }

    def __len__(self):
        return len(self._data) - self._block_size


def causal_self_attention(
    w_init,
    b_init,
    n_embd=384,
    n_head=6,
):
    return Chain(
        Linear(in_dim=n_embd, out_dim=3 * n_embd, w_init=w_init, b_init=b_init),
        partial(jnp.dsplit, indices_or_sections=3),
        partial(map, lambda x: jnp.reshape(x, x.shape[:-1] + (n_head, -1))),
    )


# n_layer = 6
# n_head = 6
# n_embd = 384
# dropout = 0.2

# weight_decay = 1e-1
# learning_rate = 1e-3
# max_iters = 5000
# lr_decay_iters = 5000
# min_lr = 1e-4
# beta1 = 0.9
# beta2 = 0.99

# warmup_iters = 100


# def observer(): ...


# def create_attn_block(T, N, H):
#     D = N * H
#     return Chain(
#         Linear(in_dim=D, out_dim=3 * D, w_init=truncated_normal(stddev=0.02)),
#         lambda x: jnp.dsplit(x, 3),
#         Parallel(
#             n=3,
#             layer=Chain(Reshape(-1, T, N, H), jnp.matrix_transpose),
#             connection=partial(jax.nn.dot_product_attention, is_causal=True),
#         ),
#         Dropout(rate=dropout),
#         jnp.matrix_transpose,
#         Reshape(-1, T, D),
#         Linear(in_dim=D, out_dim=D, w_init=truncated_normal(stddev=0.02)),
#         Dropout(rate=dropout),
#     )


# exp = Experiment(
#     name="nanoGPT",
#     trainer=Trainer(
#         learner="",
#         optimizer=optax.chain(
#             optax.clip_by_global_norm(1.0),
#             optax.adamw(
#                 learning_rate=optax.schedules.warmup_cosine_decay_schedule(
#                     init_value=learning_rate / warmup_iters,
#                     peak_value=learning_rate,
#                     warmup_steps=warmup_iters,
#                     decay_steps=lr_decay_iters,
#                     end_value=min_lr,
#                 ),
#                 b1=beta1,
#                 b2=beta2,
#                 weight_decay=weight_decay,
#                 mask=lambda p: jax.tree.map(lambda x: x.ndim != 1, p),
#             ),
#         ),
#     ),
#     dataset_factory=lambda: dataset(),
# )
