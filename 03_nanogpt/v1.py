#####
# Derived from ../02_mnist/v4.py
#####

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.nn.initializers import (
    Initializer,
    variance_scaling,
    truncated_normal,
    ones,
    zeros,
)
from jax.tree_util import PyTreeDef

import optax

import tensorflow_datasets as tfds
import tensorflow as tf

from jaxtyping import PRNGKeyArray, PyTree, Array, Num, Int
from typing import Any, Callable, Iterable, Optional

from plum import Dispatcher


dispatch = Dispatcher(warn_redefinition=True)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict

#####
# common
#####


# https://github.com/google-deepmind/penzai/blob/aac7808a3d1269ea9885094d78d2274d56fc7449/penzai/core/tree_util.py#L37C1-L59C36
def tree_flatten_exactly_one_level(
    tree: Any,
) -> Optional[tuple[list[tuple[Any, Any]], PyTreeDef]]:
    """Flattens a PyTree exactly one level, or returns None if it's not a PyTree.

    Args:
      tree: Tree to flatten.

    Returns:
      If ``tree`` has any children, returns a tuple ``(children, treedef)`` where
      children is a list of ``(key, child)`` pairs. Otherwise, returns ``None``.
    """
    paths_and_subtrees, treedef = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda subtree: subtree is not tree
    )
    if jax.tree_util.treedef_is_leaf(treedef):
        return None

    keys_and_subtrees = [(key, subtree) for ((key,), subtree) in paths_and_subtrees]
    return keys_and_subtrees, treedef


def post_walk(x, f, *, with_path=False, root_path=()):
    if flattened := tree_flatten_exactly_one_level(x):
        keys_and_subtrees, treedef = flattened
        vals = [
            post_walk(subtree, f, with_path=with_path, root_path=(*root_path, key))
            for key, subtree in keys_and_subtrees
        ]
        tree = jax.tree.unflatten(treedef, vals)
        if with_path:
            return f(root_path, tree)
        else:
            return f(tree)
    else:
        if with_path:
            return f(root_path, x)
        else:
            return f(x)


#####
# Visualization
#####

from rich.tree import Tree
from rich.panel import Panel
from rich.console import RenderableType, Group


@dispatch
def summary(x) -> str:
    return repr(x)


@dispatch
def summary(x: int | float) -> str:
    return f"[bold cyan]{x}[/bold cyan]"


@dispatch
def summary(x: Array) -> str:
    min_val = jnp.min(x)
    max_val = jnp.max(x)
    median_val = jnp.median(x)
    mean = jnp.mean(x)
    std = jnp.std(x)
    non_zero_count = jnp.count_nonzero(x)
    # number of elements
    num_elements = jnp.size(x)
    return f"≈{mean:5g} ±{std:5g} {median_val:5g} |≥{min_val:5g}, ≤{max_val:5g}| non_zero:{non_zero_count}/{num_elements}"


@dispatch
def typeof(x) -> str:
    return x.__class__.__name__


@dispatch
def typeof(x: Array) -> str:
    return f"jax.Array{{{x.dtype} {x.shape}}}"


def to_rich(x, k="🎯") -> RenderableType:
    t = typeof(x)
    ts = f"italic color({hash(type(x)) % 256})"
    label = f"[{ts} dim]<{t}>[/{ts} dim]"
    ks = f"color({hash(k) % 256})"
    label = f"[{ks} bold]{k}[/{ks} bold]: {label}"

    if flattened := tree_flatten_exactly_one_level(x):
        root = Tree(label, guide_style=f"dim {ks or ts}")
        for k, v in flattened[0]:
            root.add(to_rich(v, k))
        return root
    else:
        label = f"{label} [bright_yellow]=>[/bright_yellow] {summary(x)}"
        return Tree(label, guide_style=f"dim {ks or ts}")


#####
# Models
#####


class BaseConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, ignored_types=(jax.stages.Wrapped,)
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # TODO: respect `FieldInfo`
        jax.tree_util.register_dataclass(
            cls, data_fields=list(cls.model_fields.keys()), meta_fields=[]
        )

    def __rich__(self):
        return to_rich(self)


class ParamBase(BaseConfig): ...


NO_PARAM = ParamBase()


class StateBase(BaseConfig): ...


DEFAULT_STATE = StateBase()


class ModelBase(BaseConfig):
    def param(self, rng: PRNGKeyArray) -> ParamBase:
        return NO_PARAM

    def state(self, rng: PRNGKeyArray) -> StateBase:
        return DEFAULT_STATE

    def init(self, rng: PRNGKeyArray) -> tuple[PyTree, PyTree]:
        rng_ps, rng_st = jax.random.split(rng)
        return self.param(rng_ps), self.state(rng_st)

    def forward(self, ps: PyTree, x: PyTree, st: PyTree) -> tuple[PyTree, PyTree]:
        raise NotImplementedError

    def __call__(self, ps: PyTree, x: PyTree, st: PyTree) -> tuple[PyTree, PyTree]:
        return self.forward(ps, x, st)


#####


class Embedding(ModelBase):
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

    class EmbeddingParam(ParamBase):
        w: Num[Array, "i o"]

    def param(self, rng: PRNGKeyArray) -> EmbeddingParam:
        return self.EmbeddingParam(w=self.w_init(rng, (self.in_dim, self.out_dim)))

    def forward(
        self, ps: EmbeddingParam, x: Int[Array, "... i"], st: StateBase = DEFAULT_STATE
    ) -> tuple[Num[Array, "... i o"], StateBase]:
        o = ps.w[x]
        return o, st


class Dropout(ModelBase):
    rate: float

    class DropoutState(StateBase):
        rng: PRNGKeyArray
        is_training: bool = True

    def state(self, rng: PRNGKeyArray) -> DropoutState:
        return self.DropoutState(rng=rng)

    def forward(
        self, ps: ParamBase, x: Array, st: DropoutState
    ) -> tuple[Array, DropoutState]:
        rng, next_rng = jax.random.split(st.rng)
        if st.is_training and self.rate > 0:
            mask = jax.random.bernoulli(rng, self.rate, x.shape)
            o = jnp.where(mask, 0, x) / (1 - self.rate)
        else:
            o = x
        return o, self.DropoutState(rng=next_rng, is_training=st.is_training)


@dispatch
def test_mode(x):
    return x


@dispatch
def test_mode(x: Dropout.DropoutState):
    return Dropout.DropoutState(rng=x.rng, is_training=False)


class Dense(ModelBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = truncated_normal()
    b_init: Initializer = truncated_normal()
    activation: None | Callable = None

    class DenseParam(ParamBase):
        w: Num[Array, "d h"]
        b: Num[Array, "h"]

    def param(self, rng: PRNGKeyArray) -> DenseParam:
        rng_w, rng_b = jax.random.split(rng)
        return self.DenseParam(
            w=self.w_init(rng_w, (self.in_dim, self.out_dim)),
            b=self.b_init(rng_b, (self.out_dim,)),
        )

    def forward(
        self, ps: DenseParam, x: Num[Array, "... d"], st: StateBase
    ) -> tuple[Num[Array, "... h"], StateBase]:
        h = jnp.einsum("...d,dh->...h", x, ps.w)
        o = h + ps.b
        if self.activation:
            o = self.activation(o)
        return o, st


class LayerNorm(ModelBase):
    dim: int
    ϵ: float = 1e-5
    w_init: Initializer = ones
    b_init: Initializer = zeros

    class LayerNormParam(ParamBase):
        w: Array
        b: Array

    def param(self, rng: PRNGKeyArray) -> LayerNormParam:
        w_rng, b_rng = jax.random.split(rng)
        return self.LayerNormParam(
            w=self.w_init(w_rng, (self.dim,)), b=self.b_init(b_rng, (self.dim,))
        )

    def forward(self, ps: LayerNormParam, x: Array, st: StateBase) -> Array:
        x_mean = x.mean(axis=-1, keepdims=True)
        x -= x_mean
        var = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.ϵ)
        # TODO: cast dtype
        return x * ps.w + ps.b


class Chain(ModelBase):
    layers: tuple[ModelBase, ...]

    class ChainParam(ParamBase):
        layers: tuple[PyTree, ...]

    def param(self, rng: PRNGKeyArray) -> ChainParam:
        rngs = jax.random.split(rng, len(self.layers))
        return self.ChainParam(
            layers=tuple(layer.param(rng) for layer, rng in zip(self.layers, rngs))
        )

    class ChainState(StateBase):
        layers: tuple[StateBase, ...]

    def state(self, rng: PRNGKeyArray) -> ChainState:
        rngs = jax.random.split(rng, len(self.layers))
        return self.ChainState(
            layers=tuple(layer.state(rng) for layer, rng in zip(self.layers, rngs))
        )

    def forward(
        self, ps: ChainParam, x: PyTree, st: ChainState
    ) -> tuple[PyTree, ChainState]:
        h = x
        _st = ()
        for l, p, s in zip(self.layers, ps.layers, st.layers):
            h, _s = l(p, h, s)
            _st = (*_st, _s)
        return h, self.ChainState(layers=_st)


class Learner(ModelBase):
    model: ModelBase
    loss_fn: Callable
    agg: Callable = jnp.mean
    feature_name: str = "feature"
    label_name: str = "label"

    def param(self, rng: PRNGKeyArray) -> ParamBase:
        return self.model.param(rng)

    def state(self, rng: PRNGKeyArray) -> StateBase:
        return self.model.state(rng)

    def forward(
        self, ps: ParamBase, input: PyTree, st: StateBase
    ) -> tuple[PyTree, StateBase]:
        x = input[self.feature_name]
        y = input[self.label_name]
        ŷ, st = self.model(ps, x, st)
        losses = self.loss_fn(ŷ, y)
        l = self.agg(losses)
        return l, st


class Trainer(ModelBase):

    learner: Learner
    optimizer: Any

    def param(self, rng: PRNGKeyArray) -> ParamBase:
        return self.learner.param(rng)

    class TrainerState(StateBase):
        learner_state: StateBase
        step: int = 0
        opt_state: Any = None
        loss: float = 0.0

    def state(self, rng: PRNGKeyArray) -> TrainerState:
        return self.TrainerState(learner_state=self.learner.state(rng))

    def init(self, rng: PRNGKeyArray) -> tuple[ParamBase, TrainerState]:
        rng_ps, rng_st = jax.random.split(rng)
        ps = self.param(rng_ps)
        st = self.TrainerState(
            learner_state=self.learner.state(rng_st), opt_state=self.optimizer.init(ps)
        )
        return ps, st

    @partial(jit, static_argnums=0)
    def forward_and_backward(self, ps, x, ps_st, opt_st):
        (loss, ps_st), grads = value_and_grad(self.learner.forward, has_aux=True)(
            ps, x, ps_st
        )
        updates, opt_st = self.optimizer.update(grads, opt_st)
        ps = optax.apply_updates(ps, updates)
        return loss, ps, ps_st, opt_st

    def __call__(
        self, ps: ParamBase, x: PyTree, st: TrainerState
    ) -> tuple[PyTree, TrainerState]:
        loss, ps, ps_st, opt_st = self.forward_and_backward(
            ps, x, st.learner_state, st.opt_state
        )
        return ps, self.TrainerState(
            learner_state=ps_st,
            step=st.step + 1,
            opt_state=opt_st,
            loss=loss,
        )


class Experiment(BaseConfig):
    name: str

    seed: int = 0
    checkpointer: None = None

    trainer: Trainer
    dataset_factory: Callable

    observer: Callable

    def run(self):
        trainer_dataset = self.dataset_factory()
        param, state = self.trainer.init(jax.random.key(self.seed))
        if self.checkpointer:
            trainer_dataset, param, state = self.checkpointer.load(
                trainer_dataset, param, state
            )

        self.observer(self.trainer, param, state)

        for batch in trainer_dataset:
            param, state = self.trainer(param, batch, state)

            if self.checkpointer:
                self.checkpointer.save(trainer_dataset, param, state)

            self.observer(self.trainer, param, state)

        return param, state


#####
# Experiment related
#####

import os
import pickle
import numpy as np


def dataset(split="train", batch_size=64, block_size=256, seed=0, n_batches=5000):
    dataset_name = "shakespeare_char"
    data_dir = os.path.join("data", dataset_name)
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")

    rng = np.random.default_rng(seed)

    for _ in range(n_batches):
        ix = rng.choice(range(0, len(data) - block_size), (batch_size,))
        x = np.take(data, [range(i, i + block_size) for i in ix])
        y = np.take(data, [range(i + 1, i + 1 + block_size) for i in ix])

        yield {"feature": x, "label": y}


n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
