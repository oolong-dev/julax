#####
# Derived from ../02_mnist/v4.py
#####

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, Array
from jax.nn.initializers import (
    Initializer,
    variance_scaling,
    truncated_normal,
    ones,
    zeros,
)
from jax.tree_util import PyTreeDef

import optax

from typing import Any, Callable, Iterable, Mapping, Sequence, TypeAlias

PyTree: TypeAlias = Any
PRNGKey: TypeAlias = Array

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
) -> None | tuple[list[tuple[Any, Any]], PyTreeDef]:
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


Param = None | Array | Iterable["Param"] | Mapping[Any, "Param"]
State = Any


class ModelBase(BaseConfig):

    def param(self, rng: PRNGKey) -> Param:
        children = {
            f: getattr(self, f)
            for f in self.model_fields_set
            if isinstance(getattr(self, f), ModelBase)
        }
        rngs = jax.random.split(rng, len(children))
        return {f: c.param(r) for (f, c), r in zip(children.items(), rngs)}

    def state(self, rng: PRNGKey) -> State:
        children = {
            f: getattr(self, f)
            for f in self.model_fields_set
            if isinstance(getattr(self, f), ModelBase)
        }

        rngs = jax.random.split(rng, len(children))
        return {f: c.state(r) for (f, c), r in zip(children.items(), rngs)}

    def init(self, rng: PRNGKey) -> tuple[State, Param]:
        rng_ps, rng_st = jax.random.split(rng)
        return self.param(rng_ps), self.state(rng_st)

    def forward(self, ps: Param, x: PyTree, st: State) -> tuple[PyTree, State]:
        raise NotImplementedError

    def __call__(self, ps: Param, x: PyTree, st: State) -> tuple[PyTree, State]:
        return self.forward(ps, x, st)


#####


class F(ModelBase):
    f: Callable

    def forward(self, ps: Param, x: PyTree, st: State) -> tuple[PyTree, State]:
        return self.f(x), st


@dispatch
def to_model(x: Callable):
    return F(f=x)


@dispatch
def to_model(x: ModelBase):
    return x


class Reshape(ModelBase):
    shape: tuple[int, ...]

    def __init__(self, *shape):
        super().__init__(shape=shape)

    def forward(self, ps: Param, x: PyTree, st: State) -> tuple[PyTree, State]:
        return jnp.reshape(x, self.shape), st


class Parallel(ModelBase):
    n: int
    layer: ModelBase
    connection: Callable

    def param(self, rng: PRNGKey) -> tuple:
        return tuple(self.layer.param(rng) for rng in jax.random.split(rng, self.n))

    def state(self, rng: PRNGKey) -> tuple:
        return tuple(self.layer.state(rng) for rng in jax.random.split(rng, self.n))

    def forward(self, ps: tuple, xs: Sequence, st: tuple) -> tuple[PyTree, tuple]:
        assert self.n == len(xs), "Number of layers must match number of inputs"
        O, S = (), ()
        for p, x, s in zip(ps, xs, st):
            _o, _s = self.layer(p, x, s)
            O += (_o,)
            S += (_s,)

        return self.connection(*O), S


class Chain(ModelBase):
    layers: tuple[ModelBase, ...]

    def __init__(self, *layers):
        layers = tuple(to_model(x) for x in layers)
        super().__init__(layers=layers)

    def param(self, rng: PRNGKey) -> tuple:
        rngs = jax.random.split(rng, len(self.layers))
        return tuple(layer.param(rng) for layer, rng in zip(self.layers, rngs))

    def state(self, rng: PRNGKey) -> tuple:
        rngs = jax.random.split(rng, len(self.layers))
        return tuple(layer.state(rng) for layer, rng in zip(self.layers, rngs))

    def forward(self, ps: tuple, x: PyTree, st: tuple) -> tuple[PyTree, tuple]:
        h = x
        S = ()
        for l, p, s in zip(self.layers, ps, st):
            h, _s = l(p, h, s)
            S += (_s,)
        return h, S


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

    class EmbeddingParam(BaseConfig):
        w: Array

    def param(self, rng: PRNGKey) -> EmbeddingParam:
        return self.EmbeddingParam(w=self.w_init(rng, (self.in_dim, self.out_dim)))

    def forward(self, ps: EmbeddingParam, x: Array, st: None) -> tuple[Array, None]:
        o = ps.w[x]
        return o, st


class Dropout(ModelBase):
    rate: float

    class DropoutState(BaseConfig):
        rng: PRNGKey
        is_training: bool = True

    def state(self, rng: PRNGKey) -> DropoutState:
        return self.DropoutState(rng=rng)

    def forward(
        self, ps: None, x: Array, st: DropoutState
    ) -> tuple[Array, DropoutState]:
        rng, next_rng = jax.random.split(st.rng)
        if st.is_training and self.rate > 0:
            mask = jax.random.bernoulli(rng, self.rate, x.shape)
            o = jnp.where(mask, 0, x) / (1 - self.rate)
        else:
            o = x
        return o, self.DropoutState(rng=next_rng, is_training=st.is_training)


# TODO: generalize
def test_mode(x):
    return jax.tree.map(
        lambda s: (
            Dropout.DropoutState(rng=s.rng, is_training=False)
            if isinstance(s, Dropout.DropoutState)
            else s
        ),
        x,
        is_leaf=lambda s: isinstance(s, Dropout.DropoutState),
    )


class Linear(ModelBase):
    in_dim: int
    out_dim: int
    w_init: Initializer
    b_init: None | Initializer = None
    activation: None | Callable = None

    class DenseParam(BaseConfig):
        w: Array
        b: None | Array

    def param(self, rng: PRNGKey) -> DenseParam:
        rng_w, rng_b = jax.random.split(rng)
        return self.DenseParam(
            w=self.w_init(rng_w, (self.in_dim, self.out_dim)),
            b=self.b_init(rng_b, (self.out_dim,)) if self.b_init else None,
        )

    def forward(self, ps: DenseParam, x: Array, st: None) -> tuple[Array, None]:
        o = jnp.einsum("...d,dh->...h", x, ps.w)
        if ps.b is not None:
            o += ps.b
        if self.activation:
            o = self.activation(o)
        return o, st


class LayerNorm(ModelBase):
    dim: int
    ϵ: float = 1e-5
    w_init: Initializer = ones
    b_init: Initializer = zeros

    class LayerNormParam(BaseConfig):
        w: Array
        b: Array

    def param(self, rng: PRNGKey) -> LayerNormParam:
        w_rng, b_rng = jax.random.split(rng)
        return self.LayerNormParam(
            w=self.w_init(w_rng, (self.dim,)), b=self.b_init(b_rng, (self.dim,))
        )

    def forward(self, ps: LayerNormParam, x: Array, st: None) -> tuple[Array, None]:
        x_mean = x.mean(axis=-1, keepdims=True)
        x -= x_mean
        var = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.ϵ)
        # TODO: cast dtype
        return x * ps.w + ps.b, st


class Learner(ModelBase):
    model: ModelBase
    loss_fn: Callable
    agg: Callable = jnp.mean
    feature_name: str = "feature"
    label_name: str = "label"

    def forward(self, ps: Param, input: PyTree, st: State) -> tuple[PyTree, State]:
        x = input[self.feature_name]
        y = input[self.label_name]
        ŷ, st = self.model(ps, x, st)
        losses = self.loss_fn(ŷ, y)
        l = self.agg(losses)
        return l, st


class Trainer(ModelBase):

    learner: Learner
    optimizer: Any

    class TrainerState(StateBase):
        learner_state: State
        step: int = 0
        opt_state: Any = None
        loss: float = 0.0

    def state(self, rng: PRNGKey) -> TrainerState:
        return self.TrainerState(learner_state=self.learner.state(rng))

    def init(self, rng: PRNGKey) -> tuple[Param, TrainerState]:
        rng_ps, rng_st = jax.random.split(rng)
        ps = self.param(rng_ps)
        st = self.TrainerState(
            learner_state=self.learner.state(rng_st), opt_state=self.optimizer.init(ps)
        )
        return {"learner": ps}, st

    @partial(jit, static_argnums=0)
    def forward_and_backward(self, ps, x, ps_st, opt_st):
        (loss, ps_st), grads = value_and_grad(self.learner.forward, has_aux=True)(
            ps, x, ps_st
        )
        updates, opt_st = self.optimizer.update(grads, opt_st)
        ps = optax.apply_updates(ps, updates)
        return loss, ps, ps_st, opt_st

    def __call__(
        self, ps: Param, x: PyTree, st: TrainerState
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

    observer: Callable[[Trainer, Param, State], None] = lambda t, p, s: None

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

weight_decay = 1e-1
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99

warmup_iters = 100


def observer(): ...


def create_attn_block(B, T, N, H):
    D = N * H
    return Chain(
        Linear(in_dim=D, out_dim=3 * D, w_init=truncated_normal(stddev=0.02)),
        lambda x: jnp.dsplit(x, 3),
        Parallel(
            n=3,
            layer=Chain(Reshape(B, T, N, H), jnp.matrix_transpose),
            connection=partial(jax.nn.dot_product_attention, is_causal=True),
        ),
        Dropout(rate=dropout),
        jnp.matrix_transpose,
        Reshape(B, T, D),
        Linear(in_dim=D, out_dim=D, w_init=truncated_normal(stddev=0.02)),
        Dropout(rate=dropout),
    )


exp = Experiment(
    name="nanoGPT",
    trainer=Trainer(
        learner="",
        optimizer=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                    init_value=learning_rate / warmup_iters,
                    peak_value=learning_rate,
                    warmup_steps=warmup_iters,
                    decay_steps=lr_decay_iters,
                    end_value=min_lr,
                ),
                b1=beta1,
                b2=beta2,
                weight_decay=weight_decay,
                mask=lambda p: jax.tree.map(lambda x: x.ndim != 1, p),
            ),
        ),
    ),
    dataset_factory=lambda: dataset(),
)
