from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.nn.initializers import Initializer, truncated_normal
from jax._src import pjit

from jaxlib._jax.pytree import SequenceKey
from jaxlib._jax import ArrayImpl

import optax

import tensorflow_datasets as tfds

from jaxtyping import PRNGKeyArray, PyTree, Array, Num
from pydantic import BaseModel, ConfigDict
from typing import Callable, Literal

from plum import Dispatcher, parametric

dispatch = Dispatcher(warn_redefinition=True)

from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.console import RenderableType, Console, Group

console = Console()

#####
# Visualization
#####


@dispatch
def summary(x) -> str:
    return str(x)


@dispatch
def summary(x: int | float) -> str:
    return f"[bold cyan]{x}[/bold cyan]"


@dispatch
def summary(x: ArrayImpl) -> str:
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
def typeof(x: ArrayImpl) -> str:
    return f"jax.Array{{{x.dtype} {x.shape}}}"


@dispatch
def to_rich(x) -> RenderableType:
    return to_rich(None, x)


@dispatch
def to_rich(path, x) -> RenderableType:
    t = typeof(x)
    ts = f"italic color({hash(type(x)) % 256})"
    label = f"[{ts} dim]<{t}>[/{ts} dim]"
    k = jax.tree_util.keystr(path, simple=True) if path else "🎯"
    ks = f"color({hash(k) % 256})"
    label = f"[{ks} bold]{k}[/{ks} bold]: {label}"

    if jax.tree_util.treedef_is_leaf(jax.tree.structure(x)):
        s = summary(x)
        if isinstance(s, str):
            title, detail = s, []
        else:
            title, detail = s[0], s[1:]
        label = f"{label} [bright_yellow]=>[/bright_yellow] {title}"
        root = Tree(label, guide_style=f"dim {ks or ts}")
        if detail:
            return Group(root, *[Panel(d) for d in detail])
        else:
            return root
    else:
        root = Tree(label, guide_style=f"dim {ks or ts}")

        children = jax.tree.leaves_with_path(
            x, is_leaf=lambda p, v: len(p) == 1, is_leaf_takes_path=True
        )

        # TODO: sort

        for k, v in children:
            root.add(to_rich(k, v))

    return root


#####
class BaseConfig(BaseModel):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # TODO: respect `FieldInfo`
        jax.tree_util.register_dataclass(
            cls, data_fields=list(cls.model_fields.keys()), meta_fields=[]
        )

    def __rich__(self):
        return to_rich(self)


class ModelBase(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    class ParamBase(BaseConfig):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def length(self) -> int:
            return sum(jnp.size(v) for v in jax.tree.leaves(self))

    def param(self, rng: PRNGKeyArray) -> ParamBase:
        raise NotImplementedError

    class StateBase(BaseConfig):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def length(self) -> int:
            return sum(
                jnp.size(v) for v in jax.tree.leaves(self) if isinstance(v, Array)
            )

    def state(self, rng: PRNGKeyArray) -> StateBase:
        return self.StateBase()

    def init(self, rng: PRNGKeyArray) -> tuple[PyTree, PyTree]:
        rng_ps, rng_st = jax.random.split(rng)
        return self.param(rng_ps), self.state(rng_st)

    def forward(self, ps: PyTree, x: PyTree, st: PyTree) -> tuple[PyTree, PyTree]:
        raise NotImplementedError

    def __call__(self, ps: PyTree, x: PyTree, st: PyTree) -> tuple[PyTree, PyTree]:
        return self.forward(ps, x, st)


@dispatch
def summary(x: ModelBase.StateBase) -> str:
    return "EMPTY"


class Dense(ModelBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = truncated_normal()
    b_init: Initializer = truncated_normal()
    activation: None | Callable = None

    class DenseParam(ModelBase.ParamBase):
        w: Num[Array, "d h"]
        b: Num[Array, "h"]

    def param(self, rng: PRNGKeyArray) -> DenseParam:
        rng_w, rng_b = jax.random.split(rng)
        return self.DenseParam(
            w=self.w_init(rng_w, (self.in_dim, self.out_dim)),
            b=self.b_init(rng_b, (self.out_dim,)),
        )

    def forward(
        self, ps: DenseParam, x: Num[Array, "... d"], st: None
    ) -> tuple[Num[Array, "... h"], None]:
        h = jnp.einsum("...d,dh->...h", x, ps.w)
        o = h + ps.b
        if self.activation:
            o = self.activation(o)
        return o, st


class Chain(ModelBase):
    layers: tuple[ModelBase, ...]

    class ChainParam(ModelBase.ParamBase):
        layers: tuple[PyTree, ...]

    def param(self, rng: PRNGKeyArray) -> PyTree:
        rngs = jax.random.split(rng, len(self.layers))
        return self.ChainParam(
            layers=tuple(layer.param(rng) for layer, rng in zip(self.layers, rngs))
        )

    class ChainState(ModelBase.StateBase):
        layers: tuple[ModelBase.StateBase, ...]

    def state(self, rng: PRNGKeyArray) -> ModelBase.StateBase:
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


step_size = 0.01
batch_size = 32

train_ds = (
    tfds.load("mnist", split="train")
    .repeat()
    .shuffle(1024, seed=123)
    .batch(batch_size, drop_remainder=True)
    .take(1000)
    .as_numpy_iterator()
)
test_ds = (
    tfds.load("mnist", split="test").batch(batch_size, drop_remainder=True).take(1000)
)

model = Chain(
    layers=(
        Dense(in_dim=784, out_dim=512, activation=jax.nn.relu),
        Dense(in_dim=512, out_dim=512, activation=jax.nn.relu),
        Dense(in_dim=512, out_dim=10),
    )
)

rng = jax.random.key(0)
params, states = model.init(rng)

optimizer = optax.sgd(0.01)
opt_state = optimizer.init(params)


def loss_fn(model, params, states, x, y):
    logits, states = model(params, x, states)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(losses), states


def accuracy(model, params, states):
    n_correct, n_total = 0, 0
    for batch in test_ds.as_numpy_iterator():
        x = jnp.reshape(batch["image"], (batch_size, -1))
        y = batch["label"]
        logits, _ = model(params, x, states)
        ŷ = jnp.argmax(logits, axis=1)
        n_correct += (ŷ == y).sum().item()
        n_total += batch_size
    return n_correct / n_total


@partial(jit, static_argnames=("model",))
def step(model, params, states, opt_state, x, y):
    (loss, states), grads = value_and_grad(loss_fn, has_aux=True, argnums=1)(
        model, params, states, x, y
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return loss, params, states, opt_state


for i, batch in enumerate(train_ds):
    if i % 100 == 0:
        acc = accuracy(model, params, states)
        print(f"Step {i}, Accuracy: {acc:.4f}")
    x, y = batch["image"], batch["label"]
    loss, params, states, opt_state = step(
        model, params, states, opt_state, jnp.reshape(x, (32, -1)), y
    )
