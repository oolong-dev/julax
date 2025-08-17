from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.nn.initializers import Initializer, truncated_normal

import optax

import tensorflow_datasets as tfds
import tensorflow as tf

from jaxtyping import PRNGKeyArray, PyTree, Array, Num
from typing import Any, Callable, Iterable

from plum import Dispatcher


dispatch = Dispatcher(warn_redefinition=True)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict

#####
# Visualization
#####

from rich.tree import Tree
from rich.panel import Panel
from rich.console import RenderableType, Group


@dispatch
def summary(x) -> str:
    return str(x)


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


class StateBase(BaseConfig): ...


DEFAULT_STATE = StateBase()


class ModelBase(BaseConfig):
    def param(self, rng: PRNGKeyArray) -> ParamBase:
        raise NotImplementedError

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


def evaluator(trainer: Trainer, params: PyTree, state: Trainer.TrainerState):
    if state.step % 100 == 0:
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
        m = trainer.learner.model
        st = state.learner_state  # TODO: convert to test mode
        n_correct, n_total = 0, 0
        for batch in dataset:
            ŷ, _ = m(params, batch["feature"], st)
            n_correct += (ŷ.argmax(axis=1) == batch["label"]).sum().item()
            n_total += 32
        acc = n_correct / n_total

        logging.info(f"Accuracy at step {state.step}: {acc}")


exp = Experiment(
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
    dataset_factory=lambda: (
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
    ),
    observer=evaluator,
)


# if __name__ == "__main__":
#     exp.run()
