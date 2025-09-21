import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, Array
from jax.nn.initializers import Initializer, truncated_normal
from jax.tree_util import PyTreeDef, register_pytree_with_keys_class, GetAttrKey
from typing import Callable, TypeAlias, Any

PRNGKey: TypeAlias = Array
PyTree: TypeAlias = Any

import optax

import tensorflow_datasets as tfds
import tensorflow as tf

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict, RootModel

from functools import partial


class BaseRoot(RootModel[dict]):

    def __getattr__(self, attr: str):
        if attr in self.root:
            return self.root[attr]
        else:
            raise AttributeError(f"Unknown attribute: {attr}")

    def tree_flatten_with_keys(self):
        keys = tuple(self.root.keys())
        return tuple((GetAttrKey(k), self.root[k]) for k in keys), keys

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))


@register_pytree_with_keys_class
class Param(BaseRoot):
    def __init__(self, **kw):
        super().__init__(root=dict(**kw))


@register_pytree_with_keys_class
class State(BaseRoot):
    def __init__(self, **kw):
        super().__init__(root=dict(**kw))


class ModelBase(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True, ignored_types=(jax.stages.Wrapped,)
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # TODO: respect `FieldInfo`
        jax.tree_util.register_dataclass(
            cls, data_fields=list(cls.model_fields.keys()), meta_fields=[]
        )

    def param(self, rng: PRNGKey) -> Param:
        models = {f: getattr(self, f) for f in self.model_fields_set}
        vals, treedef = jax.tree.flatten(
            models, is_leaf=lambda x: isinstance(x, ModelBase)
        )
        rngs = jax.random.split(rng, len(vals))
        flatten_params = [
            v.param(rng) if isinstance(v, ModelBase) else None
            for v, rng in zip(vals, rngs)
        ]
        params = jax.tree.unflatten(treedef, flatten_params)
        return Param(**{k: v for k, v in params.items() if v is not None})

    def state(self, rng: PRNGKey) -> State:
        models = {f: getattr(self, f) for f in self.model_fields_set}
        vals, treedef = jax.tree.flatten(
            models, is_leaf=lambda x: isinstance(x, ModelBase)
        )
        rngs = jax.random.split(rng, len(vals))
        flatten_states = [
            v.state(rng) if isinstance(v, ModelBase) else None
            for v, rng in zip(vals, rngs)
        ]
        states = jax.tree.unflatten(treedef, flatten_states)
        return State(**{k: v for k, v in states.items() if v is not None})

    def init(self, rng: PRNGKey) -> tuple[Param, State]:
        rng_p, rng_st = jax.random.split(rng)
        return self.param(rng_p), self.state(rng_st)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        raise NotImplementedError

    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        return self.forward(x, p, s)


#####

class Dense(ModelBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = truncated_normal()
    b_init: Initializer = truncated_normal()
    activation: Callable | None = None

    def param(self, rng: PRNGKey) -> Param:
        rng_w, rng_b = jax.random.split(rng)
        p = {"w": self.w_init(rng_w, (self.in_dim, self.out_dim))}
        if self.b_init is not None:
            p["b"] = self.b_init(rng_b, (self.out_dim,))
        return Param(**p)

    def forward(self, x: Array, p: Param, s: State) -> tuple[Array, State]:
        h = jnp.einsum("...d,dh->...h", x, p.w)
        o = h + p.b
        if self.activation:
            o = self.activation(o)
        return o, s


class Chain(ModelBase):
    layers: tuple[ModelBase, ...]

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        h = x
        S = ()
        for l, p, s in zip(self.layers, p.layers, s.layers):
            h, sᵢ = l(h, p, s)
            S += (sᵢ,)
        return h, State(layers=S)


class Learner(ModelBase):
    model: ModelBase
    loss_fn: Callable
    agg: Callable = jnp.mean
    feature_name: str = "feature"
    label_name: str = "label"

    def forward(self, input: dict, p: Param, s: State) -> tuple[PyTree, State]:
        x = input[self.feature_name]
        y = input[self.label_name]
        ŷ, S = self.model(x, p.model, s.model)
        losses = self.loss_fn(ŷ, y)
        l = self.agg(losses)
        return l, State(model=S)


class Trainer(ModelBase):

    learner: Learner
    optimizer: Any

    def state(self, rng: PRNGKey) -> State:
        return State(learner=self.learner.state(rng), optimizer=None, step=0, loss=0.0)

    # def init(self, rng: PRNGKey) -> tuple[Param, State]:
    #     rng_p, rng_s = jax.random.split(rng)
    #     p = self.param(rng_p)
    #     s = State(
    #         learner=self.learner.state(rng_s),
    #         optimizer=self.optimizer.init(p),
    #         step=0,
    #         loss=0.0,
    #     )
    #     return p, s

    @partial(jit, static_argnums=0)
    def forward_and_backward(self, x, p, s):
        (loss, Sₗ), grads = value_and_grad(
            self.learner.forward, argnums=1, has_aux=True
        )(x, p.learner, s.learner)
        updates, Sₒ = self.optimizer.update(
            grads, s.optimizer or self.optimizer.init(p.learner)
        )
        Pₗ = optax.apply_updates(p.learner, updates)
        return loss, Pₗ, Sₗ, Sₒ

    def __call__(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        loss, Pₗ, Sₗ, Sₒ = self.forward_and_backward(x, p, s)
        return Param(learner=Pₗ), State(
            learner=Sₗ, optimizer=Sₒ, step=s.step + 1, loss=loss
        )


class CheckpointManager:
    def load(self) -> tuple[bool, tuple[Param, State] | None]:
        return False, None

    def save(self, model: ModelBase, p: Param, s: State):
        pass


class Experiment(ModelBase):
    name: str

    seed: int = 0
    checkpoint_manager: CheckpointManager = CheckpointManager()

    trainer: Trainer
    dataset_factory: Callable

    observer: Callable

    def state(self, rng: PRNGKey) -> State:
        return State(
            input=self.dataset_factory(),
            trainer=self.trainer.state(rng),
        )

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        P, S = self.trainer(x, p.trainer, s.trainer)
        return Param(trainer=P), State(trainer=S, input=s.input)

    def run(self):
        is_success, param_and_state = self.checkpoint_manager.load()
        if is_success:
            p, s = param_and_state
        else:
            p, s = self.init(jax.random.key(self.seed))

        self.observer(self, p, s)

        for x in s.input:
            p, s = self(x, p, s)

            self.checkpoint_manager.save(self, p, s)
            self.observer(self, p, s)

        return p, s


def observer(x: Experiment, p: Param, s: State):
    if s.trainer.step % 100 == 0:
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
        param = p.trainer.learner.model
        state = s.trainer.learner.model  # TODO: convert to test mode
        n_correct, n_total = 0, 0
        for batch in dataset:
            ŷ, _ = model(batch["feature"], param, state)
            n_correct += (ŷ.argmax(axis=1) == batch["label"]).sum().item()
            n_total += 32
        acc = n_correct / n_total

        logging.info(f"Accuracy at step {s.trainer.step}: {acc}")


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
    observer=observer,
)
