from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax.nn.initializers import Initializer, truncated_normal

import optax

import tensorflow_datasets as tfds

from jaxtyping import PRNGKeyArray, PyTree, Array, Num
from pydantic import BaseModel, ConfigDict
from typing import Callable


class ModelBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def params(self, rng: PRNGKeyArray) -> PyTree:
        raise NotImplementedError

    def states(self, rng: PRNGKeyArray) -> PyTree:
        return None

    def init(self, rng: PRNGKeyArray) -> tuple[PyTree, PyTree]:
        rng_ps, rng_st = jax.random.split(rng)
        return self.params(rng_ps), self.states(rng_st)

    def forward(self, ps: PyTree, x: PyTree, st: PyTree) -> tuple[PyTree, PyTree]:
        raise NotImplementedError

    def __call__(self, ps: PyTree, x: PyTree, st: PyTree) -> tuple[PyTree, PyTree]:
        return self.forward(ps, x, st)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # TODO: respect `FieldInfo`
        jax.tree_util.register_dataclass(
            cls, data_fields=[], meta_fields=list(cls.model_fields.keys())
        )


class Dense(ModelBase):
    in_dim: int
    out_dim: int
    w_init: Initializer = truncated_normal()
    b_init: Initializer = truncated_normal()
    activation: None | Callable = None

    def params(self, rng: PRNGKeyArray) -> PyTree:
        rng_w, rng_b = jax.random.split(rng)
        return {
            "w": self.w_init(rng_w, (self.in_dim, self.out_dim)),
            "b": self.b_init(rng_b, (self.out_dim,)),
        }

    def forward(
        self, ps: PyTree, x: Num[Array, "... d"], st: None
    ) -> tuple[Num[Array, "... h"], None]:
        h = jnp.einsum("...d,dh->...h", x, ps["w"])
        o = h + ps["b"]
        if self.activation:
            o = self.activation(o)
        return o, st


class Chain(ModelBase):
    layers: tuple[ModelBase, ...]

    def params(self, rng: PRNGKeyArray) -> PyTree:
        rngs = jax.random.split(rng, len(self.layers))
        return [layer.params(rng) for layer, rng in zip(self.layers, rngs)]

    def states(self, rng: PRNGKeyArray) -> list[PyTree]:
        rngs = jax.random.split(rng, len(self.layers))
        return [layer.states(rng) for layer, rng in zip(self.layers, rngs)]

    def forward(
        self, ps: list[PyTree], x: PyTree, st: tuple[PyTree, ...]
    ) -> tuple[PyTree, tuple[PyTree, ...]]:
        h = x
        _st = ()
        for l, p, s in zip(self.layers, ps, st):
            h, _s = l(p, h, s)
            _st = (*_st, _s)
        return h, _st


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


@jit
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
