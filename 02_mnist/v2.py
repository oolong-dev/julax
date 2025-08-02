from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, grad
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

    def __call__(self, ps: PyTree, x: PyTree) -> PyTree:
        raise NotImplementedError


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

    def __call__(self, ps: PyTree, x: Num[Array, "... d"]) -> Num[Array, "... h"]:
        h = jnp.einsum("...d,dh->...h", x, ps["w"])
        o = h + ps["b"]
        if self.activation:
            o = self.activation(o)
        return o


class Chain(ModelBase):
    layers: tuple[ModelBase, ...]

    def params(self, rng: PRNGKeyArray) -> PyTree:
        ps = []
        for layer in self.layers:
            layer_rng, rng = jax.random.split(rng)
            ps.append(layer.params(layer_rng))
        return ps

    def __call__(self, ps: list[PyTree], x: PyTree) -> PyTree:
        h = x
        for layer, ps in zip(self.layers, ps):
            h = layer(ps, h)
        return h


step_size = 0.01
batch_size = 32

train_ds = (
    tfds.load("mnist", split="train")
    .repeat()
    .shuffle(1024)
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
params = model.params(rng)

optimizer = optax.sgd(0.01)
opt_state = optimizer.init(params)


def loss_fn(params, model, x, y):
    logits = model(params, x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(losses)


def accuracy(params, model):
    n_correct, n_total = 0, 0
    for batch in test_ds.as_numpy_iterator():
        x = jnp.reshape(batch["image"], (batch_size, -1))
        y = batch["label"]
        ŷ = jnp.argmax(model(params, x), axis=1)
        n_correct += (ŷ == y).sum().item()
        n_total += batch_size
    return n_correct / n_total


@partial(jit, static_argnames=["model"])
def step(params, model, opt_state, x, y):
    grads = grad(loss_fn)(params, model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state


for i, batch in enumerate(train_ds):
    if i % 100 == 0:
        acc = accuracy(params, model)
        print(f"Step {i}, Accuracy: {acc:.4f}")
    x, y = batch["image"], batch["label"]
    params, opt_state = step(params, model, opt_state, jnp.reshape(x, (32, -1)), y)
