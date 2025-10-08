# inspired by:
# - https://docs.jax.dev/en/latest/notebooks/neural_network_with_tfds_data.html

import jax
import jax.numpy as jnp
from jax import jit, grad
import tensorflow_datasets as tfds  # TFDS to download MNIST.

hidden_sizes = [784, 512, 512, 10]
scale = 0.01
step_size = 0.01
batch_size = 32

train_ds = tfds.load("mnist", split="train").repeat().shuffle(1024).batch(batch_size, drop_remainder=True).take(1000).as_numpy_iterator()
test_ds = tfds.load("mnist", split="test").batch(batch_size, drop_remainder=True).take(1000)

rng = jax.random.key(0)

params = []

for i in range(len(hidden_sizes) - 1):
    layer_rng, rng = jax.random.split(rng)
    w_rng, b_rng = jax.random.split(layer_rng)
    params.append(
        {
            "w": scale
            * jax.random.normal(layer_rng, (hidden_sizes[i], hidden_sizes[i + 1])),
            "b": scale * jax.random.normal(layer_rng, (hidden_sizes[i + 1],)),
        }
    )

def log_probs(params, x):
    h = x
    for layer in params[:-1]:
        h = jax.nn.relu(jnp.dot(h, layer["w"]) + layer["b"])
    logits = jnp.dot(h, params[-1]["w"]) + params[-1]["b"]
    return jax.nn.log_softmax(logits)


def loss_fn(params, x, y):
    ŷ = log_probs(params, x)
    return -jnp.mean(ŷ * y)

def accuracy(params):
    n_correct, n_total = 0, 0
    for batch in test_ds.as_numpy_iterator():
        x = jnp.reshape(batch['image'], (batch_size, -1))
        y = batch['label']
        ŷ = jnp.argmax(log_probs(params, x), axis=1)
        n_correct += (ŷ == y).sum().item()
        n_total += batch_size
    return n_correct / n_total

@jit
def step(params, x, y):
    grads = grad(loss_fn)(params, x, y)
    return [
        {"w": p["w"] - step_size * g["w"], "b": p["b"] - step_size * g["b"]}
        for p, g in zip(params, grads)
    ]


for i, batch in enumerate(train_ds):
    if i % 100 == 0:
        acc = accuracy(params)
        print(f"Step {i}, Accuracy: {acc:.4f}")
    x, y = batch["image"], batch["label"]
    params = step(params, jnp.reshape(x, (32, -1)), jax.nn.one_hot(y, 10))