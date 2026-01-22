import logging

import grain
import jax
from jax.nn.initializers import truncated_normal
import optax
import tensorflow_datasets as tfds

from julax.layers import (
    Chain,
    Learner,
    Linear,
    Trainer,
    test_mode,
)

from julax.base import Param, State

from julax.experiment import Experiment, default_observer, run

from absl import logging as absl_logging

logging.root.setLevel(logging.INFO)
absl_logging.use_python_logging()


E = Experiment(
    name="mnist",
    trainer=Trainer(
        learner=Learner(
            model=Chain(
                Linear(
                    in_dim=784,
                    out_dim=512,
                    w_init=truncated_normal(),
                ),
                jax.nn.relu,
                Linear(
                    in_dim=512,
                    out_dim=512,
                    w_init=truncated_normal(),
                ),
                jax.nn.relu,
                Linear(
                    in_dim=512,
                    out_dim=10,
                    w_init=truncated_normal(),
                ),
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
)

eval_dataset = (
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


def evaluate(step: int, exp: Experiment, param: Param, state: State):
    if step % 100 == 0:
        model = exp.trainer.learner.model
        param = param["learner"]["model"]
        state = test_mode(state["learner"]["model"])
        n_correct, n_total = 0, 0
        for batch in iter(eval_dataset):
            ŷ, _ = model(batch["feature"], param, state)
            n_correct += (ŷ.argmax(axis=1) == batch["label"]).sum().item()
            n_total += 32
        acc = n_correct / n_total

        logging.info(f"Accuracy at step {step}: {acc:.4f}")


observer = default_observer() * evaluate

run(E, observer)
