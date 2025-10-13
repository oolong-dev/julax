# /// script
# dependencies = [
#   "julax",
#   "opencv-python",
#   "tensorflow-datasets>=4.9.9",
# ]
#
# [tool.uv.sources]
# julax = { path = "../", editable = true }
# ///

import logging

from datetime import datetime
import os

import grain
import jax
from jax.nn.initializers import truncated_normal
import optax
import orbax.checkpoint as ocp
import tensorflow_datasets as tfds

from julax import (
    Chain,
    DoEveryNSteps,
    Experiment,
    Learner,
    Linear,
    Param,
    State,
    Trainer,
    default_observer,
    test_mode,
)

from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.INFO)
logging.root.setLevel(logging.INFO)


def evaluate(x: Experiment, p: Param, s: State):
    dataset = (
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
    model = x.trainer.learner.model
    param = p["trainer"]["learner"]["model"]
    state = test_mode(s["trainer"]["learner"]["model"])
    n_correct, n_total = 0, 0
    for batch in iter(dataset):
        ŷ, _ = model(batch["feature"], param, state)
        n_correct += (ŷ.argmax(axis=1) == batch["label"]).sum().item()
        n_total += 32
    acc = n_correct / n_total

    logging.info(f"Accuracy at step {s['trainer']['step']}: {acc}")


E = Experiment(
    name="mnist",
    checkpoint_manager=ocp.CheckpointManager(
        directory=os.path.join(
            os.getcwd(), "checkpoints", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ),
        options=ocp.CheckpointManagerOptions(save_interval_steps=100),
    ),
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
    observer=default_observer() * DoEveryNSteps(evaluate, n=100),
)

E.run()
