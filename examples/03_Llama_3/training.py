import optax

from julax.experiment import Experiment, run
from julax.layers import (
    Learner,
    Trainer,
)
from model import create_model
from dataset import create_dataset
from jsonargparse import auto_cli

import logging
from absl import logging as absl_logging

logging.root.setLevel(logging.INFO)
absl_logging.use_python_logging()


def create_experiment(
    data_dir: str = "./debug/data/",
    tokenizer_dir: str = "./debug/tokenizer",
    model: str = "debug",
):
    match model:
        case "debug":
            return Experiment(
                name=model,
                max_steps=10,
                trainer=Trainer(
                    learner=Learner(
                        feature_name="inputs",
                        label_name="target_labels",
                        model=create_model(model),
                        loss_fn=optax.softmax_cross_entropy_with_integer_labels,
                    ),
                    optimizer=optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.scale_by_adam(
                            b1=0.9,
                            b2=0.95,
                            eps=1e-8,
                        ),
                        optax.add_decayed_weights(0.1),
                        optax.scale_by_schedule(
                            optax.warmup_cosine_decay_schedule(
                                init_value=0.0,
                                peak_value=8e-4,
                                warmup_steps=2,
                                decay_steps=8,
                                end_value=8e-5,
                            )
                        ),
                        optax.scale(-1.0),
                    ),
                ),
                dataset=create_dataset(
                    batch_size=4,
                    seq_len=1024,
                    data_dir=data_dir,
                    tokenizer_dir=tokenizer_dir,
                    split_pattern="*.jsonl",
                    n_open_files=1,
                    n_prefetch_per_file=1,
                ),
            )
        case "llama_3.2_1b":
            return Experiment(
                name="llama_3.2_1b",
                max_steps=1000,
                trainer=Trainer(
                    learner=Learner(
                        feature_name="inputs",
                        label_name="target_labels",
                        model=create_model(),
                        loss_fn=optax.softmax_cross_entropy_with_integer_labels,
                    ),
                    optimizer=optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.scale_by_adam(
                            b1=0.9,
                            b2=0.95,
                            eps=1e-8,
                        ),
                        optax.add_decayed_weights(0.1),
                        optax.scale_by_schedule(
                            optax.warmup_cosine_decay_schedule(
                                init_value=0.0,
                                peak_value=3e-4,
                                warmup_steps=2_000,
                                decay_steps=30_000,
                                end_value=3e-5,
                            )
                        ),
                        optax.scale(-1.0),
                    ),
                ),
                dataset=create_dataset(
                    batch_size=4,
                    seq_len=8192,
                    data_dir=data_dir,
                    tokenizer_dir=tokenizer_dir,
                ),
            )
        case _:
            raise ValueError(f"Unknown model: {model}")


if __name__ == "__main__":
    exp = auto_cli(create_experiment, as_positional=False)
    run(exp)
