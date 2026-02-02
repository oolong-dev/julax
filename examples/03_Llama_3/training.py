import optax

from julax.experiment.experiment import Experiment
from julax.layers import (
    Learner,
    Trainer,
)
from model import create_model
from dataset import create_dataset
from jsonargparse import auto_cli


def create_experiment(data_dir: str, tokenizer_dir: str):
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
            seq_len=4096,
            data_dir=data_dir,
            tokenizer_dir=tokenizer_dir,
        ),
    )


if __name__ == "__main__":
    exp = auto_cli(create_experiment)
    exp.run()
