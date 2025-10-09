from typing import Any, Callable
from .core import PRNG, LayerBase, Trainer, State, Param, PyTree
import grain

import orbax.checkpoint as ocp

import logging

logger = logging.getLogger(__name__)


class Experiment(LayerBase):
    name: str = "mnist"

    seed: int = 0
    checkpoint_manager: ocp.CheckpointManager
    trainer: Trainer
    dataset: grain.IterDataset

    observer: Callable[["Experiment", Param, State], Any]

    def state(self, rng: PRNG) -> State:
        return State(input=iter(self.dataset))

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        P, S = self.trainer(x, p["trainer"], s["trainer"])
        return Param(trainer=P), State(trainer=S, input=s["input"])

    def save(self, p: Param, s: State):
        self.checkpoint_manager.save(
            s["trainer"]["step"],
            args=ocp.args.Composite(
                param=ocp.args.PyTreeSave(item=p),
                state_trainer=ocp.args.PyTreeSave(item=s["trainer"]),
                state_dataset_iter=grain.checkpoint.CheckpointSave(item=s["input"]),
            ),
        )

    def restore(self) -> tuple[Param, State]:
        p, s = self.init(self.seed)
        try:
            restored = self.checkpoint_manager.restore(
                step=None,
                args=ocp.args.Composite(
                    param=ocp.args.PyTreeRestore(
                        item=p,
                        restore_args=ocp.checkpoint_utils.construct_restore_args(p),
                    ),
                    state_trainer=ocp.args.PyTreeRestore(
                        item=s["trainer"],
                        restore_args=ocp.checkpoint_utils.construct_restore_args(
                            s["trainer"]
                        ),
                    ),
                    state_dataset_iter=grain.checkpoint.CheckpointRestore(
                        item=s["input"]
                    ),
                ),
            )
            param = restored["param"]
            state_trainer = restored["state_trainer"]
            state_dataset_iter = restored["state_dataset_iter"]
            return param, State(input=state_dataset_iter, trainer=state_trainer)
        except FileNotFoundError:
            logger.warning(
                f"No checkpoints found under {self.checkpoint_manager.directory} ! Experiment initialized with seed {self.seed}"
            )
            return p, s

    def run(self):
        p, s = self.restore()
        self.observer(self, p, s)

        for x in s["input"]:
            p, s = self(x, p, s)

            self.save(p, s)
            self.observer(self, p, s)

        return p, s
