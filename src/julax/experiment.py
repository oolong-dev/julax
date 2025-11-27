import jax
from jax.sharding import PartitionSpec

from julax.utils import create_mesh

from .core import PRNG, LayerBase, Trainer, State, Param, PyTree
import grain

import orbax.checkpoint as ocp

import logging

from pydantic import Field

from .observers import default_observer, ObserverBase

logger = logging.getLogger(__name__)


class Experiment(LayerBase):
    name: str = "mnist"

    seed: int = 0
    trainer: Trainer

    dataset: grain.IterDataset
    batch_axis_names: list[str] = ["data"]
    mesh_shape: dict[str, int] = {"data": -1}

    checkpoint_manager: ocp.CheckpointManager | None = None
    observer: ObserverBase = Field(default_factory=default_observer)

    def state(self, rng: PRNG) -> State:
        return State(input=iter(self.dataset), step=0)

    def forward(self, x: PyTree, p: Param, s: State) -> tuple[PyTree, State]:
        P, S = self.trainer(x, p["trainer"], s["trainer"])
        return Param(trainer=P), State(trainer=S, input=s["input"], step=s["step"] + 1)

    def save(self, p: Param, s: State):
        if self.checkpoint_manager:
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
        if self.checkpoint_manager is None:
            return p, s
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

    def close(self):
        if self.checkpoint_manager:
            self.checkpoint_manager.close()

    def run(self) -> tuple[Param, State]:
        with create_mesh(self.mesh_shape) as mesh:
            p, s = self.restore()
            self.observer(self, p, s)

            for x_local in s["input"]:
                x = jax.make_array_from_process_local_data(
                    sharding=jax.sharding.NamedSharding(
                        mesh, PartitionSpec(self.batch_axis_names)
                    ),
                    local_data=x_local,
                )
                p, s = self(x, p, s)

                self.observer(self, p, s)
                self.save(p, s)

            return p, s
