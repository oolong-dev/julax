from functools import cached_property
from jax.sharding import Mesh

from julax.utils import create_mesh
from pydantic import BaseModel

from julax.base import State, Param
from julax.layers import Trainer

import grain

import orbax.checkpoint as ocp

import logging

from pydantic import computed_field


logger = logging.getLogger(__name__)


class Experiment(BaseModel):
    name: str = "mnist"

    seed: int = 0
    trainer: Trainer

    dataset: grain.IterDataset

    max_steps: int | None = None
    batch_axis_names: list[str] = ["data"]
    mesh_shape: dict[str, int] = {"data": -1}

    checkpoint_manager: ocp.CheckpointManager | None = None
    # observer: ObserverBase = Field(default_factory=default_observer)

    @computed_field
    @cached_property
    def mesh(self) -> Mesh:
        return create_mesh(self.mesh_shape)

    def restore(self) -> tuple[int, Param, State, grain.DatasetIterator]:
        p, s = self.trainer.init(self.seed)
        i = iter(self.dataset)
        if self.checkpoint_manager is None:
            return 0, p, s, i

        step = self.checkpoint_manager.latest_step()

        if step is None:
            return 0, p, s, i

        restored = self.checkpoint_manager.restore(
            step=None,
            args=ocp.args.Composite(
                param=ocp.args.PyTreeRestore(
                    item=p,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(p),
                ),
                state=ocp.args.PyTreeRestore(
                    item=s,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(s),
                ),
                input=grain.checkpoint.CheckpointRestore(item=i),
            ),
        )
        return step, restored["param"], restored["state"], restored["input"]

    def save(self, step: int, p: Param, s: State, i: grain.DatasetIterator):
        if self.checkpoint_manager:
            self.checkpoint_manager.save(
                step,
                args=ocp.args.Composite(
                    param=ocp.args.PyTreeSave(item=p),
                    state=ocp.args.PyTreeSave(item=s),
                    input=grain.checkpoint.CheckpointSave(item=i),
                ),
            )

    def close(self):
        if self.checkpoint_manager:
            self.checkpoint_manager.close()
