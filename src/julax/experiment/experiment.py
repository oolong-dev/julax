from functools import cached_property, partial
from jax.sharding import Mesh

from julax.utils import create_mesh
from pydantic import BaseModel, ConfigDict

from julax.base import PyTree, State, Param
from julax.layers import Trainer

import grain

import orbax.checkpoint as ocp

import logging

from pydantic import computed_field

from humanize import naturalsize, metric

logger = logging.getLogger(__name__)

natural_binary_size = partial(naturalsize, binary=True, gnu=True)


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "mnist"

    seed: int = 0
    trainer: Trainer

    dataset: grain.IterDataset

    max_steps: int | None = None
    batch_axis_names: list[str] = ["data"]
    mesh_shape: dict[str, int] = {"data": -1}

    checkpoint_manager: ocp.CheckpointManager | None = None

    @computed_field
    @cached_property
    def mesh(self) -> Mesh:
        return create_mesh(self.mesh_shape)

    def step(self, x: PyTree, p: Param, s: State) -> tuple[Param, State]:
        return self.trainer(x, p, s)

    # TODO: AOT compile
    def precompile(self, x: PyTree, p: Param, s: State):
        traced = self.trainer.forward_and_backward.trace(self.trainer, x, p, s)
        compiled = traced.lower().compile()
        mem = compiled.memory_analysis()
        if mem is not None:
            s_mem = [
                "Compiled step memory analysis",
                f"Input memory: {natural_binary_size(mem.argument_size_in_bytes)}",
                f"Output memory: {natural_binary_size(mem.output_size_in_bytes)}",
                f"Temp memory: {natural_binary_size(mem.temp_size_in_bytes)}",
                f"Code memory: {natural_binary_size(mem.generated_code_size_in_bytes)}",
            ]
            logger.info("\n  ".join(s_mem))
        cost = compiled.cost_analysis()
        if cost is not None:
            s_cost = [
                "Compiled step cost analysis",
                f"FLOPS: {metric(cost.get('flops'))}",
                f"The number of exp/log/sin/cos ops: {metric(cost.get('transcendentals'))}",
                f"The total memory traffic: {natural_binary_size(cost.get('bytes accessed'))}",
                f"  HBM access: {natural_binary_size(cost.get('bytes accessed0{}'))}",
                f"  L2 cache access: {natural_binary_size(cost.get('bytes accessed1{}'))}",
                f"  Register usage: {natural_binary_size(cost.get('bytes accessed2{}'))}",
                f"  Output data transferred: {natural_binary_size(cost.get('bytes accessedout{}'))}",
                "Hardware utilization scores",
                f"  Tensor Cores / MatMul units: {cost.get('utilization0{}')}",
                f"  ALU (Arithmetic Logic Unit): {cost.get('utilization1{}')}",
                f"  Memory Load/Store Units: {cost.get('utilization2{}')}",
                f"  L1 Cache Operations: {cost.get('utilization3{}')}",
                f"  L2 Cache Operations: {cost.get('utilization4{}')}",
                f"  Special Function Units (exp/log/sin/cos): {cost.get('utilization5{}')}",
                f"  Integer Units (for indexing, loop counters): {cost.get('utilization6{}')}",
                f"  Branch Divergence (Control Flow Processing): {cost.get('utilization7{}')}",
                f"  Load Balancing / Dispatch]: {cost.get('utilization8{}')}",
                f"  Texture Units (or Rarely Used Compute Units): {cost.get('utilization9{}')}",
            ]
            logger.info("\n  ".join(s_cost))
        return compiled

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
