import numpy as np
import asyncio
import jax
import os
from dataclasses import dataclass
from typing import Optional, Sequence
import grain
import numpy as np
import tensorflow_datasets as tfds
from pprint import pprint
import orbax.checkpoint as ocp
from orbax.checkpoint._src.futures import future


class ScalarHandler(NumpyHandler):
    """A wrapper around NumpyHandler to deal with scalar types (int, float, etc.)."""

    def typestr(self) -> str:
        return "scalar"

    async def metadata(
        self, infos: Sequence[types.ParamInfo]
    ) -> Sequence[ScalarMetadata]:
        metadatas = await super().metadata(infos)
        return [
            ScalarMetadata(name=m.name, directory=m.directory, dtype=m.dtype)
            for m in metadatas
        ]

    async def serialize(
        self,
        values: Sequence[Scalar],  # pytype: disable=signature-mismatch
        infos: Sequence[types.ParamInfo],
        args: Optional[Sequence[types.SaveArgs]] = None,
    ) -> Sequence[future.Future]:
        """See superclass documentation."""
        values = [np.asarray(v) for v in values]
        return await super().serialize(values, infos, args)

    async def deserialize(
        self,
        infos: Sequence[types.ParamInfo],
        args: Optional[Sequence[RestoreArgs]] = None,
    ) -> Sequence[Scalar]:  # pytype: disable=signature-mismatch
        """See superclass documentation."""
        results = await super().deserialize(infos, args)
        for r in results:
            if r.ndim != 0:
                raise ValueError("Restored result is not a scalar.")
        results = [r.item() for r in results]
        if args:
            # Cast to the intended `restore_type` if it is provided.
            return [
                a.restore_type(r) if a.restore_type else r
                for a, r in zip(args, results)
            ]
        return results

    def memory_size(
        self, values: Sequence[Scalar]
    ) -> Sequence[Tuple[int, int]]:  # pytype: disable=signature-mismatch
        actual_sizes = [sys.getsizeof(v) for v in values]
        if multihost.process_index() == 0:
            write_sizes = actual_sizes
        else:
            write_sizes = [0 for _ in values]
        read_sizes = actual_sizes
        return list(zip(write_sizes, read_sizes))


class GrainHandler(ocp.type_handlers.ScalarHandler):
    @dataclass
    class RestoreArgs(ocp.type_handlers.RestoreArgs):
        item: grain.DatasetIterator | grain.DataLoaderIterator | None = None

    def typestr(self) -> str:
        return "GrainIterator"

    async def serialize(
        self,
        values: Sequence[grain.DatasetIterator | grain.DataLoaderIterator],
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[ocp.type_handlers.SaveArgs]],
    ) -> Sequence[future.Future]:
        vals = [x.get_state() for x in values]
        return await super().serialize(vals, infos, args)

    async def deserialize(
        self,
        infos: Sequence[ocp.type_handlers.ParamInfo],
        args: Optional[Sequence[RestoreArgs]],
    ) -> Sequence[grain.DatasetIterator | grain.DataLoaderIterator]:
        await asyncio.gather(
            *(arg.item.load(self._ckpt_dir(info)) for arg, info in zip(args, infos))
        )

        return [arg.item for arg in args]


ds = (
    grain.MapDataset.source(tfds.data_source("mnist", split="train"))
    .seed(seed=45)
    .shuffle()
    .to_iter_dataset()
)

num_steps = 4
ds_iter = iter(ds)

# Read some elements.
for i in range(num_steps):
    x = next(ds_iter)
    print(i, x["label"])

import shutil

if os.path.exists("/tmp/orbax"):
    shutil.rmtree("/tmp/orbax")


# from grain._src.python.checkpoint_handlers import (
#     CheckpointHandler as GrainIteratorHandler,
# )

ocp.type_handlers.register_type_handler(
    grain.DatasetIterator, GrainHandler(), override=True
)

state = {
    "iter": ds_iter,
    "a": np.arange(16),
    "b": np.ones(16),
}

mngr = ocp.CheckpointManager("/tmp/orbax")
mngr.save(
    step=num_steps,
    args=ocp.args.PyTreeSave(item=state),
)

mngr.wait_until_finished()

# # Read some elements.
# for i in range(num_steps):
#     x = next(ds_iter)
#     print(i, x["label"])

# restored = mngr.restore(
#     num_steps,
#     args=ocp.args.PyTreeRestore(
#         item=state,
#         restore_args={"iter": GrainHandler.RestoreArgs(item=ds_iter)},
#     ),
# )
