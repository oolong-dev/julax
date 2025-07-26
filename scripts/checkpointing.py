from plum import dispatch, overload
import jax
import logging

logger = logging.getLogger(__name__)

@overload
async def serialize(x: jax.Array):
    for shard in x.addressable_shards:
        data_on_host = jax.device_put(
            shard.data,
            jax.sharding.SingleDeviceSharding(shard.data.device, memory_kind='pinned_host'),
        )

@overload
async def serialize(x: str):
    ...

@dispatch
async def serialize(x):
    ...
