from typing import TypeAlias, Any
from jax import Array
from jax.sharding import PartitionSpec
import plum

PRNG: TypeAlias = Array
PyTree: TypeAlias = Any
OutShardingType: TypeAlias = PartitionSpec | None

dispatch = plum.Dispatcher(warn_redefinition=True)
