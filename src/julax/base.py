from typing import TypeAlias, Any
from pydantic import ConfigDict, RootModel
from jax import Array
from jax.sharding import PartitionSpec
import plum

PRNG: TypeAlias = Array
PyTree: TypeAlias = Any
OutShardingType: TypeAlias = PartitionSpec | None

# TODO: isinstance(jnp.dtype, jnp.float32) fails
Dtype: TypeAlias = Any

Param: TypeAlias = dict
State: TypeAlias = dict


dispatch = plum.Dispatcher(warn_redefinition=True)


class FrozenDict(RootModel[dict]):
    model_config = ConfigDict(frozen=True)

    def __getitem__(self, item):
        return self.root[item]

    def __iter__(self):
        return iter(self.root)

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    def __len__(self):
        return len(self.root)

    def __hash__(self):
        return hash(frozenset(self.root.items()))

    def __eq__(self, other):
        if isinstance(other, FrozenDict):
            return self.root == other.root
        return self.root == other
