from typing import TypeAlias, Any
from jax import Array
import plum

PRNG: TypeAlias = Array
PyTree: TypeAlias = Any

dispatch = plum.Dispatcher(warn_redefinition=True)
