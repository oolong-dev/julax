import math
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils


def identity(x):
    return x


def create_mesh(mesh_shape: dict[str, int]) -> Mesh:
    # TODO: support multi-slice
    values = list(mesh_shape.values())
    if -1 in values:
        product = math.prod(v for v in values if v != -1)
        values[values.index(-1)] = jax.device_count() // product

    devices = mesh_utils.create_device_mesh(values, jax.devices())
    return Mesh(devices, list(mesh_shape.keys()))
