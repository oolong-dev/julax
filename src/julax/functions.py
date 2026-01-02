import jax.numpy as jnp


def multiply(x: dict) -> jnp.ndarray:
    """Multiplies the values of a dictionary."""
    return jnp.multiply(*x.values())
