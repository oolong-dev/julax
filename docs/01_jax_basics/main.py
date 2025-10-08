import numpy as np
from jax import grad, jit
from jax import lax
import jax
import jax.numpy as jnp

#####
# Initialization
#####

jnp.arange(5)
# Array([0, 1, 2, 3, 4], dtype=int32)

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

x + y
# Array([5., 7., 9.], dtype=float32)

lax.add(x, y)
# Array([5., 7., 9.], dtype=float32)

x[0] = 1
# TypeError: JAX arrays are immutable and do not support in-place item assignment. Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html

# !!!
x[5]
# Array(3., dtype=float32)

x[4:5]
# Array([], shape=(0,), dtype=float32)

x[4:5].shape
# (0,)

#####
# Random
#####

rng = jax.random.key(0)
# Array((), dtype=key<fry>) overlaying:
# [0 0]

jax.random.uniform(rng, shape=(2, 3))
# Array([[0.947667  , 0.9785799 , 0.33229148],
#        [0.46866846, 0.5698887 , 0.16550303]], dtype=float32)

rng_a, rng_b = jax.random.split(rng, 2)

jax.random.uniform(rng_a, shape=(2, 3))
# Array([[0.8423141 , 0.18237865, 0.2271781 ],
#        [0.12072563, 0.19181347, 0.722015  ]], dtype=float32)

#####
# pytree
#####

ps = {"n": 5, "W": jnp.ones((2, 2)), "b": jnp.zeros(2)}

jax.tree.structure(ps)
# PyTreeDef({'W': *, 'b': *, 'n': *})

jax.tree.leaves(ps)
# [Array([[1., 1.],
#         [1., 1.]], dtype=float32),
#  Array([0., 0.], dtype=float32),
#  5]

jax.tree.map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
# [[5, 7, 9], [6, 1, 2]]

jax.tree.reduce(+, [1, [-1, 2], 3])