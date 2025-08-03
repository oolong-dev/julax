import penzai
from penzai import pz

import treescope

treescope.basic_interactive_setup(autovisualize_arrays=True)

from penzai.models import simple_mlp

mlp = simple_mlp.MLP.from_config(
    name="mlp", init_base_rng=jax.random.key(0), feature_sizes=[8, 32, 32, 8]
)

# Models and arrays are visualized automatically when you output them from a
# Colab/IPython notebook cell:
mlp
