from jax.nn.initializers import Initializer, uniform
from jsonargparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class Model:
    name: str = "default"
    init: Initializer = uniform(0.5)


parser = ArgumentParser()
parser.add_argument("--m", type=Model)
args = parser.parse_args(
    [
        "--m.name",
        "abc",
        "--m.init",
        '{"class_path": "jax.nn.initializers.truncated_normal", "init_args": {"value": -7}}',
    ]
)
