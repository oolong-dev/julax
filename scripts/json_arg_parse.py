from pydantic import BaseModel

from jsonargparse import auto_cli

from jsonargparse._common import not_subclass_type_selectors

not_subclass_type_selectors.pop("dataclass")
not_subclass_type_selectors.pop("pydantic")


class ModelBase(BaseModel): ...


class Model(ModelBase):
    name: str = "default"


class Experiment(ModelBase):
    model: ModelBase


exp = Experiment(model=Model())


def run(x: Experiment = exp):
    print(x)


if __name__ == "__main__":
    auto_cli(run)

# # parser = ArgumentParser()
# # parser.add_argument("--m", type=Model)

# # # cfg = parser.parse_path("config.yaml")

# # args = parser.parse_args(
# #     [
# #         "--m.name",
# #         "abc",
# #         "--m.init",
# #         '{"class_path": "jax.nn.initializers.truncated_normal", "init_args": {"lower": -1}}',
# #     ]
# # )

# # if __name__ == "__main__":
# #     print(auto_cli(Experiment, as_positional=False))

# from dataclasses import dataclass


# class Base(BaseModel):
#     a: str = "a"


# class B(Base):
#     b: str = "b"


# class X(Base):
#     val: Base


# def f(x: X = X(val=B())):
#     print(x)


# if __name__ == "__main__":
#     auto_cli(f)
