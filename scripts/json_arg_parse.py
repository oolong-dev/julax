from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from jsonargparse import auto_cli

# from jsonargparse._common import not_subclass_type_selectors

# not_subclass_type_selectors.pop("dataclass")
# not_subclass_type_selectors.pop("pydantic")

from typing import Annotated, ClassVar, Literal, Union, TypeAlias

Model: TypeAlias


class ModelBase(BaseModel):
    def model_dump(self, **kwargs) -> dict:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)

    _subclasses: ClassVar[dict] = {}

    @model_serializer(mode="wrap")
    def inject_type_on_serialization(self, handler):
        result = handler(self)
        # if "kind" in result:
        #     raise ValueError(f'Cannot use field "kind". It is reserved. {result}')
        result["kind"] = f"{self.__class__.__name__}"
        return result

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        ModelBase._subclasses[cls.__name__] = cls

    @model_validator(mode="wrap")
    @classmethod
    def _parse_into_subclass(cls, value, handler):
        print("!!!", cls, value)
        if isinstance(value, dict) is False:
            return handler(value)
        if cls is not ModelBase:
            return value
        class_full_name = value.pop("kind", None)
        if class_full_name is None:
            raise ValueError("Missing `kind` field")
        print("===", class_full_name)
        class_type = cls._subclasses.get(class_full_name, None)
        res = class_type.model_validate(value)
        print("***", cls, class_type, res)
        return res


class ModelX(ModelBase):
    name: str = "X"


class ModelY(ModelBase):
    name: str = "Y"


class Experiment(ModelBase):
    model: ModelBase


# defined after Experiment
class ModelZ(ModelBase):
    name: str = "Z"


exp = Experiment(model=ModelZ())


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
