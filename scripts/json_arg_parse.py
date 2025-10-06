from jsonargparse._common import not_subclass_type_selectors

# not_subclass_type_selectors.pop("dataclass")
not_subclass_type_selectors.pop("pydantic")

from typing import Any, Callable
from pydantic import BaseModel, SerializeAsAny, model_serializer, model_validator


# https://github.com/pydantic/pydantic/discussions/7008#discussioncomment-6826052
class BM(BaseModel):
    __subclasses_map__ = {}

    @model_serializer(mode="wrap")
    def __serialize_with_class_type__(self, default_serializer) -> Any:
        ret = default_serializer(self)
        if isinstance(ret, dict):
            ret["__klass__"] = (
                f"{self.__class__.__module__}.{self.__class__.__qualname__}"
            )
        return ret

    @model_validator(mode="wrap")
    @classmethod
    def __convert_to_real_type__(cls, value: Any, handler):
        if isinstance(value, dict) is False:
            return handler(value)

        # it is a dict so make sure to remove the __klass__
        # because we don't allow extra keywords but want to ensure
        # e.g Cat.model_validate(cat.model_dump()) works
        class_full_name = value.pop("__klass__", None)

        # if it's not the polymorphic base we construct via default handler
        # if not cls.__is_polymorphic_base:
        if not BM in cls.__bases__:
            return handler(value)

        # otherwise we lookup the correct polymorphic type and construct that
        # instead
        if class_full_name is None:
            raise ValueError("Missing __klass__ field")

        class_type = cls.__subclasses_map__.get(class_full_name, None)

        if class_type is None:
            # TODO could try dynamic import
            raise TypeError(
                "Trying to instantiate {class_full_name}, which has not yet been defined!"
            )

        return class_type.model_validate(value)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        cls.__subclasses_map__[f"{cls.__module__}.{cls.__qualname__}"] = cls

    def model_dump(self, **kwargs) -> dict:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)


class Pet(BM):
    name: str


class Cat(Pet):
    meows: int


class SpecialCat(Cat):
    number_of_tails: int


class Dog(Pet):
    barks: float
    friend: Pet


class Person(Pet):
    name: str

    pets: list[Pet]
    f: Callable = lambda: print("hi")


exp = Person(
    name="jt",
    pets=[
        SpecialCat(name="sc", number_of_tails=2, meows=3),
        Dog(name="dog", barks=2, friend=Cat(name="cc", meows=2)),
    ],
)


def g():
    print("Hey")


def run(x: Person = exp):
    print(x)
    print(x.f())


from jsonargparse import auto_cli

if __name__ == "__main__":
    auto_cli(run)
