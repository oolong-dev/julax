import inspect

from julax import LayerBase


def test_smoke():
    assert inspect.isclass(LayerBase)
