from collections import namedtuple
import numpy as np

from julax.layers.einops import Rearrange, Reduce
from einops import _backends, rearrange, reduce
from einops.tests import FLOAT_REDUCTIONS as REDUCTIONS

import pickle

testcase = namedtuple(
    "testcase", ["pattern", "axes_lengths", "input_shape", "wrong_shapes"]
)


rearrangement_patterns = [
    testcase(
        "b c h w -> b (c h w)",
        dict(c=20),
        (10, 20, 30, 40),
        [(), (10,), (10, 10, 10), (10, 21, 30, 40), [1, 20, 1, 1, 1]],
    ),
    testcase(
        "b c (h1 h2) (w1 w2) -> b (c h2 w2) h1 w1",
        dict(h2=2, w2=2),
        (10, 20, 30, 40),
        [(), (1, 1, 1, 1), (1, 10, 3), ()],
    ),
    testcase(
        "b ... c -> c b ...",
        dict(b=10),
        (10, 20, 30),
        [(), (10,), (5, 10)],
    ),
]


def test_rearrange_imperative():
    backend = _backends.JaxBackend()
    print("Test layer for ", backend.framework_name)

    for pattern, axes_lengths, input_shape, wrong_shapes in rearrangement_patterns:
        x = np.arange(np.prod(input_shape), dtype="float32").reshape(input_shape)
        result_numpy = rearrange(x, pattern, **axes_lengths)
        layer = Rearrange(pattern, **axes_lengths)
        for shape in wrong_shapes:
            try:
                layer(backend.from_numpy(np.zeros(shape, dtype="float32")))
            except BaseException:
                pass
            else:
                raise AssertionError("Failure expected")

        # simple pickling / unpickling
        layer2 = pickle.loads(pickle.dumps(layer))
        result1 = backend.to_numpy(layer(backend.from_numpy(x))[0])
        result2 = backend.to_numpy(layer2(backend.from_numpy(x))[0])
        assert np.allclose(result_numpy, result1)
        assert np.allclose(result1, result2)


reduction_patterns = [
    *rearrangement_patterns,
    testcase("b c h w -> b ()", dict(b=10), (10, 20, 30, 40), [(10,), (10, 20, 30)]),
    testcase(
        "b c (h1 h2) (w1 w2) -> b c h1 w1",
        dict(h1=15, h2=2, w2=2),
        (10, 20, 30, 40),
        [(10, 20, 31, 40)],
    ),
    testcase("b ... c -> b", dict(b=10), (10, 20, 30, 40), [(10,), (11, 10)]),
]


def test_reduce_imperative():
    backend = _backends.JaxBackend()
    print("Test layer for ", backend.framework_name)
    for reduction in REDUCTIONS:
        for pattern, axes_lengths, input_shape, wrong_shapes in reduction_patterns:
            print(backend, reduction, pattern, axes_lengths, input_shape, wrong_shapes)
            x = np.arange(1, 1 + np.prod(input_shape), dtype="float32").reshape(
                input_shape
            )
            if reduction == "prod":
                # make numbers smaller to avoid overflow
                x = x / x.astype("float64").mean() / 10
            else:
                x = x / x.astype("float64").mean()
            result_numpy = reduce(x, pattern, reduction, **axes_lengths)
            layer = Reduce(pattern, reduction, **axes_lengths)
            for shape in wrong_shapes:
                try:
                    layer(backend.from_numpy(np.zeros(shape, dtype="float32")))
                except BaseException:
                    pass
                else:
                    raise AssertionError("Failure expected")

            # simple pickling / unpickling
            layer2 = pickle.loads(pickle.dumps(layer))
            result1 = backend.to_numpy(layer(backend.from_numpy(x))[0])
            result2 = backend.to_numpy(layer2(backend.from_numpy(x))[0])
            assert np.allclose(result_numpy, result1)
            assert np.allclose(result1, result2)
