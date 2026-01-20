import jax
from jax.sharding import PartitionSpec

from julax.base import State, Param, dispatch
from .observers import default_observer, ObserverBase
from .experiment import Experiment


import logging

logger = logging.getLogger(__name__)


@dispatch
def run(exp: Experiment):
    return run(exp, default_observer())


@dispatch
def run(exp: Experiment, observer: ObserverBase) -> tuple[int, Param, State]:
    with exp.mesh as mesh:
        step, param, state, input_iter = exp.restore()
        observer(step, exp, param, state)

        compiled_step = None

        for x_local in input_iter:
            if exp.max_steps is not None and step >= exp.max_steps:
                logger.info(f"Reached max steps {exp.max_steps}, stopping training.")
                break
            x = jax.make_array_from_process_local_data(
                sharding=jax.sharding.NamedSharding(
                    mesh, PartitionSpec(exp.batch_axis_names)
                ),
                local_data=x_local,
            )

            if compiled_step is None:
                compiled_step = exp.precompile(x, param, state)

            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                param, state = compiled_step(x, param, state)
                step += 1

            exp.save(step, param, state, input_iter)
            observer(step, exp, param, state)
    exp.close()
    return step, param, state
