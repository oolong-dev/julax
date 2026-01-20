import logging
import time
from typing import Protocol

import jax
from julax.base import Param, State
from .experiment import Experiment

logger = logging.getLogger(__name__)


class Observer(Protocol):
    def __call__(self, step: int, exp: Experiment, param: Param, state: State): ...


class ObserverBase:
    def __call__(self, step: int, exp: Experiment, param: Param, state: State):
        raise NotImplementedError

    def __mul__(self, other: Observer) -> "CompositeObserver":
        return CompositeObserver(self, other)

    def __rmul__(self, other: Observer) -> "CompositeObserver":
        return CompositeObserver(other, self)


class CompositeObserver(ObserverBase):
    def __init__(self, *observers: Observer):
        self.observers = []
        for obs in observers:
            if isinstance(obs, CompositeObserver):
                self.observers.extend(obs.observers)
            else:
                self.observers.append(obs)

    def __call__(self, step: int, exp: Experiment, param: Param, state: State):
        for observer in self.observers:
            observer(step, exp, param, state)


@jax.jit
def _get_loss(state: State) -> float:
    return state["loss"]


class LogLossEveryNSteps(ObserverBase):
    def __init__(self, n: int = 100, first_n: int = 10):
        self.n = n
        self.first_n = first_n

    def __call__(self, step: int, exp: Experiment, param: Param, state: State):
        if step > 0 and (step <= self.first_n or step % self.n == 0):
            logger.info("Step %s: loss=%s", step, _get_loss(state))


class LogAvgStepTime(ObserverBase):
    def __init__(self, n: int = 100):
        self.n = n
        self.last_time = None
        self.step_count = 0

    def __call__(self, step: int, exp: Experiment, param: Param, state: State):
        if self.last_time is None:
            self.last_time = time.perf_counter()
            self.step_count = 0
            return

        self.step_count += 1

        if self.step_count % self.n == 0:
            now = time.perf_counter()
            avg_time = (now - self.last_time) / self.step_count
            logger.info(
                f"Step {step}: avg step time over last {self.step_count} steps: {avg_time:.6f}s"
            )
            self.last_time = now
            self.step_count = 0


class ProfileAtSteps(ObserverBase):
    def __init__(
        self,
        dir: str,
        steps: list[int] | None = None,
        span: int = 3,
        is_rank0_only: bool = True,
    ):
        steps = steps or [3]
        self.start_steps = set(x for x in steps)
        self.stop_steps = set(x + span for x in steps)
        self.dir = dir
        self.is_rank0_only = is_rank0_only

    def __call__(self, step: int, exp: Experiment, param: Param, state: State):
        if self.is_rank0_only and jax.process_index() != 0:
            return

        if step in self.start_steps:
            logger.info(f"Start profiling from step {step}...")
            jax.profiler.start_trace(self.dir)
        if step in self.stop_steps:
            jax.tree.map(lambda x: x.block_until_ready(), (param, state))
            jax.profiler.stop_trace()
            logger.info(f"Stopped profiling at step {step}.")


def default_observer() -> CompositeObserver:
    return LogLossEveryNSteps() * LogAvgStepTime()
