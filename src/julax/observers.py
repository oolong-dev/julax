import logging
import time
from typing import Protocol

import jax
from .core import Param, State

logger = logging.getLogger(__name__)


class Observer(Protocol):
    def __call__(self, x, p: Param, s: State): ...


class ObserverBase:
    def __call__(self, x, p: Param, s: State):
        raise NotImplementedError

    def __mul__(self, other: Observer) -> "CompositeObserver":
        return CompositeObserver([self, other])

    def __rmul__(self, other: Observer) -> "CompositeObserver":
        return CompositeObserver([other, self])


class DoEveryNSteps(ObserverBase):
    def __init__(self, observer: Observer, n: int = 1):
        self.n = n
        self.observer = observer

    def __call__(self, x, p: Param, s: State):
        step = s["step"]
        if step % self.n == 0:
            return self.observer(x, p, s)


class DoAtStep0(ObserverBase):
    def __init__(self, observer: Observer):
        self.observer = observer

    def __call__(self, x, p: Param, s: State):
        step = s["step"]
        if step == 0:
            return self.observer(x, p, s)


class CompositeObserver(ObserverBase):
    def __init__(self, observers: list[Observer]):
        self.observers = []
        for obs in observers:
            if isinstance(obs, CompositeObserver):
                self.observers.extend(obs.observers)
            else:
                self.observers.append(obs)

    def __call__(self, x, p: Param, s: State):
        for observer in self.observers:
            observer(x, p, s)


class LossLogger(ObserverBase):
    def __call__(self, x, p: Param, s: State):
        loss = s["trainer"]["loss"]
        step = s["step"]
        jax.debug.print("Step {step}: loss={loss}", step=step, loss=loss)


class StepTimeLogger(ObserverBase):
    def __init__(self, n: int = 100):
        self.n = n
        self.last_time = None
        self.step_count = 0

    def __call__(self, x, p: Param, s: State):
        if self.last_time is None:
            self.last_time = time.perf_counter()
            self.step_count = 0
            return

        self.step_count += 1

        if self.step_count % self.n == 0:
            now = time.perf_counter()
            avg_time = (now - self.last_time) / self.step_count
            step = s["step"]
            logger.info(
                f"Step {step}: avg step time over last {self.step_count} steps: {avg_time:.6f}s"
            )
            self.last_time = now
            self.step_count = 0


def default_observer() -> CompositeObserver:
    return DoEveryNSteps(LossLogger(), n=10) * StepTimeLogger()
