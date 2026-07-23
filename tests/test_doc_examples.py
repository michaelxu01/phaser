# type: ignore
"""
Tests for the minimal custom-hook and custom-observer examples shown in
docs/architecture/observers.md and docs/architecture/testing.md.

The implementations below are reproduced verbatim in those pages, so the
documented examples are exercised by these tests rather than merely described.
Runs on the numpy backend only, with no optional dependencies and no datasets.
"""

import typing as t
from dataclasses import dataclass

import numpy

from phaser.hooks import RawData
from phaser.observer import Observer
from phaser.utils.num import Sampling


# --- Minimal custom post_load hook (docs/architecture/testing.md) ---------

@dataclass
class SubtractDarkProps:
    """Properties for the 'subtract_dark' custom post_load hook."""
    dark: float = 0.0


def subtract_dark(raw_data: RawData, props: SubtractDarkProps) -> RawData:
    """Minimal custom post_load hook: subtract a constant dark count and
    clip negative intensities to zero. Matches the `Hook[RawData, RawData]`
    callable contract: called as `f(raw_data, props=props)`."""
    raw_data['patterns'] = numpy.clip(raw_data['patterns'] - props.dark, 0.0, None)
    return raw_data


def _make_raw_data(patterns: numpy.ndarray) -> RawData:
    return {
        'patterns': patterns,
        'mask': numpy.ones(patterns.shape[-2:], dtype=numpy.float32),
        'sampling': Sampling(patterns.shape[-2:], sampling=(1.0, 1.0)),
    }


def test_subtract_dark_hook():
    raw_data = _make_raw_data(
        numpy.array([[[[1.0, 2.0], [0.5, 10.0]]]], dtype=numpy.float32)
    )
    props = SubtractDarkProps(dark=1.0)

    result = subtract_dark(raw_data, props)

    assert result is raw_data
    numpy.testing.assert_allclose(
        result['patterns'], [[[[0.0, 1.0], [0.0, 9.0]]]]
    )


def test_subtract_dark_hook_clips_at_zero():
    raw_data = _make_raw_data(numpy.array([[[[0.0, 0.2]]]], dtype=numpy.float32))
    result = subtract_dark(raw_data, SubtractDarkProps(dark=1.0))
    numpy.testing.assert_allclose(result['patterns'], [[[[0.0, 0.0]]]])


# --- Minimal custom observer (docs/architecture/observers.md) -------------

class LossHistoryObserver(Observer):
    """Minimal custom observer: records the reported total_loss at every
    iteration and counts engines/groups seen. Implements only the events it
    needs; every other `Observer` method keeps its no-op default."""

    def __init__(self):
        self.losses: t.List[float] = []
        self.engines_started: int = 0
        self.groups_seen: int = 0
        self.finished: bool = False

    def init_engine(self, init_state, *, recons_name, plan, **kwargs):
        self.engines_started += 1

    def update_group(self, state, force: bool = False):
        self.groups_seen += 1

    def update_iteration(self, state, i: int, n: int, errors: t.Dict[str, float]):
        if (loss := errors.get('total_loss')) is not None:
            self.losses.append(loss)

    def finish_recons(self, state):
        self.finished = True


def test_loss_history_observer_event_sequence():
    """Drives LossHistoryObserver through a synthetic event sequence matching
    the verified call order (see docs/architecture/observers.md): init_recons,
    start_recons, then per engine init_engine/start_engine, update_group and
    update_iteration for each group/iteration, finish_engine; then
    finish_recons and close for the whole reconstruction."""
    observer = LossHistoryObserver()

    observer.init_recons(plan=None)
    observer.start_recons(init_state=None)

    observer.init_engine(None, recons_name='test', plan=None)
    observer.start_engine(None)

    for (i, loss) in enumerate([1.0, 0.5, 0.2], start=1):
        observer.update_group(None)
        observer.update_group(None)
        observer.update_iteration(None, i, 3, {'total_loss': loss})

    observer.finish_engine(None)
    observer.finish_recons(None)
    observer.close(None)

    assert observer.engines_started == 1
    assert observer.groups_seen == 6
    assert observer.losses == [1.0, 0.5, 0.2]
    assert observer.finished is True


def test_loss_history_observer_ignores_iterations_without_loss():
    observer = LossHistoryObserver()
    observer.update_iteration(None, 1, 1, {})
    assert observer.losses == []
