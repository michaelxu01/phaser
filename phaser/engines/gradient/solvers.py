import logging
import typing as t

import numpy

import phaser.utils.tree as tree
from phaser.hooks.solver import GradientSolver, GradientSolverArgs
from phaser.hooks.schedule import ScheduleLike, Schedule
from phaser.types import ReconsVar, process_schedule
from phaser.plan import AdamSolverPlan, PolyakSGDSolverPlan, SGDSolverPlan
from phaser.state import ReconsState
from .run import extract_vars

#import optax
#from optax import GradientTransformation, GradientTransformationExtraArgs
#from optax.schedules import StatefulSchedule
OptState: t.TypeAlias = tree.Tree
Params: t.TypeAlias = tree.Tree
Updates: t.TypeAlias = Params

class TransformInitFn(t.Protocol):
    def __call__(self, params: Params) -> OptState:
        ...


class TransformUpdateFn(t.Protocol):
    def __call__(
      self, updates: Updates, state: OptState, params: t.Optional[Params] = None,
      **extra_args: t.Any,
  ) -> t.Tuple[Updates, OptState]:
        ...


class GradientTransformation(t.NamedTuple):
    init: TransformInitFn
    update: TransformUpdateFn


ScheduledSolverState: t.TypeAlias = t.Tuple[t.Any, t.Dict[str, t.Optional[float]]]


class ScheduledSolver(GradientSolver[ScheduledSolverState]):
    def __init__(self, name: str, factory: t.Callable[..., GradientTransformation], hyperparams: t.Mapping[str, ScheduleLike],
                 params: t.Iterable[ReconsVar]):
        self.factory: t.Callable[..., GradientTransformation] = factory
        #self.inner: GradientTransformationExtraArgs = optax.with_extra_args_support(solver)

        self.hyperparams: t.Dict[str, Schedule] = {k: process_schedule(v) for (k, v) in hyperparams.items()}
        self.params: t.FrozenSet[ReconsVar] = frozenset(params)

        self.name: str = name # or self.inner.__class__.__name__

    def init_state(self, sim: ReconsState) -> ScheduledSolverState:
        return (
            None,
            {k: None for (k, v) in self.hyperparams.items()},
        )

    def _resolve(self, hparams: t.Mapping[str, t.Optional[float]]) -> GradientTransformation:
        return self.factory(**{k: hparams[k] for k in self.hyperparams.keys()})

    def update_for_iter(self, sim: ReconsState, state: ScheduledSolverState, niter: int) -> ScheduledSolverState:
        hparams_state: t.Dict[str, t.Optional[float]] = {k: v({'state': sim, 'niter': niter}) for (k, v) in self.hyperparams.items()}
        return (
            self._resolve(hparams_state).init(params=extract_vars(sim, self.params)[0]) if state[0] is None else state[0],
            hparams_state
        )

    def update(
        self, sim: 'ReconsState', state: ScheduledSolverState, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], ScheduledSolverState]:
        (inner_state, hparams_state) = state
        (updates, inner_state) = self._resolve(hparams_state).update(
            grad, inner_state, params=extract_vars(sim, self.params)[0], value=loss, loss=loss
        )
        return (t.cast(t.Dict[ReconsVar, t.Any], updates), (inner_state, hparams_state))


class SGDSolver(ScheduledSolver):
    def __init__(self, args: GradientSolverArgs, props: SGDSolverPlan):
        hparams = {
            'learning_rate': props.learning_rate
        }

        if props.momentum is not None:
            hparams['momentum'] = props.momentum
            def factory(**kwargs: t.Any) -> GradientTransformation:
                return chain(
                    trace(kwargs['momentum'], props.nesterov),
                    scale_by_learning_rate(kwargs['learning_rate'], flip_sign=False),
                )
        else:
            def factory(**kwargs: t.Any) -> GradientTransformation:
                return scale_by_learning_rate(kwargs['learning_rate'], flip_sign=False)

        super().__init__('sgd', factory, hparams, args['params'])


class AdamSolver(ScheduledSolver):
    def __init__(self, args: GradientSolverArgs, props: AdamSolverPlan):
        hparams = {
            'learning_rate': props.learning_rate
        }

        def factory(**kwargs) -> GradientTransformation:
            return optax.chain(
                optax.scale_by_adam(props.b1, props.b2, props.eps, props.eps_root, nesterov=props.nesterov),
                optax.scale_by_learning_rate(learning_rate=kwargs['learning_rate'], flip_sign=False),
            )

        super().__init__('adam', factory, hparams, args['params'])


class PolyakSGDSolver(ScheduledSolver):
    def __init__(self, args: GradientSolverArgs, props: PolyakSGDSolverPlan):
        hparams = {
            'max_learning_rate': props.max_learning_rate,
            'scaling': props.scaling,
        }

        def factory(**kwargs) -> GradientTransformation:
            return optax.chain(
                optax.scale_by_learning_rate(kwargs['scaling'], flip_sign=False),
                optax.scale_by_polyak(
                    max_learning_rate=kwargs['max_learning_rate'], f_min=props.f_min,
                    eps=props.eps, #variant='sps',
                )
            )

        super().__init__('polyak_sgd', factory, hparams, args['params'])


def chain(
    *args: GradientTransformation
) -> GradientTransformation:
    init_fns = tuple(arg.init for arg in args)
    update_fns = tuple(arg.update for arg in args)

    def init_fn(params: Params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, params=None, **extra_args):
        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, params, **extra_args)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return GradientTransformation(init_fn, update_fn)


def trace(
    decay: float,
    nesterov: bool = False,
    accumulator_dtype: t.Optional[t.Any] = None,
) -> GradientTransformation:

    def init_fn(params):
        return tree.zeros_like(params, dtype=accumulator_dtype)

    def update_fn(updates: Updates, state: Updates, params=None, **extra_args: t.Any):
        del params
        f = lambda g, t: g + decay * t  # noqa: E731
        new_trace = tree.map(
            lambda g, t: None if g is None else f(g, t),
            updates,
            state.trace,
            is_leaf=lambda g: g is None,
        )
        updates = tree.map(f, updates, new_trace) if nesterov else new_trace
        new_trace = tree.cast(new_trace, accumulator_dtype)
        return updates, new_trace

    return GradientTransformation(init_fn, update_fn)


def scale_by_learning_rate(
    learning_rate: float, *,
    flip_sign: bool = False,
) -> GradientTransformation:
    if flip_sign:
        learning_rate *= -1

    def update_fn(updates: Updates, state: None, params=None, **extra_args: t.Any):
        del params
        updates = tree.map(lambda g: learning_rate * g, updates)
        return updates, state

    return GradientTransformation(lambda params: None, update_fn)