import logging
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

from phaser.utils.num import as_array, abs2
from phaser.hooks.solver import GradientSolver, GradientSolverArgs
from phaser.types import ReconsVar
from phaser.plan import GradientEnginePlan, AdamSolverPlan, FixedSolverPlan
from phaser.state import ReconsState
from .run import extract_vars, apply_update

import optax
from optax import GradientTransformation, GradientTransformationExtraArgs


class OptaxSolver(GradientSolver[t.Any]):
    def __init__(self, solver: GradientTransformation, params: t.Iterable[ReconsVar], name: t.Optional[str] = None):
        self.inner: GradientTransformationExtraArgs = optax.with_extra_args_support(solver)
        self.params: t.FrozenSet[ReconsVar] = frozenset(params)

        self.name: str = name or self.inner.__class__.__name__

    def init_state(self, sim: ReconsState) -> t.Any:
        return self.inner.init(extract_vars(sim, self.params)[0])

    def update(
        self, sim: 'ReconsState', state: t.Any, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], t.Any]:
        (updates, state) = self.inner.update(grad, state, params=extract_vars(sim, self.params)[0], value=loss, loss=loss)
        return (t.cast(t.Dict[ReconsVar, t.Any], updates), state)


class FixedSolver(OptaxSolver):
    def __init__(self, args: GradientSolverArgs, props: FixedSolverPlan):
        super().__init__(
            optax.scale_by_learning_rate(props.learning_rate, flip_sign=False),
            args['params'], "fixed"
        )


class AdamSolver(OptaxSolver):
    def __init__(self, args: GradientSolverArgs, props: AdamSolverPlan):
        super().__init__(
            optax.chain(
                optax.scale_by_adam(props.b1, props.b2, props.eps, props.eps_root, nesterov=props.nesterov),
                optax.scale_by_learning_rate(props.learning_rate, flip_sign=False),
            ),
            args['params'], "adam"
        )