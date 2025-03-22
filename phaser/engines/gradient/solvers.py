import logging
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

from phaser.utils.num import as_array, abs2
from phaser.hooks.solver import GradientSolver, GradientSolverArgs
from phaser.types import ReconsVar
from phaser.plan import GradientEnginePlan, AdamSolverPlan, FixedSolverPlan
from phaser.state import ReconsState
from .run import select_vars, apply_update

import optax
from optax import GradientTransformation, GradientTransformationExtraArgs


class OptaxSolver(GradientSolver[t.Any]):
    def __init__(self, solver: GradientTransformation, params: t.Iterable[ReconsVar], name: t.Optional[str] = None):
        self.inner: GradientTransformationExtraArgs = optax.with_extra_args_support(solver)
        self._params: t.FrozenSet[ReconsVar] = frozenset(params)

        self._name: str = name or self.inner.__class__.__name__

    def name(self) -> str:
        return self._name

    @property
    def params(self) -> t.AbstractSet[ReconsVar]:
        return self._params

    def init_state(self, sim: ReconsState, xp: t.Any) -> t.Any:
        return self.inner.init(select_vars(sim, self._params))

    def update(
        self, sim: 'ReconsState', state: t.Any, grad: t.Dict[ReconsVar, t.Any], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, t.Any], t.Any]:
        (updates, state) = self.inner.update(grad, state, params=select_vars(sim, self._params), value=loss, loss=loss)
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