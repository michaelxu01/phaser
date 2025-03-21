import logging
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

from phaser.utils.num import as_array, abs2
from phaser.hooks.solver import GradientSolver
from phaser.types import ReconsVar
from phaser.plan import GradientEnginePlan, AdamSolverPlan, FixedSolverPlan
from phaser.state import ReconsState
from .run import select_params, update_with_params


def tree_zeros_like(tree: t.Any) -> t.Any:
    import jax
    return jax.tree.map(lambda x: jax.numpy.zeros_like(x), tree)


def tree_update_moment(updates: t.Any, moments: t.Any, decay: float, order: float):
    import jax

    return jax.tree.map(
        lambda g, t: (
            (1 - decay) * (g**order) + decay * t if g is not None else None
        ),
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


def tree_update_moment_per_elem_norm(updates: t.Any, moments: t.Any, decay: float, order: float) -> t.Any:
    import jax

    def orderth_norm(g: jax.Array):
        half_order = order / 2.0
        return abs2(g) ** half_order

    return jax.tree.map(
        lambda g, t: (
            (1 - decay) * orderth_norm(g) + decay * t if g is not None else None
        ),
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


def tree_add_scalar_mul(
    tree_x: t.Any, scalar: ArrayLike, tree_y: t.Any
) -> t.Any:
    import jax
    import jax.numpy

    scalar = as_array(scalar, jax.numpy)

    return jax.tree.map(
        lambda x, y: None if x is None else x + scalar.astype(x.dtype) * y,
        tree_x,
        tree_y,
        is_leaf=lambda x: x is None,
    )


class FixedSolver(GradientSolver[None]):
    def __init__(self, plan: GradientEnginePlan, props: FixedSolverPlan):
        self._params: t.FrozenSet[ReconsVar] = frozenset(props.params)
        self.learning_rate: float = props.learning_rate

    @property
    def params(self) -> t.AbstractSet[ReconsVar]:
        return self._params

    @classmethod
    def name(cls) -> str:
        return "fixed"

    def init_state(self, sim: ReconsState, xp: t.Any) -> None:
        return None

    def run_group(
        self, sim: ReconsState, state: None, grad: t.Dict[ReconsVar, t.Any],
        iter: int, loss: float, 
    ) -> t.Tuple[ReconsState, None]:
        params = select_params(sim, self._params)
        params = tree_add_scalar_mul(params, self.learning_rate, grad)
        return (update_with_params(sim, params), state)

    def run_iter(
        self, sim: ReconsState, state: None, grad: t.Dict[ReconsVar, t.Any],
        iter: int, loss: float, 
    ) -> t.Tuple[ReconsState, None]:
        return (sim, state)


class AdamState(t.TypedDict):
    mu: t.Any
    nu: t.Any
    i: t.Any


class AdamSolver(GradientSolver[AdamState]):
    def __init__(self, plan: GradientEnginePlan, props: AdamSolverPlan):
        self.params: t.FrozenSet[ReconsVar] = frozenset(props.params)

        self.learning_rate: float = props.learning_rate
        self.b1: float = props.b1
        self.b2: float = props.b2
        self.eps: float = props.eps
        self.eps_root: float = props.eps_root

    @classmethod
    def name(cls) -> str:
        return "adam"

    def init_state(self, sim: ReconsState, xp: t.Any) -> AdamState:
        params = select_params(sim, self.params)
        return {
            'mu': tree_zeros_like(params),
            'nu': tree_zeros_like(params),
            'i': xp.zeros((), dtype=numpy.uint64),
        }

    def run_group(
        self, sim: ReconsState, state: AdamState, grad: t.Dict[ReconsVar, t.Any],
        iter: int, loss: float, 
    ) -> t.Tuple[ReconsState, AdamState]:
        params = select_params(sim, self.params)

        mu = tree_update_moment(grad, state['mu'], self.b1, 1)
        nu = tree_update_moment_per_elem_norm(grad, state['nu'], self.b2, 2)

        mu_hat = tree_bias_correction(mu, b1, count_inc)

        updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )

        params = tree_add_scalar_mul(params, self.learning_rate, grad)
        return (update_with_params(sim, params), state)

    def run_iter(
        self, sim: ReconsState, state: AdamState, grad: t.Dict[ReconsVar, t.Any],
        iter: int, loss: float,
    ) -> t.Tuple[ReconsState, AdamState]:
        return (sim, state)