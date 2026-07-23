# Solver hooks

A [**solver**](../../concepts/glossary.md#solver) is a stateful object — not a plain
hook function — that performs the actual per-group or per-iteration update of
reconstruction variables. Three distinct, unrelated protocols share the name
"solver" (`phaser/hooks/solver.py`), each configured by its own plan field:

| Solver kind | Plan field | Engine | Registered names |
| --- | --- | --- | --- |
| **Conventional solver** | `ConventionalEnginePlan.solver` | conventional only | `epie`, `lsqml` |
| **Gradient solver** | `GradientEnginePlan.solvers` (a dict keyed by variable set) | gradient only | `sgd`, `adam`, `polyak_sgd` |
| **Position solver** | `ConventionalEnginePlan.position_solver` (optional) | conventional only | `steepest_descent`, `momentum` |

The gradient engine has **no separate position-solver field** — it optimizes scan
positions by assigning an ordinary gradient solver (e.g. `sgd` or `adam`) to the
`positions` variable in `solvers`, exactly like `object`, `probe`, or `tilt`. Position
solvers (`steepest_descent`, `momentum`) exist only for the conventional engines, where
ePIE/LSQML compute a raw position-gradient step internally and a separate
`position_solver` turns it into an actual position update — confirmed by every gradient
example under `examples/` assigning `positions` inside `solvers:` (e.g.
`examples/czo_grad.yaml:49-52`, `examples/si_grad.yaml:46-50`) and every conventional
`lsqml`/`epie` example using a top-level `position_solver:` field instead
(`examples/mos2_lsqml.yaml:39-42`).

!!! warning "Restriction — position solving is under review"
    Position-solving code is under code-owner review (checklist blockers B7 and B8); this
    page documents its lifecycle and interface honestly but does **not** present it as
    fully supported. Two verified issues:

    - **ePIE's position-gradient computation ignores its `update_position` flag.**
      `EPIESolver.run_iteration` never passes `update_position` to `epie_run`
      (`phaser/engines/conventional/solvers.py:389-398`), unlike `LSQMLSolver`, which does
      (`solvers.py:118`). `epie_run`'s `update_position` parameter therefore always
      defaults to `True` and the position-gradient step is always computed — whether it is
      actually *applied* is still correctly gated by `update_positions` one level up
      (`solvers.py:404-406`), so this wastes compute rather than producing wrong results,
      but ePIE's position support should not be presented as equivalent to LSQML's.
    - **The built-in `steepest_descent` position solver does not read `max_step_size` at
      all.** `SteepestDescentPositionSolver.__init__` assigns `self.max_step_size =
      props.step_size` — not `props.max_step_size`
      (`phaser/engines/common/position_correction.py:14-16`), demonstrated below in
      [Built-in implementations](#built-in-implementations). `steepest_descent`'s
      `max_step_size` field is undocumented here pending a code-owner decision (fix the
      assignment, or remove the field) — do not rely on it doing anything under
      `steepest_descent` until resolved. `momentum` (`MomentumPositionSolver`) reads
      `max_step_size` correctly (`position_correction.py:38-41`) and is unaffected.

## Lifecycle point

- **Conventional solver** — constructed once per engine run:
  `solver = props.solver(props)` (`phaser/engines/conventional/run.py:52`), then
  `sim = solver.init(sim)` immediately (one-time setup, e.g. LSQML's magnitude
  accumulators). `solver.presolve(...)` runs once before the iteration loop, to precompute
  normalization state and rescale the initial probe intensity (`conventional/run.py:81-85`).
  `solver.run_iteration(...)` then runs once per engine iteration, looping over every group
  internally (`conventional/run.py:97-106`).
- **Gradient solver** — constructed once per engine run via `process_solvers`
  (`phaser/engines/gradient/run.py:31-67,166`), which splits the `solvers` dict into
  **per-group** solvers (variable set disjoint from `{'positions', 'tilt'}`) and
  **per-iteration** solvers (variable set a subset of `{'positions', 'tilt'}`). `init_state`
  runs once at construction (`SolverStates.init_state`, `gradient/run.py:143-150,231`).
  `update_for_iter` runs once per iteration, before that iteration's groups, to resolve any
  schedule-valued hyperparameters (`gradient/run.py:268-275`). A per-group solver's `update`
  runs once per group, inside the JIT-traced `run_group` (`gradient/run.py:398-405`); a
  per-iteration solver's `update` runs once per iteration, after all of that iteration's
  groups (`gradient/run.py:312-319`).
- **Position solver** — constructed once per conventional-engine run:
  `position_solver = props.position_solver(None)` if configured
  (`conventional/run.py:58`), with `init_state` called immediately after
  (`conventional/run.py:59`). `perform_update` runs once per iteration, only when that
  iteration's `update_positions` flag is true, and only after the conventional solver's
  `run_iteration` has already produced a raw, mean-subtracted position-gradient step
  (`conventional/run.py:113-124`).

## Callable signature and property schema

### Conventional solver

`ConventionalSolverHook` is `Hook['ConventionalEnginePlan', ConventionalSolver]`
(`phaser/hooks/solver.py:144-145`) — unusually among hook families, it is called with the
**whole `ConventionalEnginePlan`** as its argument (`solver(props)`,
`conventional/run.py:52`), not `None` or a small TypedDict. `ConventionalSolver`
(`phaser/hooks/solver.py:102-141`) is an abstract base class:

```python
class ConventionalSolver(abc.ABC):
    @classmethod
    def name(cls) -> str: ...
    def init(self, sim: SimulationState) -> SimulationState: ...
    def presolve(
        self, sim: SimulationState, groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns, pattern_mask, propagators,
    ) -> SimulationState: ...
    def run_iteration(
        self, sim: SimulationState, groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns, pattern_mask, propagators,
        update_object: bool, update_probe: bool, update_positions: bool,
        calc_error: bool, calc_error_mask, observer,
    ) -> t.Tuple[SimulationState, NDArray, t.List[NDArray]]: ...
```

Properties (`phaser/plan.py:120-138`), every field `ScheduleLike` (a plain float or a
[schedule hook](schedules-and-flags.md)):

| Hook | Field | Default | Meaning |
| --- | --- | --- | --- |
| `lsqml` | `stochastic` | `True` | Declared in `LSQMLSolverPlan` but not read anywhere in `phaser/engines/conventional/solvers.py` — appears unused by the current implementation. |
| `lsqml` | `beta_object`, `beta_probe` | `1.0` | Step-size scaling for the object/probe update. |
| `lsqml` | `illum_reg_object`, `illum_reg_probe` | `1e-2` | Regularizes the illumination-magnitude division in the object/probe update (avoids dividing by a near-zero probe/object magnitude). |
| `lsqml` | `gamma` | `1e-4` | Regularizes the per-mode step-length estimate `alpha`. |
| `epie` | `beta_object`, `beta_probe` | `1.0` | Step-size scaling for the object/probe update. |

### Gradient solver

`GradientSolverHook` is `Hook['GradientSolverArgs', GradientSolver]`
(`phaser/hooks/solver.py:164-170`), called as `solver({'plan': plan, 'params': vars})`
(`gradient/run.py:51,60`) where `vars` is the `frozenset[ReconsVar]` this solver was
assigned in the plan's `solvers` mapping. `GradientSolver` (`phaser/hooks/solver.py:148-161`)
is a structural protocol:

```python
class GradientSolver(t.Protocol[StateT]):
    name: str
    params: t.FrozenSet[ReconsVar]
    def init_state(self, sim: ReconsState) -> StateT: ...
    def update_for_iter(self, sim: ReconsState, state: StateT, niter: int) -> StateT: ...
    def update(
        self, sim: ReconsState, state: StateT, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], StateT]: ...
```

Properties (`phaser/plan.py:159-181`; every schedule-typed field is `ScheduleLike`):

| Hook | Fields | Notes |
| --- | --- | --- |
| `sgd` | `learning_rate` (required), `momentum: Optional = None`, `nesterov: bool = True` | With `momentum` set, applies Optax-style momentum accumulation, optionally Nesterov-corrected, before scaling by `learning_rate`. |
| `adam` | `learning_rate` (required), `b1=0.9`, `b2=0.999`, `eps=1e-8`, `eps_root=0.0`, `nesterov=False` | Standard Adam moment estimates, bias-corrected, optionally Nesterov-corrected. |
| `polyak_sgd` | `max_learning_rate` (required), `f_min` (required), `scaling=1.0`, `eps=0.0` | Polyak step size: `min(gap / (‖grad‖² + eps), max_learning_rate)` where `gap = loss - f_min`, then scaled by `scaling`. |

`SGDSolverPlan`/`AdamSolverPlan`/`PolyakSGDSolverPlan` are registered in
`GradientSolverHook.known` (`phaser/plan.py:183-185`). The implementations are Phaser's own
reimplementation of the corresponding [Optax](https://github.com/google-deepmind/optax)
transforms over Phaser's generic array/pytree utilities (`phaser/engines/gradient/solvers.py`
module docstring) — **no `optax` package dependency is imported or required**.

### Position solver

`PositionSolverHook` is `Hook[None, PositionSolver]` (`phaser/hooks/solver.py:95-99`),
called as `props.position_solver(None)` (`conventional/run.py:58`). `PositionSolver`
(`phaser/hooks/solver.py:66-76`):

```python
class PositionSolver(HasState[StateT], t.Protocol[StateT]):
    def perform_update(
        self, positions: NDArray[numpy.floating], gradients: NDArray[numpy.floating], state: StateT,
    ) -> t.Tuple[NDArray[numpy.floating], StateT]: ...
```

Properties (`phaser/hooks/solver.py:79-93`):

| Hook | Fields | Notes |
| --- | --- | --- |
| `steepest_descent` | `step_size: float = 1e-2`, `max_step_size: Optional[float] = None` | **`max_step_size` is not read** — see the restriction above. |
| `momentum` | `step_size: float = 1e-2`, `max_step_size: Optional[float] = None`, `momentum: float = 0.9` | `max_step_size` is read and applied correctly. |

## Accepted state and returned value

- **Conventional solver** — `sim: SimulationState`
  (`phaser/engines/common/simulation.py:89-134`) bundles a `ReconsState` with the engine's
  noise model, group/iteration constraints, `xp`, and `dtype`. `groups` yields integer index
  arrays selecting which scan positions belong to each group. `propagators` is either `None`
  (single-slice) or a complex array of shape `(n_slices-1, y, x)` for multislice free-space
  propagation. `run_iteration` returns `(sim, pos_update, iter_errors)`: an updated
  `SimulationState`; `pos_update`, a `(n_pos, 2)` array of raw position-gradient steps
  (`[y, x]`, meaningful only when `update_positions` was true); and `iter_errors`, a list of
  per-group error arrays.
- **Gradient solver** — `update` receives `grad`, a dict keyed by whichever `ReconsVar`s
  this solver owns, mapping to arrays shaped like the corresponding state field: `object`
  updates are complex `(z, y, x)`; `probe` updates are complex `(modes, y, x)`; `positions`
  and `tilt` updates are real `(n_pos, 2)` (`[y, x]` or `[ty, tx]` in mrad for tilt). It
  returns an `update` dict in the same shapes, which `apply_update`
  (`gradient/run.py:107-121`) adds directly onto `state.object.data` /
  `state.probe.data` / `state.scan` / `state.tilt` (subtracting the mean update first for
  `positions`).
- **Position solver** — `perform_update(positions, gradients, state)`: both `positions` and
  `gradients` are real arrays shaped `(n_pos, 2)` (`[y, x]`, physical units matching the
  scan, typically Å); it returns `(update, state)` in the same shape, added onto
  `sim.state.scan` by the caller (`conventional/run.py:119-124`), after a second
  mean-subtraction.

## Built-in implementations

Registered in `ConventionalSolverHook.known`/`GradientSolverHook.known`/
`PositionSolverHook.known` (`phaser/plan.py:137-138,183-185`; `phaser/hooks/solver.py:96-99`):

| Family | Name | Class | One-line description |
| --- | --- | --- | --- |
| Conventional | `epie` | `EPIESolver` | Extended ptychographic iterative engine: per-group object/probe update from a normalized conjugate-gradient-like step. |
| Conventional | `lsqml` | `LSQMLSolver` | Least-squares maximum-likelihood solver: per-group update using accumulated illumination-magnitude normalization (`obj_mag`/`probe_mag`) from the previous iteration. |
| Gradient | `sgd` | `SGDSolver` | Fixed or scheduled learning rate, optional momentum (Nesterov or plain). |
| Gradient | `adam` | `AdamSolver` | Adam moment-based adaptive learning rate, optional Nesterov correction. |
| Gradient | `polyak_sgd` | `PolyakSGDSolver` | Polyak step size using the current loss value and a target `f_min`. |
| Position | `steepest_descent` | `SteepestDescentPositionSolver` | `update = step_size * gradients`, magnitude-capped at `max_step_size` — except the cap is currently always `step_size` itself (confirmed bug, see restriction). |
| Position | `momentum` | `MomentumPositionSolver` | Adds a momentum term (`momentum * previous_update`) before the same magnitude cap, which is correctly read from `max_step_size`. |

Executed evidence for the `steepest_descent` bug (`phaser/engines/common/position_correction.py:14-16`):

```python
from phaser.engines.common.position_correction import SteepestDescentPositionSolver
from phaser.hooks.solver import SteepestDescentPositionSolverProps
import numpy

# User requests small steps (step_size=0.01) with a generous cap (max_step_size=10.0)
props = SteepestDescentPositionSolverProps(step_size=0.01, max_step_size=10.0)
solver = SteepestDescentPositionSolver(None, props)
print(solver.max_step_size)  # 0.01 -- NOT 10.0

positions = numpy.zeros((3, 2), dtype=numpy.float32)
gradients = numpy.array([[100.0, 0.0], [0.0, 50.0], [1.0, 1.0]], dtype=numpy.float32)
update, _ = solver.perform_update(positions, gradients, None)
print(numpy.linalg.norm(update, axis=-1))  # [0.01 0.01 0.01] -- every step capped at step_size, not 10.0
```

This ran exactly as shown: `solver.max_step_size` printed `0.01` (the `step_size` value,
not the requested `10.0`), and every returned update had magnitude `0.01` regardless of the
larger gradients and requested cap.

## Minimal custom implementation

A custom **gradient solver** needs no `ReconsState` at all to test the update rule itself
— it operates on plain dicts of arrays. This fixed-step solver was executed directly:

```python
import typing as t
import numpy

from phaser.types import ReconsVar, Dataclass
from phaser.state import ReconsState


class FixedStepSolverProps(Dataclass):
    learning_rate: float = 1e-2


class FixedStepSolver:
    """A minimal GradientSolver: a plain fixed learning rate applied to
    whichever ReconsVars it is assigned. Carries no state across calls."""

    def __init__(self, args, props: FixedStepSolverProps):
        self.learning_rate: float = props.learning_rate
        self.params: t.FrozenSet[ReconsVar] = frozenset(args['params'])
        self.name: str = 'fixed_step'

    def init_state(self, sim: ReconsState) -> None:
        return None

    def update_for_iter(self, sim: ReconsState, state: None, niter: int) -> None:
        return state

    def update(
        self, sim: ReconsState, state: None, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], None]:
        update = {k: -self.learning_rate * v for (k, v) in grad.items()}
        return (update, state)
```

Executed against synthetic gradients:

```python
solver = FixedStepSolver({'plan': None, 'params': {'object', 'probe'}}, FixedStepSolverProps(learning_rate=0.1))
state = solver.init_state(None)
state = solver.update_for_iter(None, state, niter=10)

grad = {
    'object': numpy.ones((2, 4, 4), dtype=numpy.complex64) * (1.0 + 0.5j),
    'probe': numpy.full((1, 4, 4), 2.0, dtype=numpy.complex64),
}
update, state = solver.update(None, state, grad, loss=0.123)
# update['object'][0,0,0] == -0.1-0.05j == -0.1 * grad['object']
# update['probe'][0,0,0]  == -0.2+0j    == -0.1 * grad['probe']
```

Both assertions (`update['object'] == -0.1 * grad['object']`, same for `probe`) held.

A custom **position solver** is equally standalone — it only needs `(n_pos, 2)` arrays.
This one clips every update to an explicit maximum magnitude on every call (unlike the
built-in `steepest_descent`'s currently-broken cap):

```python
import typing as t
import numpy
from numpy.typing import NDArray

from phaser.utils.num import get_array_module
from phaser.state import ReconsState
from phaser.types import Dataclass


class ClippedStepPositionSolverProps(Dataclass):
    step_size: float = 1e-2
    max_step_size: float = 0.5


class ClippedStepPositionSolver:
    def __init__(self, args: None, props: ClippedStepPositionSolverProps):
        self.step_size = props.step_size
        self.max_step_size = props.max_step_size

    def init_state(self, sim: ReconsState) -> None:
        return None

    def perform_update(
        self, positions: NDArray[numpy.floating], gradients: NDArray[numpy.floating], state: None,
    ) -> t.Tuple[NDArray[numpy.floating], None]:
        xp = get_array_module(positions, gradients)
        update = self.step_size * gradients
        update_mag = xp.linalg.norm(update, axis=-1, keepdims=True)
        update = update * xp.minimum(update_mag, self.max_step_size) / xp.maximum(update_mag, 1e-12)
        return (update, state)
```

Executed with `step_size=1.0, max_step_size=0.5` against gradients of magnitude up to
~7 (`[-5, 5]`): every returned update had magnitude `<= 0.5`, confirming the cap applies
(magnitudes printed: `[0.5, 0.01, 0.5, 0.0, 0.5]` for five test gradients of varying size).

A custom **conventional solver** is not exercised here: implementing `ConventionalSolver`
requires a working `SimulationState` (a real `ReconsState`, resolved noise model, and
propagators) — a substantially larger undertaking than the other two solver kinds.
Described at the signature level only (above), not executed for this page.

## YAML invocation

Built-in gradient solver, multi-variable key (parses via the comma-separated
`ReconsVars` converter, `phaser/types.py:364-386`):

```yaml
engines:
  - type: gradient
    noise_model: {type: amplitude}
    solvers:
      "object, probe":
        type: adam
        learning_rate: 1.0e-2
      positions:
        type: sgd
        learning_rate: 1.0e-3
```

Validated directly against `ReconsPlan.from_data`: it parses, and
`plan.engines[0].props.solvers.keys()` is
`[frozenset({'object', 'probe'}), frozenset({'positions'})]` — confirming the
comma-separated syntax and that a single key maps to one shared solver instance for all
listed variables.

Built-in conventional solver and position solver:

```yaml
engines:
  - type: conventional
    noise_model: {type: anscombe}
    solver:
      type: lsqml
      beta_object: 1.0
      beta_probe: 1.0
    position_solver:
      type: momentum
      step_size: 8.0e-2
      momentum: 0.9
    update_positions: {after: 10}
```

External, by `package.module:function` reference — properties are passed straight
through as an unvalidated `dict`:

```yaml
engines:
  - type: gradient
    noise_model: {type: amplitude}
    solvers:
      "object, probe":
        type: "my_package.my_module:FixedStepSolver"
        learning_rate: 0.05
```

**Rejected configurations**, verified by calling `process_solvers` directly
(`phaser/engines/gradient/run.py:31-67`):

- **Duplicate variable assignment** — two solver entries both claiming `object` raises
  `ValueError("Duplicate solvers for variable(s) 'object'.")`.
- **A single solver mixing a per-group variable with a per-iteration variable** — a solver
  assigned `{"object", "positions"}` raises
  `ValueError("The same solver can't handle both per-iteration ('positions') and per-group ('object') variables")`.

## Engine and backend restrictions

- Conventional solver hooks (`epie`, `lsqml`) and position solver hooks
  (`steepest_descent`, `momentum`) only exist on `ConventionalEnginePlan`; there is no
  equivalent field on `GradientEnginePlan`.
- Gradient solver hooks (`sgd`, `adam`, `polyak_sgd`) only exist on
  `GradientEnginePlan.solvers`; the gradient engine additionally requires the JAX or Torch
  backend ([Engine families and backends](../overview.md#engine-families-and-backends)) —
  the engine's restriction, not the solver's, but it applies to every gradient solver by
  construction, since none can run without the gradient engine.
- A gradient solver's assigned variables must be entirely per-group (`object`, `probe`) or
  entirely per-iteration (`positions`, `tilt`) — never both — and no variable may be
  claimed by more than one solver (verified above).
- Position solvers are under code-owner review (B7/B8) — see the restriction admonition at
  the top of this page.

## Optional dependencies

None of the built-in solvers require an optional package: the gradient solvers reimplement
their Optax-derived math directly over Phaser's own array/pytree utilities (no `optax`
import), and the conventional/position solvers use only NumPy-compatible array operations.
Running a gradient solver at all still requires the gradient engine's JAX-or-Torch backend
requirement (`pyproject.toml`'s `jax`/`torch` optional dependency groups).

## Testing pattern

- **Gradient solver** — construct it directly with a `GradientSolverArgs`-shaped dict
  (`{'plan': None or a real plan, 'params': {...}}`), call `init_state`/`update_for_iter`
  with `sim=None` if your solver does not read `sim`, then call `update` with small
  synthetic gradient arrays keyed by the variables you assigned, and assert the returned
  update matches your solver's expected closed-form output — exactly as executed above.
- **Position solver** — construct it directly with its properties dataclass, call
  `init_state(None)`, then `perform_update` with small `(n_pos, 2)` position/gradient
  arrays, and assert the returned update's per-position magnitude respects your solver's
  intended cap or scaling — exactly as executed above. This is also the pattern that
  exposed the `steepest_descent` bug: construct the solver with `step_size` and
  `max_step_size` set to two clearly different values and check which one the returned
  magnitude actually respects.
- **Conventional solver** — not demonstrated here; the pattern would construct a small
  `SimulationState` directly (a minimal `ReconsState` with a tiny object/probe/scan, a
  resolved noise model, and `propagators=None` for single-slice), call `init`, then
  `presolve` with a one-group iterator, then `run_iteration` with the same, and assert the
  returned object/probe data changed in the expected direction. Not exercised for this
  page — unverified guidance, not a confirmed recipe.

## Maintainer sources

- `phaser/hooks/solver.py`
- `phaser/plan.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/gradient/solvers.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/engines/common/position_correction.py`
- `phaser/engines/common/simulation.py`
- `phaser/types.py`
- `examples/mos2_lsqml.yaml`
- `examples/czo_grad.yaml`
- `examples/si_grad.yaml`
