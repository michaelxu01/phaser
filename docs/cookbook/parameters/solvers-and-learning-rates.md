# Solvers and learning rates

A [solver](../../concepts/glossary.md#solver) turns a computed gradient or ptychographic
update into a change to the object, probe, positions, or tilt. Phaser has three unrelated
solver families that share the name "solver" but apply to different engines and
variables — see [Solver hooks](../../architecture/hooks/solvers.md) for the full
interface. This page is about choosing *values* for each, not writing a new solver. The
gradient engine has no separate position-solver field: it optimizes scan positions (and
tilt) with an ordinary gradient solver assigned to the `positions` (or `tilt`) variable in
`solvers:`, exactly as it does for `object`/`probe`.

| Solver kind | Plan field | Engine | Built-in names |
| --- | --- | --- | --- |
| Gradient solver | `GradientEnginePlan.solvers` (dict keyed by variable set) | gradient only | `sgd`, `adam`, `polyak_sgd` |
| Conventional solver | `ConventionalEnginePlan.solver` | conventional only | `epie`, `lsqml` |
| Position solver | `ConventionalEnginePlan.position_solver` (optional) | conventional only | `steepest_descent`, `momentum` |

## Gradient solvers: keying `solvers` by variable

`GradientEnginePlan.solvers` is a `dict` keyed by sets of reconstruction variables —
`object`, `probe`, `positions`, `tilt` — written as a single variable or a
comma-separated string, with a gradient-solver hook as the value:

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

Here `object`/`probe` share one `adam` instance (one optimizer state for both) and
`positions` gets its own `sgd` solver — validated: `plan.engines[0].props.solvers.keys()`
returns `[frozenset({'object', 'probe'}), frozenset({'positions'})]`
(`docs/architecture/hooks/solvers.md#yaml-invocation`).

### Per-group vs. per-iteration cadence

`object`/`probe` are **per-group** (update runs once per group, inside the group loop);
`positions`/`tilt` are **per-iteration** (update runs once per iteration, after all of
that iteration's groups) (`phaser/engines/gradient/run.py:31-67`,
[Solver hooks — lifecycle](../../architecture/hooks/solvers.md#lifecycle-point)). This is
enforced, not just convention — one entry can't mix cadences, and no variable may be
claimed by two entries:

| Violation | Raises |
| --- | --- |
| One entry mixes per-group and per-iteration variables (e.g. `{object, positions}`) | `ValueError("The same solver can't handle both per-iteration ('positions') and per-group ('object') variables")` |
| Same variable claimed by two entries (e.g. `object` twice) | `ValueError("Duplicate solvers for variable(s) 'object'.")` |

## Gradient solver types

Phaser's own reimplementation of the matching [Optax](https://github.com/google-deepmind/optax)
transforms (no `optax` import, `phaser/engines/gradient/solvers.py` docstring);
types/defaults in the
[generated gradient-solver reference](../../generated/hooks/gradient-solver.md).

| Name | What it does | Key properties |
| --- | --- | --- |
| `sgd` | Fixed or scheduled learning rate, with optional (Nesterov-corrected) momentum | `learning_rate` (required, `float` or [schedule](schedules-and-flags.md)), `momentum` (optional, default `None`), `nesterov` (default `True`) |
| `adam` | Adam adaptive moment estimation, bias-corrected, optional Nesterov correction | `learning_rate` (required), `b1=0.9`, `b2=0.999`, `eps=1e-8`, `eps_root=0.0`, `nesterov=False` |
| `polyak_sgd` | Polyak step size: scales the step by how far the current loss is above a target `f_min` | `max_learning_rate` (required), `f_min` (required), `scaling=1.0`, `eps=0.0` |

`learning_rate`/`momentum`/`scaling` are schedule-valued (plain float or
[schedule](schedules-and-flags.md)); no enforced bound, and values below are
example-evidenced, not asserted optimal. A learning rate has no fixed physical unit: it
scales with the target variable (object/probe amplitude, or position/tilt in Å/mrad) and
how the loss is scaled (see
[Noise models](noise-models.md#prerequisite-patterns-must-be-scaled-to-physical-counts)).
Gradient solvers require `GradientEnginePlan` plus JAX or Torch (see
[Choosing a reconstruction engine](../engine-selection.md)); the update runs once per
group (`object`/`probe`) or once per iteration (`positions`/`tilt`), with schedule-valued
properties re-evaluated once per iteration (`update_for_iter`).

### Typical values seen in `examples/`

| Variable | Solver | Learning rate | Source |
| --- | --- | --- | --- |
| `object` | `adam` (`nesterov: true`) | `1.0e-3` – `7.0e-2` | `examples/si_grad.yaml` (`5.0e-3`), `examples/prsco3_grad.yaml` (`5.0e-3`, then `1.0e-3` in a second engine stage), `examples/si_grad_exp.yaml` (`7.0e-2`), `examples/mos2_grad.yaml` (`1.0e-2`) |
| `probe` | `adam` (`nesterov: true`) | `1.0e-2` – `0.2` | `examples/si_grad.yaml` (`0.1`), `examples/prsco3_grad.yaml` (`0.1`, then `0.2`), `examples/mos2_grad.yaml` (`1.0e-2`), `examples/czo_grad.yaml` (`0.01`, lower than object "cannot go too high" per its comment) |
| `positions` | `sgd` (`momentum: 0.90`, `nesterov: true`) | `0.5` | `examples/si_grad.yaml`, `examples/prsco3_grad.yaml`, `examples/si_grad_exp.yaml` (all `0.5`, `momentum: 0.90`) |
| `positions` | `adam` (`nesterov: true`) | `0.05` | `examples/czo_grad.yaml` (comment: "cannot go too high") |
| `tilt` | `adam` (`nesterov: true`) | `1.0e-3` – `0.05` | `examples/czo_grad.yaml` (`0.05` when introduced late; `1.0e-3` once every variable is being refined together in a later engine stage) |

### Minimal example

```yaml
engines:
  - type: gradient
    noise_model: {type: poisson, eps: 2.0}
    solvers:
      object:
        type: adam
        learning_rate: 5.0e-3
        nesterov: true
      probe:
        type: adam
        learning_rate: 0.1
        nesterov: true
      positions:
        type: sgd
        learning_rate: 0.5
        momentum: 0.90
        nesterov: true
```

adapted from `examples/si_grad.yaml`.

## Conventional solvers: `epie` and `lsqml`

| Property | `epie` | `lsqml` | Meaning |
| --- | --- | --- | --- |
| `beta_object` | default `1.0` | default `1.0` | Step-size scaling for the object update |
| `beta_probe` | default `1.0` | default `1.0` | Step-size scaling for the probe update |
| `gamma` | — | default `1e-4` | Regularizes the per-mode step-length estimate `alpha` |
| `illum_reg_object` | — | default `1e-2` | Regularizes the illumination-magnitude division in the object update (avoids dividing by a near-zero probe magnitude) |
| `illum_reg_probe` | — | default `1e-2` | Regularizes the illumination-magnitude division in the probe update |
| `stochastic` | — | default `True` | Declared in the schema but not read anywhere in `phaser/engines/conventional/solvers.py` — appears to have no effect on the current implementation |

All five fields above are schedule-valued (`ScheduleLike`, see
[Schedules and flags](schedules-and-flags.md)) dimensionless step-size scale factors on
top of each solver's own normalized update rule, with no enforced bound; types/defaults
are in the
[generated conventional-solver reference](../../generated/hooks/conventional-solver.md).
`epie`/`lsqml` only exist on `ConventionalEnginePlan`, re-evaluated once per iteration
inside `run_iteration`, before that iteration's groups
(`docs/architecture/hooks/schedules-and-flags.md`). Typical values in `examples/` cluster
around `0.5`–`1.0` for `beta_object`/`beta_probe`; `examples/prsco3_lsqml.yaml` ramps both
from `0` toward `1` over the first few iterations using an `expr` schedule,
`'1.0 - np.exp(-i / 3)'` (see [Schedules and flags](schedules-and-flags.md) for the trust
warning on `expr` schedules).

### Minimal example

```yaml
engines:
  - type: conventional
    noise_model: {type: anscombe, eps: 1.0e-4}
    solver:
      type: lsqml
      beta_object: 1.0
      beta_probe: 1.0
      gamma: 1.0e-4
      illum_reg_object: 1.0e-2
      illum_reg_probe: 1.0e-2
```

adapted from `examples/mos2_lsqml.yaml`.

## Position solvers

!!! warning "Restriction — position solving is under review"
    Position-solving code is under code-owner review; this page documents its interface
    honestly, not as fully verified. Two specific, verified issues (see
    [Solver hooks](../../architecture/hooks/solvers.md) for full detail and executed
    evidence):

    - **ePIE's position-gradient computation ignores its own `update_position` flag** —
      `EPIESolver.run_iteration` never passes it through, so the position-gradient step is
      always computed internally regardless of the flag. Whether it is actually *applied*
      is still correctly gated elsewhere, so this wastes compute rather than producing an
      incorrect reconstruction — but ePIE's position support should not be presented as
      equivalent to LSQML's.
    - **The built-in `steepest_descent` position solver never reads its `max_step_size`
      field.** `SteepestDescentPositionSolver.__init__` assigns
      `self.max_step_size = props.step_size` — not `props.max_step_size` — so the per-step
      cap always equals `step_size` itself, never the larger cap a user configures.
      Confirmed by direct execution
      (`docs/architecture/hooks/solvers.md#built-in-implementations`). **Do not rely on
      `max_step_size` doing anything under `steepest_descent`** until this is resolved.
      `momentum` reads `max_step_size` correctly and is unaffected.

| Property | `steepest_descent` | `momentum` | Units |
| --- | --- | --- | --- |
| `step_size` | default `0.01` | default `0.01` | Å per unit of the raw position gradient |
| `max_step_size` | default `None`, **not actually read (see restriction)** | default `None`, read and applied correctly | Å, the per-step magnitude cap |
| `momentum` | — | default `0.9` | dimensionless |

Type/default details are in the
[generated position-solver reference](../../generated/hooks/position-solver.md).
`perform_update` runs once per iteration, only when `update_positions` (a
[flag](schedules-and-flags.md)) is true that iteration, after the conventional solver's
`run_iteration` has produced a raw, mean-subtracted position-gradient step — otherwise
the solver is constructed but never invoked. Guidance beyond the `examples/` values below
is pending; no test sweeps these.

### Typical values seen in `examples/`

| Solver | `step_size` | `momentum` | `max_step_size` | Source |
| --- | --- | --- | --- | --- |
| `momentum` | `8.0e-2` | `0.90` | `0.2` | `examples/mos2_lsqml.yaml`, `examples/prsco3_lsqml.yaml` |
| `momentum` | `1.0e-3` | `0.90` | `0.2` | `examples/si_lsqml.yaml` |

### Minimal example

```yaml
engines:
  - type: conventional
    noise_model: {type: anscombe}
    solver: {type: lsqml}
    position_solver:
      type: momentum
      step_size: 8.0e-2
      momentum: 0.90
      max_step_size: 0.2
    update_positions: {after: 30}
```

adapted from `examples/mos2_lsqml.yaml`.

## Maintainer sources

- `phaser/plan.py`
- `phaser/hooks/solver.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/gradient/solvers.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/engines/common/position_correction.py`
- `docs/architecture/hooks/solvers.md`
- `docs/generated/hooks/gradient-solver.md`
- `docs/generated/hooks/conventional-solver.md`
- `docs/generated/hooks/position-solver.md`
- `examples/si_grad.yaml`
- `examples/prsco3_grad.yaml`
- `examples/si_grad_exp.yaml`
- `examples/mos2_grad.yaml`
- `examples/czo_grad.yaml`
- `examples/mos2_lsqml.yaml`
- `examples/si_lsqml.yaml`
- `examples/prsco3_lsqml.yaml`
