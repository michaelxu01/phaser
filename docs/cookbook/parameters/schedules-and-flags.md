# Schedules and flags

Many plan fields that look like a fixed `true`/`false` or fixed number can instead vary
across an engine's iterations. A [flag](../../concepts/glossary.md#flag) produces a boolean
that turns on or off at a chosen iteration (for example, starting position refinement only
after iteration 20); a [schedule](../../concepts/glossary.md#schedule) produces a number
that changes over the run (for example, decaying a learning rate). Both are described in
full mechanical detail in
[Schedule and flag hooks](../../architecture/hooks/schedules-and-flags.md); this page is
about choosing values for them.

!!! danger "Trust warning"
    An **expression schedule** (`type: expr`) evaluates its `expr` string with Python's
    `eval` (`phaser/hooks/schedule.py:62-69`). The globals dictionary passed to `eval` does
    not restrict builtins, so the expression can reach `__import__`, `open`, `exec`, and
    anything else importable — not just the five names documented below. **A plan file that
    uses an expression schedule is equivalent to a script: running an untrusted plan is
    running untrusted code.** This matters even more if you submit plans to a shared web
    manager or worker (see `architecture/interfaces.md`) — anyone who can submit a plan can
    execute code as whoever runs it. Only use `expr` schedules in plans you trust or wrote
    yourself, and never copy one from an untrusted source without reading it first.

## Flags: `after`/`before`/`every`

A flag field accepts a plain `bool`, a `SimpleFlag` mapping, or an external
`"package.module:function"` reference. There are **no built-in named flag hooks** — the
flag registry is empty (`FlagHook.known`, `phaser/hooks/schedule.py:19`) — so `type:
some_name` is never meaningful for a flag field; only `SimpleFlag`'s three keys are.

```python
after: int = 0
every: int = 1
before: t.Optional[int] = None
```

A `SimpleFlag` is `True` on iteration `i` (the current engine's 1-indexed iteration,
`state.iter.engine_iter`) exactly when `i > after`, `i < before` (if `before` is set), and
`i - after` is a multiple of `every`. With every field left at its default, a `SimpleFlag`
is `True` on every iteration from `1` onward.

- **Type/default:** `after: int = 0`, `every: int = 1`, `before: Optional[int] = None`. See
  the [generated `SimpleFlag` reference](../../generated/plan/index.md#simpleflag).
- **Units:** iteration count, within the current engine (not the whole reconstruction —
  see [Termination and diagnostics](termination-and-diagnostics.md) for the engine-vs-total
  iteration distinction).
- **Valid range:** any non-negative integers; `after=0, every=1` is a no-op default that
  is true from iteration 1.
- **Lifecycle stage:** evaluated once per iteration, at the start of that iteration, inside
  a running engine.
- **Engines/backends:** identical on both engine families — every `FlagLike` field is
  declared once on the shared `EnginePlan` base
  (`phaser/plan.py:59,76-85`).

### Fields that accept a flag

`shuffle_groups`, `update_probe`, `update_object`, `update_positions`, `update_tilt`,
`calc_error`, `save`, `save_images` all accept a `FlagLike` value (plain `bool`,
`SimpleFlag`, or external hook).

!!! warning "Restriction"
    `update_tilt` is inherited by `ConventionalEnginePlan` from the shared `EnginePlan` base
    but has no effect there: only the gradient engine reads it. A conventional engine has no
    tilt-refinement solver — it only forward-applies a fixed tilt — so setting
    `update_tilt:` on a conventional engine changes nothing.

### Minimal example

```yaml
engines:
  - type: conventional
    # ...
    update_probe: {after: 5}
    update_positions: {after: 30}
```

evidenced by `examples/mos2_lsqml.yaml`, which starts probe updates after iteration 5 and
position updates after iteration 30, presumably to let the object/probe stabilize before
positions are allowed to move.

## Schedules: `constant`, `piecewise`, `expr`

A schedule field accepts a plain `float`, a built-in schedule hook, or an external
`"package.module:function"` reference.

| Name | Property schema | Fields | What it does |
| --- | --- | --- | --- |
| `constant` | `ConstantScheduleProps` | `value: float` | Always returns `value` — equivalent to writing the plain float directly, useful when a nested field requires a schedule-typed value |
| `piecewise` | `PiecewiseScheduleProps` | `init: ScheduleLike`, `steps: dict[int, ScheduleLike]` | Returns `init` before the first configured threshold in `steps`, then jumps to each threshold's own value (or nested schedule) once the current iteration reaches it |
| `expr` | `ExprScheduleProps` | `expr: str` | Evaluates `expr` as Python, with five names available: `i`, `iter`, `state`, `niter`, `np` (see the trust warning above) |

Type/default details are in the
[generated schedule reference](../../generated/hooks/schedule.md).

- **Units:** whatever unit the field the schedule is assigned to expects (a learning rate
  is dimensionless-relative, see
  [Solvers and learning rates](solvers-and-learning-rates.md); a conventional solver
  coefficient like `beta_object` is a dimensionless scale factor).
- **Valid range:** guidance pending beyond the `examples/` pattern below — no test sweeps
  schedule shapes.
- **Lifecycle stage:** a schedule is re-evaluated once per iteration by whichever solver
  owns the field it is assigned to — a gradient solver's schedule-valued properties are
  re-evaluated in `update_for_iter`, outside the JIT-traced computation; a conventional
  solver's schedule-valued properties are re-evaluated once per iteration inside
  `run_iteration`.
- **Engines/backends:** schedules are accepted identically on both engine families; the
  fields that currently accept a schedule are every conventional solver's tunable
  coefficient (`beta_object`, `beta_probe`, `illum_reg_object`, `illum_reg_probe`, `gamma`
  for `lsqml`; `beta_object`, `beta_probe` for `epie`) and every gradient solver's
  learning-rate-like fields (`sgd.learning_rate`/`momentum`, `adam.learning_rate`,
  `polyak_sgd.max_learning_rate`/`scaling` — note `polyak_sgd.f_min` is a plain `float`,
  not schedule-valued).

### `piecewise` example: a stepped learning-rate decay

```yaml
solvers:
  object:
    type: adam
    learning_rate:
      type: piecewise
      init: 1.0e-3
      steps:
        50: 5.0e-4
        100: 1.0e-4
```

Before iteration 50 this returns `1.0e-3`; from iteration 50 up to (not including) 100 it
returns `5.0e-4`; from iteration 100 onward it returns `1.0e-4`. A `piecewise` schedule's
`init` and each entry in `steps` can themselves be another schedule (including another
`piecewise` or an `expr`), letting you nest a ramp inside a step.

### `expr` example: an exponential ramp-in

```yaml
solver:
  type: lsqml
  beta_object:
    type: expr
    expr: '1.0 - np.exp(-i / 3)'
  beta_probe:
    type: expr
    expr: '1.0 - np.exp(-i / 3)'
```

evidenced by `examples/prsco3_lsqml.yaml`, which ramps both `beta_object` and `beta_probe`
from near `0` toward `1` over the first several iterations rather than starting at full
step size immediately. Remember the trust warning above before using `expr` in a plan you
didn't write.

### Minimal example

```yaml
engines:
  - type: gradient
    solvers:
      object:
        type: adam
        learning_rate:
          type: piecewise
          init: 1.0e-3
          steps:
            50: 5.0e-4
            100: 1.0e-4
```

## Interactions

- A schedule's iteration count (`i`, `iter.engine_iter`) is always relative to the
  **current engine**, not the whole reconstruction — a later engine stage's schedule
  restarts from `i=1`. See
  [Termination and diagnostics](termination-and-diagnostics.md#niter-and-iteration-counting)
  for how this interacts with `niter`.
- Flags and schedules never mutate state themselves — only
  [constraints](regularization.md) do. A flag only decides *whether* an update or check
  happens this iteration; it does not perform the update.
- An external (`package.module:function`) flag or schedule's properties are passed through
  as a plain, unvalidated `dict` — a typo in a custom property name is only discovered when
  the hook is actually called, not at `phaser validate` time.

## Maintainer sources

- `phaser/hooks/schedule.py`
- `phaser/types.py`
- `phaser/plan.py`
- `docs/architecture/hooks/schedules-and-flags.md`
- `docs/generated/hooks/schedule.md`
- `docs/generated/hooks/flag.md`
- `docs/generated/plan/index.md`
- `examples/mos2_lsqml.yaml`
- `examples/prsco3_lsqml.yaml`
