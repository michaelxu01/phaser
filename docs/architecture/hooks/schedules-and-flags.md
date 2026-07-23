# Schedule and flag hooks

A [**flag**](../../concepts/glossary.md#flag) produces a boolean; a
[**schedule**](../../concepts/glossary.md#schedule) produces a number. Both vary across an
engine's iterations, and both back plan fields that would otherwise be a fixed `bool` or
`float` for the whole engine — for example, turning on position refinement only after
iteration 20, or decaying a learning rate over the course of a run.

!!! danger "Trust warning"
    An **expression schedule** (`type: expr`) evaluates its `expr` string with Python's
    `eval` (`phaser/hooks/schedule.py:62-69`), and the globals dictionary passed to `eval`
    does not set `__builtins__: {}` — Python fills it in automatically, so the expression
    can reach `__import__`, `open`, `exec`, and anything else importable, not just the five
    names documented below. A plan file that uses an expression schedule is therefore
    equivalent to a script: running an untrusted plan is running untrusted code. This
    matters even more for the web manager and local/Slurm workers described in
    [Interfaces and deployment](../interfaces.md) — anyone who can submit a plan to a
    worker can execute code as that worker. Only run plans you trust, and never present an
    expression schedule in introductory material without this warning.

## Lifecycle point

Flags and schedules run **once per iteration, inside a running engine** — after
[initialization](initialization.md), [post-init](post-init.md), and any
[engine-boundary reshaping](../lifecycle.md#engine-boundary-reshaping), repeated for every
iteration of every engine in `plan.engines`
([hook families table](index.md#hook-families-and-when-they-run) compares this cadence to
every other hook family). Concretely:

- both engines evaluate `update_probe`, `update_object`, `update_positions`, `calc_error`,
  and `shuffle_groups` once at the start of each iteration
  (`phaser/engines/conventional/run.py:94-104`, `phaser/engines/gradient/run.py:255-261`);
  the gradient engine also evaluates `update_tilt` there
  (`phaser/engines/gradient/run.py:172-177,256`);
- a **gradient solver's** schedule-valued properties (for example `learning_rate`) are
  re-evaluated once per iteration by `GradientSolver.update_for_iter`
  (`phaser/engines/gradient/solvers.py:84-85`), *outside* the JIT-traced computation — a
  call-site comment notes: "this needs to be done outside the JIT context, which makes this
  kinda hacky" (`phaser/engines/gradient/run.py:266-267`); the same holds for a
  **conventional solver's** schedule-valued properties (for example LSQML's `beta_object`,
  `gamma`), evaluated once per iteration inside `LSQMLSolver.run_iteration`
  (`phaser/engines/conventional/solvers.py`);
- `plan.save` and `plan.save_images` are flags evaluated by the built-in `SaveObserver` once
  per iteration (`phaser/observer.py:226-227,269`), not by the engine loop itself
  ([Observers](../observers.md)).

Unlike most other hook families, a schedule or flag hook is **not** constructed in a
separate step before use — it is called directly, once per iteration, with the same
argument shape every time. A source comment marks this as provisional:
`# TODO: make these hooks two-step` (`phaser/hooks/schedule.py:32`). Contrast
[cost regularizers](cost-regularizers.md), [group constraints](group-constraints.md), and
[iteration constraints](iteration-constraints.md), which *are* two-step: a hook is called
once with no arguments to build a stateful object, and that object's own method is called on
the constraint's own cadence thereafter.

## Callable signature and property schema

```python
class FlagArgs(t.TypedDict):
    state: 'ReconsState'
    niter: int

Flag: t.TypeAlias = t.Callable[['FlagArgs'], bool]
Schedule: t.TypeAlias = t.Callable[['FlagArgs'], float]

FlagLike: t.TypeAlias = t.Union[bool, SimpleFlag, FlagHook]
ScheduleLike: t.TypeAlias = t.Union[float, ScheduleHook]
```

(`phaser/hooks/schedule.py:13-29`; `SimpleFlag` is in `phaser/types.py`.) A resolved flag
or schedule is called with one `FlagArgs` argument and returns a `bool` or `float`
directly — there is no `props=` argument at the call site, unlike a two-step hook, because
`props` are already bound into the callable when it was constructed from the plan.

`niter` is the **current engine's** total iteration count (`EnginePlan.niter`), not the
whole reconstruction's; `state` is the live `ReconsState`, from which every built-in reads
`state.iter.engine_iter` — the 1-indexed iteration count *within this engine*
(`phaser/state.py`, [glossary: iteration](../../concepts/glossary.md#iteration)).

### `SimpleFlag`

```python
class SimpleFlag(Dataclass):
    after: int = 0
    every: int = 1
    before: t.Optional[int] = None

    def __call__(self, args: 'FlagArgs') -> bool:
        i = args['state'].iter.engine_iter
        return (
            (self.before is None or i < self.before)
            and i > self.after
            and (i - self.after) % self.every == 0
        )
```

(`phaser/types.py:209-226`.) A `SimpleFlag` is `True` on iteration `i` exactly when
`i > after`, `i < before` (if `before` is set), and `i - after` is a multiple of `every`.
With the default `SimpleFlag(after=0, every=1, before=None)` (`EnginePlan.calc_error`'s
default, `phaser/plan.py:81`), this is `True` on every iteration `i >= 1`. `SimpleFlag`
does not need to be resolved through a registry — it implements `__call__` and `resolve()`
itself (`phaser/types.py:220-229`) and is accepted directly as a `FlagLike` value.

### Built-in schedule property schemas

| Built-in | Property schema | Fields |
| --- | --- | --- |
| `constant` | `ConstantScheduleProps` | `value: float` |
| `piecewise` | `PiecewiseScheduleProps` | `init: ScheduleLike`, `steps: t.Dict[int, ScheduleLike]` |
| `expr` | `ExprScheduleProps` | `expr: str` |

(`phaser/hooks/schedule.py:35-59`.) `piecewise_schedule` picks the largest configured
threshold in `steps` that is `<= state.iter.engine_iter`, evaluates *that* entry's
schedule, and falls back to `init` if the current iteration is below every threshold
(`phaser/hooks/schedule.py:48-55`) — thresholds and their schedules can themselves be
constants, so a piecewise schedule can nest another `piecewise` or `expr` schedule as one
of its steps.

## Accepted state and returned value

Input: a `FlagArgs` dictionary, `{'state': ReconsState, 'niter': int}`. Output: `bool` for
a flag, `float` for a schedule. Neither reads or returns anything else — a flag or
schedule cannot itself mutate `state`; only [constraints](group-constraints.md) do that.

## Built-in implementations

**Schedules** (`ScheduleHook.known`, `phaser/hooks/schedule.py:75-77`): `constant`,
`piecewise`, `expr` — see the property tables above.

**Flags:** `FlagHook.known` is an empty registry (`phaser/hooks/schedule.py:19`) — there
are **no built-in named flag hooks**. A flag field is either a plain `bool`, a
`SimpleFlag(after=, every=, before=)`, or an external `package.module:function` reference;
`type: some_registered_name` is only meaningful for a *schedule* field.

### `expr_schedule`

```python
def expr_schedule(args: FlagArgs, props: ExprScheduleProps) -> float:
    val = float(eval(props.expr, {
        'i': args['state'].iter.engine_iter,
        'iter': args['state'].iter,
        'state': args['state'],
        'niter': args['niter'],
        'np': numpy,
    }))
    return val
```

(`phaser/hooks/schedule.py:62-72`.) The five names exposed to the expression are exactly
`i` (the current 1-indexed engine iteration, an `int`), `iter` (the full `IterState`
object), `state` (the full `ReconsState`), `niter` (this engine's `niter`), and `np` (the
`numpy` module). As the trust warning above states, the expression is **not** sandboxed to
these five names ([Callable signature](#callable-signature-and-property-schema)).

The generated reference for built-in hook property schemas does not exist yet (tracked in
the [implementation checklist](../../design/implementation-checklist.md), Phase 2); this
section is verified directly against `phaser/hooks/schedule.py` and `phaser/types.py`.

## Minimal custom implementation

A custom flag and a custom schedule are both plain Python functions matching `Flag`/
`Schedule` above — no base class or decorator needed. Run against a synthetic state with
`numpy` (no JAX/Torch needed, since flags and schedules never touch gradients):

```python
def every_third_after_5(args: 'FlagArgs', props: dict) -> bool:
    """True every third iteration, starting after iteration 5."""
    i = args['state'].iter.engine_iter
    return bool(i > 5 and i % 3 == 0)


def linear_decay(args: 'FlagArgs', props: dict) -> float:
    """Linearly decay from props['start'] to 0 over the engine's iterations."""
    i = args['state'].iter.engine_iter
    niter = args['niter']
    return float(props['start'] * (1.0 - i / niter))
```

Executed against a synthetic `ReconsState` (`iter.engine_iter` set to `2` and then `9`,
`niter=10`), with the NumPy backend:

```text
custom flag @ iter=2: False
custom flag @ iter=9: True
custom schedule @ iter=9, niter=10, start=1.0: 0.09999999999999998
```

which matches the intended behavior: the flag is `False` at iteration 2 (`2 > 5` is
false), `True` at iteration 9 (`9 > 5` and `9 % 3 == 0`), and the schedule linearly
decays from `1.0` toward `0.0`, reaching `0.1` at iteration 9 of 10.

## YAML invocation

Built-in short name:

```yaml
engines:
  - type: gradient
    update_positions:
      after: 20        # SimpleFlag: start correcting positions after iteration 20
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

External reference (a plan-defined schedule, not one of the three registered names) —
external hook properties are **not** schema-validated, so a typo in `start` below is only
discovered when the schedule actually runs, not at `phaser validate` time:

```yaml
        learning_rate:
          type: "my_package.my_schedules:linear_decay"
          start: 1.0e-3
```

## Engine and backend restrictions

Both engine families (`ConventionalEnginePlan` and `GradientEnginePlan` both inherit every
`FlagLike`/`ScheduleLike` field from `EnginePlan`, `phaser/plan.py:43-98`) accept flags and
schedules identically — there is no engine restriction in the hook mechanism itself.
Fields that currently accept a `FlagLike` value: `shuffle_groups`, `update_probe`,
`update_object`, `update_positions`, `update_tilt`, `calc_error`, `save`, `save_images`
(`phaser/plan.py:59,76-85`). Fields that accept a `ScheduleLike` value: every conventional
solver's tunable coefficients (`LSQMLSolverPlan.beta_object`/`beta_probe`/
`illum_reg_object`/`illum_reg_probe`/`gamma`, `EPIESolverPlan.beta_object`/`beta_probe`,
`phaser/plan.py:120-134`) and every gradient solver's learning-rate-like fields
(`SGDSolverPlan.learning_rate`/`momentum`, `AdamSolverPlan.learning_rate`,
`PolyakSGDSolverPlan.max_learning_rate`/`scaling`, `phaser/plan.py:159-179` — note
`PolyakSGDSolverPlan.f_min` is a plain `float`, not schedule-valued).

!!! warning "Restriction"
    `update_tilt` is inherited by `ConventionalEnginePlan` from the shared `EnginePlan` base
    but has no effect there: only the gradient engine reads it
    (`phaser/engines/gradient/run.py:176`); the conventional engine's run loop never reads
    `props.update_tilt`. See the [overview](../overview.md#engine-families-and-backends) for
    the same restriction on tilt refinement generally.

No backend restriction is specific to schedules or flags; the gradient engine's own
`jax`/`torch` backend requirement (see the [overview](../overview.md#engine-families-and-backends))
applies regardless of whether any field it evaluates is schedule-valued.

## Optional dependencies

None. Built-in schedules use only `numpy`, already a core dependency
(`pyproject.toml`); no entry in `ScheduleHook.known` declares an optional-dependency
tuple. An external schedule or flag hook may import anything installed in the environment
— including, per the trust warning, anything at all.

## Testing pattern

Because a flag or schedule is a plain function of `FlagArgs`, it can be tested without
running any engine: construct a minimal `ReconsState` (or a stand-in object exposing only
`.iter.engine_iter`), call the function directly for the iteration numbers of interest,
and assert on the returned `bool`/`float`. This is the pattern used to produce the
"Minimal custom implementation" output above — no `ReconsPlan`, engine, or backend
selection is required. If a schedule is registered as a built-in (added to
`ScheduleHook.known`), also verify it round-trips through YAML with `phaser validate` on a
plan that references it by its registered name.

## Maintainer sources

- `phaser/hooks/schedule.py`
- `phaser/types.py`
- `phaser/plan.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/gradient/solvers.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/observer.py`
