# Engine hooks

An [**engine**](../../concepts/glossary.md#engine) hook runs one whole reconstruction
stage — a full sequence of iterations over the data — against the shared
[`ReconsState`](../state-and-conventions.md), and returns the updated state. A plan's
`engines` field is an ordered list of engine hooks; `phaser.execute.execute_plan` runs them
one after another, threading the same state through every one (see the
[reconstruction lifecycle](../lifecycle.md) for the full parse-to-save sequence this page
assumes). Two engines are built in — **conventional** (ePIE, LSQML) and **gradient
descent** — configured by `ConventionalEnginePlan` and `GradientEnginePlan` respectively
(`phaser/plan.py`).

## Lifecycle point

`execute_plan` (`phaser/execute.py:21-47`) calls `execute_engine` once per entry in
`plan.engines`, in order. For each entry, `execute_engine` (`phaser/execute.py:50-94`):

1. reads `plan = t.cast(EnginePlan, engine.props)` and, if `plan.early_termination` is set,
   wraps the observer set in a `PatienceObserver`;
2. calls `prepare_for_engine` (`phaser/execute.py:384-452`), which may reshape the shared
   state to match this engine's configuration — resampling the probe/patterns to
   `sim_shape`, resampling or padding the object, changing probe mode count, reslicing the
   object, and creating a zeroed tilt map — **before** the engine hook itself ever runs
   ([Engine-boundary reshaping](../lifecycle.md#engine-boundary-reshaping));
3. resets `state.iter` for the new engine (`engine_num` incremented, `engine_iter` reset to
   `0`, `n_engine_iters` set to `plan.niter`);
4. calls the engine hook itself with a fixed `EngineArgs` payload (below) and assigns its
   returned `ReconsState` back onto the shared `PreparedRecons`.

An engine hook is therefore called exactly once per `plan.engines` entry — never once per
group or iteration; the engine's own run function handles its internal group/iteration loop
and calls the observer at the right points within it
([Accepted state and returned value](#accepted-state-and-returned-value),
[Minimal custom implementation](#minimal-custom-implementation) below).

## Callable signature and property schema

`EngineHook` is `Hook[EngineArgs, ReconsState]` (`phaser/hooks/__init__.py:227-238`):

```python
class EngineArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    dtype: t.Type[numpy.floating]
    xp: t.Any
    recons_name: str
    observer: 'Observer'
    seed: t.Any

class EngineHook(Hook[EngineArgs, 'ReconsState']):
    known = {}  # filled in by plan.py
```

`EngineHook.known` starts empty in its defining module and is populated in `phaser/plan.py:187-188`:

```python
EngineHook.known['conventional'] = ('phaser.engines.conventional.run:run_engine', ConventionalEnginePlan)
EngineHook.known['gradient'] = ('phaser.engines.gradient.run:run_engine', GradientEnginePlan)
```

Both properties classes are `EnginePlan` subclasses (`phaser/plan.py:43-99`, kw-only).
Shared `EnginePlan` fields (used by every engine, either directly by the engine's run
function or by the lifecycle code around it):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `sim_shape` | `Optional[Tuple[int, int]]` | `None` | Simulation array shape `(y, x)`; triggers probe/pattern resampling at the engine boundary if it differs from the current shape. |
| `resize_method` | `'pad_crop' \| 'resample'` | `'pad_crop'` | How `sim_shape` changes are applied — fixed physical extent vs. fixed pixel size. |
| `probe_modes` | `int` | `1` | Number of probe modes this engine simulates; see [mixed-state](../../concepts/glossary.md#mixed-state). |
| `base_mode_power` | `float` | `0.7` | Intensity fraction assigned to the base mode when *increasing* probe mode count at an engine boundary. |
| `bwlim_frac` | `Optional[float]` | `2/3` | Bandwidth-limiting fraction used when building slice propagators. |
| `obj_pad_px` | `float` | `5.0` | Padding (pixels, at the probe's pixel size) added around the scan extent when sizing/expanding the object. |
| `slices` | `Optional[Slices]` | `None` | Slice thicknesses for this engine; triggers reslicing at the engine boundary if they differ from the object's current thicknesses. |
| `niter` | `int` | `10` | Number of iterations this engine runs. |
| `grouping` | `Optional[int]` | `None` | Target [group](../../concepts/glossary.md#group) size. |
| `compact` | `bool` | `False` | Whether groups are assigned compactly (spatially contiguous) rather than sparsely. |
| `shuffle_groups` | `Optional[FlagLike]` | `None` | **Sentinel default**: resolved at runtime as `props.shuffle_groups or not props.compact` (`phaser/engines/conventional/run.py:31`, `phaser/engines/gradient/run.py:179`, identically in both engines) — leaving it `None` means "shuffle groups only when `compact` is `False`," not "never shuffle." |
| `buffer_n_groups` | `Optional[int]` | `2` | Groups of patterns to buffer on-device; `0` disables buffering, `None` preloads everything. |
| `jit_unroll_slices` | `None \| bool \| int` | `None` | Multislice unrolling under JIT (JAX backend only); the gradient engine's run function separately defaults a bare `None` to `10` (`gradient/run.py:162`). |
| `update_probe`, `update_object`, `update_positions`, `update_tilt` | `FlagLike` | `True`, `True`, `False`, `False` | Whether each variable is updated this iteration — see [Schedules and flags](schedules-and-flags.md). `update_tilt` is inert on the conventional engine (see restrictions below). |
| `calc_error`, `calc_error_fraction` | `FlagLike`, `float` | `SimpleFlag(every=1)`, `0.1` | Whether/how often to compute a reported error, and what fraction of groups to sample when computing it. |
| `save`, `save_images`, `save_options` | `FlagLike`, `FlagLike`, `SaveOptions` | `False`, `False`, defaults | Checkpoint/image save cadence and formatting — consumed by the built-in `SaveObserver`, not by the engine's run function itself. |
| `early_termination`, `early_termination_smoothing` | `Optional[int]`, `float` | `None`, `0.9` | Iterations without improvement before stopping, and the smoothing factor applied to the tracked error. |
| `check_every_group`, `send_every_group` | `bool`, `bool` | `False`, `False` | Whether to check for non-finite values, and whether to notify the observer, after every group rather than only every iteration. |

`ConventionalEnginePlan` additionally declares `noise_model`, `solver`,
`position_solver` (optional), `group_constraints`, `iter_constraints`.
`GradientEnginePlan` additionally declares `noise_model`, `solvers` (a dict keyed by
variable set), `regularizers`, `group_constraints`, `iter_constraints` — see
[Noise-model hooks](noise-models.md), [Solver hooks](solvers.md),
[Cost regularizers](cost-regularizers.md), [Group constraints](group-constraints.md), and
[Iteration constraints](iteration-constraints.md).

## Accepted state and returned value

The engine hook receives one `EngineArgs` dict and returns one `ReconsState`
(`phaser/execute.py:77-85`):

- `data: Patterns` — `patterns` (measured intensities) and `pattern_mask`, already
  resampled to this engine's `sim_shape` if `prepare_for_engine` changed it.
- `state: ReconsState` — already reshaped for this engine (probe/object sampling, probe
  mode count, slice count, tilt map) by the time the engine hook sees it; the engine's job
  is to update it, not to reconcile it with the plan's configuration.
- `dtype`, `xp` — the selected real dtype and backend module for this run.
- `recons_name: str` — the reconstruction's name, passed to `observer.init_engine`.
- `observer: Observer` — see [Obligations toward observers](#obligations-toward-observers)
  below.
- `seed` — passed through from `execute_engine` (currently always `None` at that call site,
  `phaser/execute.py:84`).

The returned `ReconsState` becomes `recons.state` for every subsequent engine in
`plan.engines`, and (after the last engine) is what `observer.finish_recons` and the
built-in `SaveObserver` see.

### Obligations toward observers

A conforming engine calls, in order, over one run: `observer.init_engine` (once, with
`recons_name=` and `plan=`; both built-in engines also pass
`noise_model=noise_model.name()`, an extra keyword every `Observer.init_engine`
implementation accepts via `**kwargs`), `observer.start_engine` (once, after any
presolve/rescaling step), then per group `observer.update_group` and per iteration
`observer.update_iteration`, and finally `observer.finish_engine` (once). Skipping these
doesn't raise an error in general, but the built-in `SaveObserver` requires
`init_engine`/`start_engine` to have run before `update_iteration` — calling
`update_iteration` first raises `AssertionError` from `SaveObserver`'s internal `out_dir`
check (`phaser/observer.py:263`), confirmed by executing the custom engine below with the
default observers before calling `init_engine`/`start_engine`.

## Built-in implementations

Registered in `EngineHook.known` (`phaser/plan.py:187-188`):

| Name | Function | Props | One-line description |
| --- | --- | --- | --- |
| `conventional` | `phaser.engines.conventional.run:run_engine` | `ConventionalEnginePlan` | Runs ePIE or LSQML (whichever `solver` names) for `niter` iterations, with an optional position solver. |
| `gradient` | `phaser.engines.gradient.run:run_engine` | `GradientEnginePlan` | Runs gradient descent for `niter` iterations, differentiating the configured noise model's loss through JAX or Torch, using per-group and per-iteration gradient solvers. |

## Minimal custom implementation

Writing a custom engine has one structural consequence unlike any other hook family:
`execute_plan`, `execute_engine`, `prepare_for_engine`, and `initialize_reconstruction` all
read plan fields directly off `engine.props` (cast as `EnginePlan`), not only the resolved
engine function. Confirmed by execution: registering an engine purely as an **external**
`"package.module:function"` reference, with no entry in `EngineHook.known`, fails before the
custom function is ever called:

```python
plan = ReconsPlan.from_data({
    'name': 'test', 'raw_data': 'scratch_loader:load_empty',
    'engines': [{'type': 'custom_engine_probe:noop_engine', 'obj_pad_px': 5.0}],
})
execute_plan(plan, xp=numpy)
```

```text
AttributeError: 'dict' object has no attribute 'niter'
  File "phaser/execute.py", line 33, in <genexpr>
    t.cast(EnginePlan, engine.props).niter
```

This happens because an external hook's properties are always a plain, schema-unvalidated
`dict` (`phaser/hooks/hook.py:113-139`) — fine for hook families whose only consumer is the
resolved function itself, but the engine family's properties are *also* read by attribute
by the surrounding lifecycle code, and a plain `dict` has no attribute access.

**The working pattern** is the same one `phaser/plan.py` uses for the two built-in
engines: give the engine an `EnginePlan`-derived properties dataclass and register it in
`EngineHook.known` under a short name. `known` is an ordinary mutable class attribute, so any
importing code can add to it before parsing a plan — no need to modify Phaser itself:

```python
import logging
import numpy
from phaser.state import ReconsState
from phaser.hooks import EngineArgs
from phaser.plan import EnginePlan


class NoopEnginePlan(EnginePlan, kw_only=True):
    """Inherits every EnginePlan field (niter, sim_shape, obj_pad_px, ...) and adds none."""
    pass


def noop_engine(args: EngineArgs, props: NoopEnginePlan) -> ReconsState:
    """A custom engine that performs no reconstruction: it only advances the
    iteration counter `props.niter` times, calling the observer exactly as a
    real engine would around its actual object/probe/position update."""
    state = args['state']
    observer = args['observer']
    logger = logging.getLogger(__name__)

    observer.init_engine(state, recons_name=args['recons_name'], plan=props)
    observer.start_engine(state)

    for i in range(1, props.niter + 1):
        state.iter.engine_iter = i
        # a real engine would update state.object/probe/scan here
        observer.update_group(state, False)
        observer.update_iteration(state, i, props.niter, {'total_loss': 0.0})
        logger.info(f"noop_engine: iteration {i}/{props.niter}")

    observer.finish_engine(state)
    return state
```

Registered and run end to end through a real `ReconsPlan`/`execute_plan` call, with a
tiny synthetic 4×4-position, 8×8-pixel dataset built the same way
`tests/test_initialization.py` builds one:

```python
from phaser.hooks import EngineHook
from phaser.execute import execute_plan

EngineHook.known['noop'] = ('custom_engine_probe:noop_engine', NoopEnginePlan)

plan = ReconsPlan.from_data({
    'name': 'test',
    'raw_data': 'scratch_loader:load_empty',   # a minimal manual RawData hook
    'engines': [{'type': 'noop', 'niter': 3}],
})
execute_plan(plan, xp=numpy, override_observers=[])
```

This completed end to end (`initializing reconstruction` → `Preparing for engine #1...` →
three `noop_engine: iteration i/3` log lines → `Reconstruction finished!`), confirming: the
plan parses and validates `NoopEnginePlan`'s inherited `EnginePlan` fields exactly like a
built-in engine; `prepare_for_engine` runs against it without error; the custom function
receives a real `EngineArgs` with a fully-initialized `ReconsState`; and the returned state
flows to `finish_recons` normally. `override_observers=[]` avoids the default
`SaveObserver`, which requires `out_dir`-related setup not relevant to this
signature-focused example.

## YAML invocation

Built-in, by registered name:

```yaml
engines:
  - type: gradient
    niter: 100
    noise_model: {type: poisson, eps: 2.0}
    solvers:
      object: {type: adam, learning_rate: 5.0e-3}
```

A **registered custom** engine, after the user's own code adds it to `EngineHook.known`
(above), behaves exactly like a built-in short name in YAML, including full schema
validation of its properties:

```yaml
engines:
  - type: noop
    niter: 3
```

There is no supported **external** `"package.module:function"` form for engines with
unvalidated inline properties, unlike every other hook family — see
[Minimal custom implementation](#minimal-custom-implementation) above for why; this differs
from the general external-hook description on [Hooks](index.md).

## Engine and backend restrictions

- The **gradient** engine requires the `jax` or `torch` backend; `prepare_for_engine` raises
  `ValueError` immediately for any other backend when the engine is a `GradientEnginePlan`
  (`phaser/execute.py:386-387`) — [Engine families and backends](../overview.md#engine-families-and-backends).
  Not JAX-only, despite older project documentation.
- The **conventional** engine is not backend-restricted by this code path.
- `update_tilt` is inert on the conventional engine: `phaser/engines/conventional/run.py`
  never reads it, though it's inherited from the shared `EnginePlan` base class. Tilt
  refinement is a gradient-engine-only optimization (assign a gradient solver to the `tilt`
  variable) — [Solver hooks](solvers.md),
  [Engine families and backends](../overview.md#engine-families-and-backends).
- A custom engine registered via `EngineHook.known` inherits whichever restrictions its own
  properties dataclass and run function choose to enforce; nothing in the lifecycle code
  imposes a backend check unless the properties class is (or subclasses) `GradientEnginePlan`
  and the surrounding code specifically checks `isinstance(engine, GradientEnginePlan)`
  (`phaser/execute.py:386,446`).

## Optional dependencies

None from the engine-hook mechanism itself. The built-in `gradient` engine requires the
`jax` or `torch` optional dependency group (`pyproject.toml`) to run at all — see
[Engine families and backends](../overview.md#engine-families-and-backends). The built-in
`conventional` engine has no additional optional dependency.

## Testing pattern

Build a small, fully-initialized `ReconsState` the same way
`tests/test_initialization.py` does — a manual `raw_data` hook returning tiny synthetic
`patterns`/`mask`/`sampling`/scan metadata (a few scan positions, an 8×8 or 16×16 detector)
— then either:

- run the whole plan through `execute_plan`/`initialize_reconstruction` (as executed
  above) and assert on the returned `PreparedRecons.state`, or
- call the engine hook directly with a hand-built `EngineArgs` dict (state from a prior
  `initialize_reconstruction` call, `dtype`/`xp` matching it, and a plain `Observer()` for
  a no-op observer) to isolate the engine from plan parsing.

For a real (non-`noop`) custom engine, assert the specific state fields your engine claims
to update (`state.object.data`, `state.probe.data`, `state.scan`, `state.tilt`) actually
changed, and that `state.iter.engine_iter`/`total_iter` advanced as expected — `niter`
times, starting from whatever `total_iter` was before this engine ran.

## Maintainer sources

- `phaser/hooks/__init__.py`
- `phaser/plan.py`
- `phaser/execute.py`
- `phaser/observer.py`
- `phaser/state.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/gradient/run.py`
- `tests/test_initialization.py`
