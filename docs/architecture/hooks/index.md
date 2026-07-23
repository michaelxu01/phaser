# Hooks

A [**hook**](../../concepts/glossary.md#hook) is Phaser's configuration-driven
construction mechanism: a plan field that names a behavior — either a short, registered
built-in name or an external `"package.module:function"` reference — and is called with
properties parsed from the plan's YAML/JSON. This page describes hook anatomy in general;
each family below has its own page, following the
[hook-family template](../../design/authoring-guide.md#hook-family-page-template).

## Hook families and when they run

| Family | Runs when | One-line lifecycle description |
| --- | --- | --- |
| [Raw-data loaders](raw-data-loaders.md) | Once, first, before any state exists | Loads patterns, mask, sampling, wavelength, and any loader-supplied scan/tilt/probe metadata from a file or array. |
| [Initialization](initialization.md) | Once, during state construction | Constructs (or, on restart, is skipped in favor of reusing) the probe, object, scan, and tilt from metadata merged with `init.*` — see [Initialization merge semantics](../lifecycle.md#raw-data-loading-and-the-initialization-merge). |
| [Post-load](post-load.md) | Once, right after the raw-data hook, before initialization | Processes the raw `patterns`/`mask`/metadata dictionary — crop, offset, scale, bin, or add synthetic Poisson noise. |
| [Post-init](post-init.md) | Once, after a complete `ReconsState` exists, before the first engine | Processes `(patterns, state)` together — drop invalid patterns, align the diffraction origin. |
| [Schedules and flags](schedules-and-flags.md) | Once per iteration, inside a running engine | Produces a number (schedule) or boolean (flag) instead of constructing state; backs fields like learning rates and `update_probe`/`update_object`. |
| [Noise models](noise-models.md) | Once per engine, then called every group/iteration | Supplies the loss and, where implemented, the reciprocal-space wave update relating simulated and measured intensity. **Restriction:** Poisson's wave update raises `NotImplementedError` (`phaser/engines/common/noise_models.py:112`), so it works with the gradient engine (loss only) but not the conventional engines, which require a wave update. |
| [Solvers](solvers.md) | Constructed once per engine, then called every group or iteration | Conventional solvers (ePIE, LSQML) drive a full iteration; gradient solvers (SGD, Adam, Polyak-SGD) update a declared, disjoint set of variables; position solvers adjust scan positions from position gradients. |
| [Cost regularizers](cost-regularizers.md) | Evaluated wherever the gradient loss is; **gradient engine only** | Adds a differentiable term to the objective, encoding a prior belief about object, probe, or tilt. |
| [Group constraints](group-constraints.md) | After every group, on both engine families | Mutates state directly (no objective term). |
| [Iteration constraints](iteration-constraints.md) | After every iteration, on both engine families | Mutates state directly (no objective term). |
| [Engines](engines.md) | Once per `plan.engines` entry, in order | Runs a whole reconstruction stage (conventional or gradient) against the shared state; `prepare_for_engine` may reshape that state first — see [Engine-boundary reshaping](../lifecycle.md#engine-boundary-reshaping). |

## Suggested reading order

Read this page first, then whichever family page matches the extension surface you are
working on. If you are new to the hook mechanism generally, read
[Raw-data loaders](raw-data-loaders.md) first — it is the simplest complete example of
the pattern.

## Hook anatomy

Every hook family is a subclass of `Hook[T, U]` (`phaser/hooks/hook.py`) with a
class-level `known` registry: `t.Dict[str, Tuple[ref, props_type]]` (or a 3-tuple whose
third element is a tuple of declared optional-dependency package names). Given a YAML
value, a hook resolves one of two ways:

- **a registered short name**, such as `type: empad` — the `known` registry maps it to
  `(function_reference, properties_dataclass)` (plus dependencies, if any); the
  properties dataclass validates the remaining fields against a schema at plan-parse
  time, so unknown or mistyped properties are caught before any hook runs
  (`HookConverter.try_convert`, `phaser/hooks/hook.py:113-139`);
- **an external reference**, `type: "package.module:function"` — resolved by
  `importlib.import_module` and called directly. Its properties are passed through as a
  plain dictionary and are **not** schema-validated, because Phaser has no way to know
  their shape ahead of time — a typo in an external hook's properties is only discovered
  when the hook actually runs, not at parse time.

### Where registries live

A family's `known` dictionary is populated either directly in its defining module or
later, from `phaser/plan.py`, once the concrete implementations it names are available:

| Registry | Defined in | Populated in |
| --- | --- | --- |
| `RawDataHook.known` | `phaser/hooks/__init__.py` | same module |
| `ProbeHook.known`, `ObjectHook.known`, `ScanHook.known`, `TiltHook.known` | `phaser/hooks/__init__.py` | same module |
| `PostLoadHook.known`, `PostInitHook.known` | `phaser/hooks/__init__.py` | same module |
| `FlagHook.known`, `ScheduleHook.known` | `phaser/hooks/schedule.py` | same module |
| `PositionSolverHook.known` | `phaser/hooks/solver.py` | same module |
| `NoiseModelHook.known`, `ConventionalSolverHook.known`, `GradientSolverHook.known`, `EngineHook.known` | `phaser/hooks/solver.py` / `phaser/hooks/__init__.py` (empty `{}`) | `phaser/plan.py`, alongside the plan classes that reference them |
| `IterConstraintHook.known`, `GroupConstraintHook.known`, `CostRegularizerHook.known` | `phaser/hooks/regularization.py` | same module |

This split exists because engines, noise models, conventional solvers, and gradient
solvers are registered next to the plan dataclasses that describe their properties
(`phaser/plan.py`), while every other family is self-contained in its own
`phaser/hooks/` module. This matches the [sources of truth](../../design/documentation-architecture.md#sources-of-truth):
built-in hook names and schemas come from each family's `known` registry, *including*
registrations performed in `phaser/plan.py`.

### Lazy resolution and caching

A hook's function reference is not imported at plan-parse time. `Hook.resolve()`
(`phaser/hooks/hook.py:56-59`) only calls `_resolve_ref()` (`phaser/hooks/hook.py:32-54`) —
which does the `importlib.import_module` — the first time the hook is called as a function
(`Hook.__call__`, `phaser/hooks/hook.py:61-62`), and caches the resolved callable on
`self.f` so every subsequent call reuses it instead of re-importing:

```python
def resolve(self) -> t.Callable[..., U]:
    if self.f is None:
        self.f = self._resolve_ref()
    return self.f
```

Practically, this means a plan can *reference* a hook whose optional dependency isn't
installed (e.g. a `gatan` raw-data loader, which declares a dependency on `rsciio`:
`RawDataHook.known['gatan']`, `phaser/hooks/__init__.py`) as long as that hook is never
actually invoked. `_resolve_ref` checks any declared dependencies (`check_dependencies`,
`phaser/hooks/_dependencies.py`) immediately before importing, so a missing optional
dependency produces a clear error at the point the hook would first run, not at parse time
and not as a bare `ImportError` deep in an unrelated module.

### A resolved hook is a plain callable

Once resolved, a hook is called with one argument (a fixed dict shape per family — see
each family page) and, internally, its stored `props` (`Hook.__call__`,
`phaser/hooks/hook.py:61-62`):

```python
def __call__(self, args: T) -> U:
    return self.resolve()(args, props=self.props if self.props is not None else {})
```

A hook is invoked once (raw-data loaders, initializers, `post_load`/`post_init`, engines)
or once per iteration (schedules and flags) — see the table above for each family's
cadence. This distinguishes hooks from the other extension surfaces described in the
[overview's extension-mechanism table](../overview.md#extension-mechanisms): a hook
itself carries no state between calls, whereas solvers, regularizers, and constraints
maintain their own state across a whole engine run, and observers react to execution
events rather than being invoked as plan-configured behavior at all.

## Maintainer sources

- `phaser/hooks/hook.py`
- `phaser/hooks/__init__.py`
- `phaser/hooks/schedule.py`
- `phaser/hooks/solver.py`
- `phaser/hooks/regularization.py`
- `phaser/plan.py`
- `phaser/engines/common/noise_models.py`
