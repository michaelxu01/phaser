# Iteration-constraint hooks

An **iteration constraint** mutates reconstruction state directly, once per full
iteration, rather than contributing a differentiable term to an objective. This is one of
three [**regularizer**](../../concepts/glossary.md#regularizer) lifecycles defined in
`phaser/hooks/regularization.py`. Contrast this page (mutation after every iteration, on
**both** engine families) with [Group constraints](group-constraints.md) (the same kind of
direct mutation, but after every group instead) and [Cost regularizers](cost-regularizers.md)
(a differentiable loss term, gradient engine only). Several implementations are shared
with group constraints — see the built-ins table below for which lifecycle each name is
registered under.

## Lifecycle point

An iteration constraint runs **after every full iteration, on both engine families**,
after that iteration's per-group and per-iteration solver updates have both been applied,
and before the iteration's position-solver step (if any):

- in the **gradient engine**, in `run_engine`'s per-iteration loop, right after the
  per-iteration solvers (`positions`, `tilt`) apply their update
  (`phaser/engines/gradient/run.py:334-337`);
- in the **conventional engines**, in `run_engine`'s per-iteration loop, right after
  `solver.run_iteration` returns and the object/probe finite-value check passes, via
  `sim.apply_iter_constraints()` (`phaser/engines/conventional/run.py:111`, calling
  `phaser/engines/common/simulation.py:151-160`) — before the position-solver update that
  follows at `phaser/engines/conventional/run.py:113-125`.

Because an iteration constraint runs once per iteration rather than once per group, it has
no `group` argument — it only ever sees the whole `ReconsState` after the complete
iteration.

## Callable signature and property schema

`IterConstraintHook` (`Hook[None, IterConstraint]`, `phaser/hooks/regularization.py:77`)
is a **two-step** hook, exactly like [group constraints](group-constraints.md#callable-signature-and-property-schema):
the hook is called once, with `None`, to construct an `IterConstraint` instance; that
instance's `apply_iter` is then called after every iteration.

```python
@t.runtime_checkable
class IterConstraint(HasState[StateT], t.Protocol[StateT]):
    def apply_iter(
        self, sim: 'ReconsState', state: StateT
    ) -> t.Tuple['ReconsState', StateT]:
        ...
```

(`phaser/hooks/regularization.py:21-24`.) `HasState[StateT]` additionally requires
`init_state(self, sim: ReconsState) -> StateT`, called once when the engine starts.

Property schemas used by built-ins unique to this family (the rest —
`ClampObjectAmplitudeProps`, `LimitProbeSupportProps`, `ObjLowPassProps`, `GaussianProps`,
`NonNegObjectPhaseProps` — are shared with
[group constraints](group-constraints.md#callable-signature-and-property-schema)):

| Property schema | Fields |
| --- | --- |
| `RegularizeLayersProps` | `sigma: float = 50.0` (Å, standard deviation of the inter-layer Gaussian filter), `weight: float = 0.9` |
| `UnstructuredGaussianProps` (base) | `attr_path: str`, `sigma: float`, `weight: float = 0.9` |
| `TiltGaussianProps` | `UnstructuredGaussianProps` with `attr_path` fixed to `'tilt'` |
| `OPRGaussianProps` | `UnstructuredGaussianProps` with `attr_path` fixed to `'opr.data'` |

(`phaser/hooks/regularization.py:44-61`.)

## Accepted state and returned value

`apply_iter(sim, state)` receives the full `ReconsState` after the just-completed
iteration and the constraint's own carried state from `init_state` or the previous call.
It returns `(sim, new_state)` — the (possibly mutated) `ReconsState` and the constraint's
updated carried state. As with group constraints, there is no loss value.

## Built-in implementations

Registered in `IterConstraintHook.known` (`phaser/hooks/regularization.py:77-89`) and
implemented in `phaser/engines/common/regularizers.py`. `clamp_object_amplitude`,
`limit_probe_support`, `obj_low_pass`, `obj_gaussian`, `remove_phase_ramp`, and
`nonneg_object_phase` behave exactly as described on the
[group constraints page](group-constraints.md#built-in-implementations) — this family adds
three more:

| Name | Effect |
| --- | --- |
| `layers` | Gaussian-blurs the object **across the slice axis** (multislice only): approximates layer spacing as the mean of `object.thicknesses`, builds a Gaussian kernel of radius `~2*sigma` in that spacing, and convolves `log(object.data)` along the slice axis (the log is used because the transmission function is multiplicative, not additive), blended by `weight`. A no-op — `apply_iter` returns immediately — if the object has fewer than 2 slices (single-slice) |
| `tilt_gaussian` | Spatially Gaussian-blurs `state.tilt` (`attr_path='tilt'`): maps each scan position's tilt value onto the object's pixel grid via nearest-neighbor lookup (`scipy.spatial.KDTree`), blurs that image in Fourier space, and samples the blurred image back at the original scan positions, blended by `weight` |
| `opr_gaussian` | The same spatial-blur procedure as `tilt_gaussian`, but targeting `state.opr.data` (`attr_path='opr.data'`) instead of `state.tilt` |

!!! warning "Restriction"
    **`opr_gaussian` is not currently usable.** `ReconsState` (`phaser/state.py`) has no
    `opr` attribute anywhere in the codebase — a search of `phaser/` for `opr` finds only
    the string `'opr.data'` inside `OPRGaussianProps`/`UnstructuredGaussian` itself.
    Constructing and calling it against a real `ReconsState` raises
    `AttributeError: Can't get path 'opr.data' in reconstruction state`
    (`UnstructuredGaussian.init_state`, `phaser/engines/common/regularizers.py:521-526`),
    verified directly. This looks like a registered hook for a feature (per-scan-position
    "OPR" state, presumably orthogonal-probe-relaxation-style mixed-state probes) not yet
    exposed on `ReconsState`.

!!! warning "Restriction"
    **`tilt_gaussian` is backend-dependent.** `UnstructuredGaussian.apply_iter`
    (`phaser/engines/common/regularizers.py:547-583`) resolves `attr_path` to `sim.tilt` —
    a raw array, not a wrapped state object — then does `getattr(attr, 'data', attr)` to
    unwrap a `.data` field from state objects that *are* wrapped (like
    `ObjectState`/`ProbeState`). Verified directly: under the **NumPy** backend,
    `ndarray.data` is a real (buffer-protocol) attribute, so this fallback incorrectly
    extracts a `memoryview` instead of the array, and `apply_iter` fails with
    `AttributeError: 'memoryview' object has no attribute 'reshape'`. Under a **JAX**
    array, which has no `.data` attribute, the fallback correctly returns the array
    itself, and `apply_iter` completes successfully — verified on both backends. Because
    tilt refinement is a gradient-engine-only optimization requiring JAX or Torch
    ([overview](../overview.md#engine-families-and-backends)), `tilt_gaussian` is likely to
    work in its typical use case, but would fail if attached to a NumPy-backed engine (e.g.
    a conventional engine with a static tilt from `tilt_propagators` but no
    tilt-refinement solver).

The generated reference for this hook family does not exist yet (Phase 2 of the
[implementation plan](../../design/implementation-plan.md)); this table, including the two
restrictions above, was verified directly against `phaser/hooks/regularization.py` and
`phaser/engines/common/regularizers.py` by executing each implementation against a
synthetic `ReconsState`.

## Minimal custom implementation

An iteration constraint needs only `init_state` and `apply_iter`. This example clamps the
object's phase to a fixed range after every iteration:

```python
class ClampObjectPhaseRange:
    """After each iteration, clamp the object phase to +/- max_phase radians."""
    def __init__(self, args, props):
        self.max_phase = props['max_phase']

    def init_state(self, sim):
        return None

    def apply_iter(self, sim, state):
        xp = numpy
        amp = xp.abs(sim.object.data)
        phase = xp.clip(xp.angle(sim.object.data), -self.max_phase, self.max_phase)
        sim.object.data = (amp * xp.exp(1j * phase)).astype(sim.object.data.dtype)
        return (sim, state)
```

Run against a synthetic `ReconsState` (object phase perturbed to `~1.28` rad, NumPy
backend, `max_phase=1.0`):

```text
max abs phase before iter constraint: 1.2831853071795865
max abs phase after iter constraint: 1.0
```

confirming `apply_iter` is called and clamps the phase as intended.

## YAML invocation

Built-in short name:

```yaml
engines:
  - type: gradient   # or type: conventional
    # ...
    slices:
      n: 10
      total_thickness: 200
    iter_constraints:
      - type: layers
        sigma: 50.0
        weight: 0.9
```

External reference — properties are passed through as a plain dictionary and are **not**
schema-validated:

```yaml
    iter_constraints:
      - type: "my_package.my_constraints:ClampObjectPhaseRange"
        max_phase: 1.0
```

Verified with `phaser validate`: a plan with an `iter_constraints` list containing a
built-in name validates successfully under both a `gradient` and a `conventional` engine.

## Engine and backend restrictions

**Both engine families.** `ConventionalEnginePlan.iter_constraints` and
`GradientEnginePlan.iter_constraints` are both typed `t.List[IterConstraintHook]`
(`phaser/plan.py:147,156`) — unlike [cost regularizers](cost-regularizers.md), which are
gradient-only. No backend is enforced by the constraint mechanism itself; see the two
`Restriction` admonitions above for `tilt_gaussian`'s backend dependence and
`opr_gaussian`'s unconditional failure.

## Optional dependencies

None declared through the hook mechanism (no entry in `IterConstraintHook.known` carries
an optional-dependency tuple). `tilt_gaussian`/`opr_gaussian` import `scipy.spatial.KDTree`
at call time, but `scipy` is already a core (non-optional) dependency of Phaser
(`pyproject.toml`), not a hook-declared optional one.

## Testing pattern

An iteration constraint can be tested without running any engine: construct the object
directly (`ic = ClampObjectPhaseRange(None, {...})`), build a synthetic `ReconsState`, call
`ic.init_state(sim)` once, then `ic.apply_iter(sim, state)` and assert on the mutated
`sim` — the pattern used to produce the output above, and the same pattern that surfaced
the `opr_gaussian`/`tilt_gaussian` restrictions above (calling the built-in directly
against a synthetic state, rather than only reading its source). `tests/` currently has no
coverage of `phaser/hooks/regularization.py`
([implementation checklist](../../design/implementation-checklist.md), blocker B12); a new
custom constraint should add its own unit test in this style.

## Maintainer sources

- `phaser/hooks/regularization.py`
- `phaser/engines/common/regularizers.py`
- `phaser/engines/common/simulation.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/gradient/run.py`
- `phaser/plan.py`
