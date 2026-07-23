# Group-constraint hooks

A **group constraint** mutates reconstruction state directly, on a fixed cadence, rather
than contributing a differentiable term to an objective. This is one of three
[**regularizer**](../../concepts/glossary.md#regularizer) lifecycles defined in
`phaser/hooks/regularization.py`. Contrast this page (mutation after every group, on
**both** engine families) with [Cost regularizers](cost-regularizers.md) (a differentiable
loss term, gradient engine only) and [Iteration constraints](iteration-constraints.md)
(mutation after every full iteration instead of every group). Several implementations are
shared between this page and iteration constraints — the built-ins table below states
which lifecycle each name is registered under.

## Lifecycle point

A group constraint runs **after every group, on both engine families**:

- in the **gradient engine**, inside `run_group` (itself JIT-compiled when the backend is
  JAX), immediately after that group's per-group solvers have applied their update to
  `state` (`phaser/engines/gradient/run.py:398-410`);
- in the **conventional engines**, immediately after each group's ePIE or LSQML update,
  via `SimulationState.apply_group_constraints(group)`
  (`phaser/engines/conventional/solvers.py:128` for LSQML, `:402` for ePIE, calling
  `phaser/engines/common/simulation.py:140-149`).

Both call sites pass the constraint the same two things: the current group's scan-position
indices and the state as it stood immediately after that group's solver update — a group
constraint never sees an intermediate, not-yet-updated state.

## Callable signature and property schema

`GroupConstraintHook` (`Hook[None, GroupConstraint]`, `phaser/hooks/regularization.py:92`)
is a **two-step** hook, exactly like [cost regularizers](cost-regularizers.md#callable-signature-and-property-schema):
the hook is called once, with `None`, to construct a `GroupConstraint` instance; that
instance's `apply_group` is then called after every group.

```python
@t.runtime_checkable
class GroupConstraint(HasState[StateT], t.Protocol[StateT]):
    def apply_group(
        self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT
    ) -> t.Tuple['ReconsState', StateT]:
        ...
```

(`phaser/hooks/regularization.py:15-18`.) `HasState[StateT]` additionally requires
`init_state(self, sim: ReconsState) -> StateT`, called once when the engine starts to
produce the constraint's initial carried state (for example, a precomputed Fourier-space
filter that does not change between calls).

Property schemas used by the built-ins below:

| Property schema | Fields |
| --- | --- |
| `ClampObjectAmplitudeProps` | `amplitude: t.Union[float, t.List[t.Optional[float]]] = 1.1` — a single value is treated as a maximum; a two-element list `[min, max]` (either may be `None`) sets both bounds |
| `LimitProbeSupportProps` | `max_angle: float` (mrad) |
| `ObjLowPassProps` | `max_freq: float = 0.4` (cycles/pixel; Nyquist is `0.5`) |
| `GaussianProps` | `sigma: float` (Å, standard deviation), `weight: float = 0.9` |
| `NonNegObjectPhaseProps` | `weight: float = 1.0` |
| (no properties) | `remove_phase_ramp` accepts an empty properties object (`t.Dict[str, t.Any]`) |

(`phaser/hooks/regularization.py:36-74`.)

## Accepted state and returned value

`apply_group(group, sim, state)` receives:

- `group` — the current group's scan-position indices, `NDArray[numpy.integer]`, shape
  `(group_size,)`;
- `sim` — the full `ReconsState`, already updated by this group's solver step;
- `state` — the constraint's own carried state from `init_state` or the previous call.

It returns `(sim, new_state)` — the (possibly mutated) `ReconsState` and the constraint's
updated carried state. Unlike a cost regularizer, there is no loss value: the only effect
is the returned state.

## Built-in implementations

Registered in `GroupConstraintHook.known` (`phaser/hooks/regularization.py:92-101`) and
implemented in `phaser/engines/common/regularizers.py`. Every name below is *also*
registered as an [iteration constraint](iteration-constraints.md#built-in-implementations)
except where noted, since most implementations define both `apply_group` and `apply_iter`
(the latter usually just calling the former, or vice versa).

| Name | Effect |
| --- | --- |
| `clamp_object_amplitude` | Clamps the object's amplitude into `[min, max]` (or just caps it at `max` if `amplitude` is a scalar), rescaling each pixel's complex value to the clamped amplitude while preserving phase (`ClampObjectAmplitude.apply_group`/`apply_iter`, identical either way) |
| `limit_probe_support` | Applies a hard reciprocal-space aperture to the probe at the angle (mrad) corresponding to `max_angle`, computed once in `init_state` from the probe's sampling and `sim.wavelength`, then applied every call as `ifft2(fft2(probe) * mask)` |
| `obj_low_pass` | Hard low-pass filters the object in Fourier space at `max_freq` cycles/pixel |
| `obj_gaussian` | Soft (Gaussian) low-pass filter of the object in Fourier space, blended with the original by `weight` — `1.0 - weight * (1.0 - gaussian_filter)`, so `weight=1` applies the filter fully and `weight=0` is a no-op |
| `remove_phase_ramp` | Removes a linear phase ramp from the object's phase, restricted to the object's active region mask (`ObjectSampling.get_region_mask`) |
| `nonneg_object_phase` | Pushes negative object-phase values toward zero, blended by `weight` (`weight=1` fully clips negative phase to zero; `weight=0` is a no-op) |

`layers`, `opr_gaussian`, and `tilt_gaussian` are **not** registered as group constraints —
their implementations (`RegularizeLayers`, `UnstructuredGaussian`) define only
`apply_iter`, not `apply_group`; see
[Iteration constraints](iteration-constraints.md#built-in-implementations).

The generated reference for this hook family does not exist yet (Phase 2 of the
[implementation plan](../../design/implementation-plan.md)); this table was verified
directly against `phaser/hooks/regularization.py` and
`phaser/engines/common/regularizers.py`.

## Minimal custom implementation

A group constraint needs only `init_state` and `apply_group`. This example rescales the
object so its mean amplitude matches a target value after every group:

```python
class RescaleObjectMeanAmplitude:
    """After each group, rescale the object so its mean amplitude equals a target."""
    def __init__(self, args, props):
        self.target = props['target']

    def init_state(self, sim):
        return None

    def apply_group(self, group, sim, state):
        xp = numpy
        amp = xp.abs(sim.object.data)
        mean_amp = xp.mean(amp)
        scale = self.target / mean_amp if mean_amp > 0 else 1.0
        sim.object.data = sim.object.data * scale
        return (sim, state)
```

Run against a synthetic `ReconsState` (object perturbed to mean amplitude `0.5`, NumPy
backend, `target=1.0`):

```text
mean object amp before group constraint: 0.5
mean object amp after group constraint: 1.0
```

confirming `apply_group` is called and rescales the object as intended.

## YAML invocation

Built-in short name:

```yaml
engines:
  - type: gradient   # or type: conventional
    # ...
    group_constraints:
      - type: clamp_object_amplitude
        amplitude: 1.1
      - type: limit_probe_support
        max_angle: 30.0
```

External reference — properties are passed through as a plain dictionary and are **not**
schema-validated:

```yaml
    group_constraints:
      - type: "my_package.my_constraints:RescaleObjectMeanAmplitude"
        target: 1.0
```

Verified with `phaser validate`: a plan with a `group_constraints` list containing a
built-in name validates successfully under both a `gradient` and a `conventional` engine.

## Engine and backend restrictions

**Both engine families.** `ConventionalEnginePlan.group_constraints` and
`GradientEnginePlan.group_constraints` are both typed
`t.List[GroupConstraintHook]` (`phaser/plan.py:146,155`) — unlike
[cost regularizers](cost-regularizers.md), which are gradient-only. No backend is
enforced by the constraint mechanism itself, but a constraint attached to the gradient
engine runs inside `run_group`'s JIT-traced region when the backend is JAX
(`phaser/engines/gradient/run.py:354-358`), so a custom implementation intended for the
gradient engine should use backend-generic operations (`get_array_module`) and avoid
Python control flow that depends on concrete array values — see
[the JAX guide](../jax.md) once written.

!!! warning "Restriction"
    `tilt_gaussian` and `opr_gaussian` are registered only as
    [iteration constraints](iteration-constraints.md#built-in-implementations), not as
    group constraints — see that page for two verified issues with their current
    implementation.

## Optional dependencies

None. No entry in `GroupConstraintHook.known` declares an optional-dependency tuple;
every built-in listed above uses only backend-generic (`xp`) array operations.

## Testing pattern

A group constraint can be tested without running any engine: construct the object
directly (`gc = RescaleObjectMeanAmplitude(None, {...})`), build a synthetic `ReconsState`,
call `gc.init_state(sim)` once, then `gc.apply_group(group, sim, state)` and assert on the
mutated `sim` — this is exactly the pattern used to produce the output above. `tests/`
currently has no coverage of `phaser/hooks/regularization.py` (recorded in the
[implementation checklist](../../design/implementation-checklist.md), blocker B12); a new
custom constraint should add its own unit test in this style.

## Maintainer sources

- `phaser/hooks/regularization.py`
- `phaser/engines/common/regularizers.py`
- `phaser/engines/common/simulation.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/engines/gradient/run.py`
- `phaser/plan.py`
