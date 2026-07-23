# Regularization

"Regularization" in Phaser is three mechanisms, distinguished by *when* and *how* they act
on state. This page is about which lifecycle addresses which decision and which built-in to
reach for; the mechanics and executed examples are on the `architecture/hooks/` pages.

!!! warning "Terminology"
    The schema field is **`regularizers`**, never `regularizations`
    (`GradientEnginePlan.regularizers`, `phaser/plan.py:154`). A plan using
    `regularizations:` fails `phaser validate`. Use *regularizer*, not *regularization*, as
    the name of a hook or a plan field; "regularization" remains fine as the general
    scientific concept, as used in this page's title.

## The three lifecycles, in scientist's terms

| Lifecycle | Plan field | What it does | Engines |
| --- | --- | --- | --- |
| [Cost regularizer](#cost-regularizers-a-soft-penalty-in-the-objective) | `regularizers` | Adds a differentiable penalty term to the loss the gradient engine minimizes — a *soft* preference | **Gradient only** |
| [Group constraint](#group-and-iteration-constraints-hard-projections) | `group_constraints` | Directly mutates state (object, probe, ...) after every group — a *hard* projection back onto some allowed set | Both |
| [Iteration constraint](#group-and-iteration-constraints-hard-projections) | `iter_constraints` | Directly mutates state after every full iteration — the same kind of hard projection, at a coarser cadence | Both |

Use a **cost regularizer** for a soft nudge (it competes with the data fit, weighted by
`cost`); use a **constraint** for a hard rule that must always hold (cap object amplitude,
limit probe support), which unconditionally projects state every time it runs.

!!! warning "Restriction — cost regularizers are gradient-only"
    `GradientEnginePlan` declares `regularizers`; `ConventionalEnginePlan` does not, so a
    `regularizers:` key on a conventional engine fails `phaser validate` (a conventional
    solver differentiates no scalar loss to add to). Group and iteration constraints are
    declared identically on both engines (`phaser/plan.py`) — no engine restriction.

## Cost regularizers: a soft penalty in the objective

A cost regularizer's loss is computed once per group and added to the
[noise-model](noise-models.md) detector loss to form the total the gradient engine
differentiates. `cost` is a **dimensionless weight** relative to that loss — no physical unit
is documented in code (the "in electrons" claim could not be verified). Since the detector
loss is only meaningful in scaled counts, a fixed `cost`'s effective strength shifts with
[pattern scaling](noise-models.md#prerequisite-patterns-must-be-scaled-to-physical-counts).

| Built-in | Penalizes | `cost` unit | Notes |
| --- | --- | --- | --- |
| `obj_l1` | `sum(abs(object - 1))` | dimensionless weight | L1 distance of object amplitude/phase from the vacuum value `1` |
| `obj_l2` | `sum(abs(object - 1)^2)` | dimensionless weight | L2 (squared) distance from vacuum |
| `obj_phase_l1` | `sum(abs(angle(object)))` | dimensionless weight | L1 norm of object phase |
| `obj_recip_l1` | `sum(abs(fft2(prod(object, axis=0))))` | dimensionless weight | L1 norm of the projected object's diffracted amplitude |
| `obj_tv` | isotropic total variation of the object | dimensionless weight | has an `eps` field (default `1e-8`) avoiding a non-differentiable point at zero gradient |
| `obj_tikh` / `obj_tikhonov` | squared finite differences of the object along `y`/`x` | dimensionless weight | two aliases, one implementation |
| `layers_tv` | total variation across the **slice** axis | dimensionless weight | `0` if the object has fewer than 2 slices |
| `layers_tikh` / `layers_tikhonov` | squared finite differences across the slice axis | dimensionless weight | `0` if fewer than 2 slices; two aliases |
| `probe_phase_tikh` / `probe_phase_tikhonov` | squared finite differences of the probe's Fourier-space phase | dimensionless weight | two aliases |
| `probe_recip_tv` | isotropic total variation of the probe's Fourier-space amplitude | dimensionless weight | has an `eps` field |
| `probe_recip_tikh` / `probe_recip_tikhonov` | squared finite differences of the probe's Fourier-space amplitude | dimensionless weight | two aliases |

Each built-in scales its penalty by the fraction of scan positions in the group, except the
probe-only ones (fixed scale). Property tables:
[generated reference](../../generated/hooks/cost-regularizer.md); lifecycle and executed
example: [Cost-regularizer hooks](../../architecture/hooks/cost-regularizers.md).

Applied once per group, gradient engine only. No bound on `cost`: `examples/si_grad.yaml`
combines `obj_l2: 0.4` and `obj_tikh: 0.2`; `czo_grad.yaml`/`prsco3_grad.yaml` add
`layers_tikh` at `5.0e+2` alongside smaller weights — the right relative scale is
data-dependent.

### Minimal example

```yaml
engines:
  - type: gradient
    # ...
    regularizers:
      - type: obj_l2
        cost: 0.4
      - type: obj_tikh
        cost: 0.2
      - type: layers_tikh
        cost: 5.0e+2
```

adapted from `examples/si_grad.yaml`.

## Group and iteration constraints: hard projections

Both kinds share most built-ins and differ only in cadence: a **group constraint** runs
after every group, an **iteration constraint** after every full iteration; both on both
engines. Neither returns a loss — the only effect is the mutated state.

| Built-in | Registered as | Effect | Key property |
| --- | --- | --- | --- |
| `clamp_object_amplitude` | group and iteration | Clamps object amplitude into `[min, max]` (or caps at `max` if `amplitude` is a scalar), preserving phase | `amplitude` (default `1.1`) |
| `limit_probe_support` | group and iteration | Hard reciprocal-space aperture on the probe at `max_angle` | `max_angle` (mrad, required) |
| `obj_low_pass` | group and iteration | Hard low-pass filter of the object in Fourier space | `max_freq` (cycles/pixel, default `0.4`; Nyquist is `0.5`) |
| `obj_gaussian` | group and iteration | Soft (Gaussian) low-pass filter of the object, blended by `weight` | `sigma` (Å), `weight` (default `0.9`) |
| `remove_phase_ramp` | group and iteration | Removes a linear phase ramp from the object's phase, restricted to its active region | none (accepts an empty properties object) |
| `nonneg_object_phase` | group and iteration | Pushes negative object-phase values toward zero, blended by `weight` | `weight` (default `1.0`) |
| `layers` | iteration only | Gaussian-blurs the object across the slice axis (multislice only; no-op if fewer than 2 slices) | `sigma` (Å, default `50.0`), `weight` (default `0.9`) |
| `tilt_gaussian` | iteration only | Spatially Gaussian-blurs `state.tilt` across scan positions | `sigma` (Å), `weight` (default `0.9`) — see restriction below |
| `opr_gaussian` | iteration only | Intended to spatially Gaussian-blur a per-position "OPR" state | **not usable — see restriction below** |

Property tables: [generated group-constraint](../../generated/hooks/group-constraint.md) /
[iter-constraint](../../generated/hooks/iter-constraint.md) references; full lifecycle and
executed examples: [Group constraints](../../architecture/hooks/group-constraints.md) /
[Iteration constraints](../../architecture/hooks/iteration-constraints.md).

!!! warning "Restriction — `opr_gaussian` is not usable"
    `ReconsState` has no `opr` attribute anywhere in the codebase. Calling `opr_gaussian`
    against a real reconstruction state was verified to raise
    `AttributeError: Can't get path 'opr.data' in reconstruction state`. This looks like a
    registered hook for an unimplemented feature (per-scan-position "OPR" state) — **do not
    configure `opr_gaussian`**; it is not usable in the current codebase.

!!! warning "Restriction — `tilt_gaussian` is backend-dependent"
    `tilt_gaussian`'s implementation resolves an internal `.data` attribute in a way that
    fails under the **NumPy** backend (`AttributeError: 'memoryview' object has no attribute
    'reshape'`) but works under **JAX**, verified by direct execution on both. Because tilt
    refinement itself requires the gradient engine's `jax`/`torch` backend, `tilt_gaussian`
    is likely to work in its typical use case, but would fail if attached to a NumPy-backed
    engine (for example, a conventional engine applying a static tilt with no tilt-refinement
    solver).

Group constraints run right after that group's solver update; iteration constraints run
after all of an iteration's solver updates, before any position-solver step. No bound on any
numeric field; typical values are in the example below.

### Minimal example

```yaml
engines:
  - type: conventional   # or type: gradient
    # ...
    group_constraints:
      - type: clamp_object_amplitude
        amplitude: 1.1
    iter_constraints:
      - type: limit_probe_support
        max_angle: 26.0
```

adapted from `examples/mos2_lsqml.yaml` (`clamp_object_amplitude`) and
`examples/czo_grad.yaml` (`limit_probe_support`, `max_angle: 26.0`).

## Interactions

- A cost regularizer's effective strength moves with the noise model's count scale (see
  [Noise models](noise-models.md#prerequisite-patterns-must-be-scaled-to-physical-counts));
  re-tune `cost` if you change pattern scaling.
- `layers`/`layers_tikh`/`layers_tv` only have an effect with a multislice object (2 or more
  slices) — see [Simulation geometry](simulation-geometry.md) for `slices`.
- `limit_probe_support`'s `max_angle` and probe-mode count interact with
  [Simulation geometry](simulation-geometry.md)'s probe sampling — an aperture set smaller
  than the actual probe convergence angle will clip real signal, not just noise.
- Fields with a `float` value in the tables above (for example `weight`, `sigma`,
  `max_freq`) are plain floats in every built-in shown — none of the constraint or
  regularizer properties are schedule-valued in the current schema, unlike solver
  coefficients (see [Schedules and flags](schedules-and-flags.md)).

## Maintainer sources

- `phaser/hooks/regularization.py`
- `phaser/engines/common/regularizers.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/conventional/run.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/engines/common/simulation.py`
- `phaser/plan.py`
- `docs/architecture/hooks/cost-regularizers.md`
- `docs/architecture/hooks/group-constraints.md`
- `docs/architecture/hooks/iteration-constraints.md`
- `docs/architecture/state-and-conventions.md`
- `docs/generated/hooks/cost-regularizer.md`
- `docs/generated/hooks/group-constraint.md`
- `docs/generated/hooks/iter-constraint.md`
- `examples/si_grad.yaml`
- `examples/czo_grad.yaml`
- `examples/prsco3_grad.yaml`
- `examples/mos2_lsqml.yaml`
