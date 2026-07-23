# Noise-model hooks

A [**noise model**](../../concepts/glossary.md#noise-model) supplies the statistical
relationship between a simulated diffraction intensity and a measured pattern: a scalar
loss, and (for conventional solvers) a reciprocal-space wave update. Every engine plan
declares exactly one noise-model hook — `ConventionalEnginePlan.noise_model` and
`GradientEnginePlan.noise_model` are both a single `NoiseModelHook`, not a list
(`phaser/plan.py:142,151`).

!!! warning "Restriction"
    The built-in **Poisson** noise model implements only half of the `NoiseModel`
    interface: `calc_loss` works, but `calc_wave_update` raises `NotImplementedError()`
    unconditionally (`phaser/engines/common/noise_models.py:104-112`). The gradient engine
    calls only `calc_loss` (`phaser/engines/gradient/run.py:460`); the conventional
    engines' solvers call only `calc_wave_update` (`phaser/engines/conventional/solvers.py:255,490`).
    **Poisson therefore works with the gradient engine and crashes at runtime on the
    conventional engines (ePIE, LSQML)** — the schema accepts `noise_model: poisson` for
    either engine type without complaint at validation time, so this is invisible until the
    noise model is actually called. **Amplitude** and **Anscombe** implement both methods
    and work with both engine families.

## Lifecycle point

A noise model is constructed once per engine run, immediately after the engine's other
setup and before the first group: `noise_model = props.noise_model(None)`
(`phaser/engines/gradient/run.py:164`; identically `phaser/engines/conventional/run.py:22`).
Its `init_state` is called once at construction (directly in the gradient engine;
indirectly through `SimulationState.__init__` for the conventional engines,
`phaser/engines/common/simulation.py:129`). After that:

- the **gradient engine** calls `calc_loss` once per group, inside the JIT-traced
  `run_model` function (`phaser/engines/gradient/run.py:460`), to produce the
  differentiable objective that `tree.grad` differentiates;
- the **conventional engines** (ePIE, LSQML) call `calc_wave_update` once per group,
  inside `epie_run`/`lsqml_run` (`phaser/engines/conventional/solvers.py:490,255`), to get
  the reciprocal-space update `chi` that both solvers back-propagate through their slices
  to update the object and probe.

Both methods "may be called in a JAX jit context, so must have no side effects" (docstring,
`phaser/hooks/solver.py:31-58`) — a custom noise model must be a pure function of its
arguments and its own `state`, never mutating external state or performing I/O.

Independent of the configured noise model, the conventional engines separately compute a
fixed squared-intensity-difference `errors` value per group for progress reporting
(`phaser/engines/conventional/solvers.py:253,489`) — not the noise-model loss, and unaffected
by switching noise models; only the wave update `chi` driving the object/probe update comes
from the configured noise model.

## Callable signature and property schema

`NoiseModelHook` is `Hook[None, NoiseModel]` (`phaser/hooks/solver.py:62-63`): it is called
with no arguments (`props.noise_model(None)`) and returns a `NoiseModel` instance. The
`NoiseModel` protocol (`phaser/hooks/solver.py:26-59`) is:

```python
class NoiseModel(HasState[StateT], t.Protocol[StateT]):
    @classmethod
    def name(cls) -> str: ...

    def init_state(self, sim: ReconsState) -> StateT: ...

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating],
        state: StateT,
    ) -> t.Tuple[Float, StateT]: ...

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating],
        state: StateT,
    ) -> t.Tuple[NDArray[numpy.complexfloating], StateT]: ...
```

Every implementation's `__init__` takes `(args: None, props: <its properties dataclass>)`,
matching how `Hook.resolve()(args, props=...)` calls it (`phaser/hooks/hook.py:61-62`).

### Property schemas

All three built-in property classes derive from `AmplitudeNoisePlan` (`phaser/plan.py:101-113`):

| Field | Type | Default | Used by | Meaning |
| --- | --- | --- | --- | --- |
| `gaussian_variance` | `float` | `0.1` | `amplitude`, `anscombe` (not read by `poisson`) | Added to `1.0` to form the fixed denominator `self.var` that scales the amplitude loss (`phaser/engines/common/noise_models.py:22`) |
| `eps` | `float` | `1.0e-3` (amplitude/anscombe); `1.0e-3` (poisson, redeclared) | all three | Amplitude/Anscombe: added to the denominator of the wave-update ratio to avoid division by zero. Poisson: added inside both `log` terms of the negative log-likelihood, for the same reason |
| `offset` | `float` | `0.0` (amplitude); `0.375` (anscombe, overridden — the standard Anscombe-transform offset) | `amplitude`, `anscombe` (schema field only, unused by `poisson`) | Added under both square roots before differencing, stabilizing the transform for low counts |

`PoissonNoisePlan` (`phaser/plan.py:111-112`) inherits the `gaussian_variance` and `offset`
fields from `AmplitudeNoisePlan` for schema reasons (it is a subclass), but
`PoissonNoiseModel.__init__` (`phaser/engines/common/noise_models.py:80-81`) reads only
`props.eps` — setting `gaussian_variance` or `offset` under a `poisson` noise model has no
effect, and nothing rejects those fields under `poisson` at validation time.

Registration: `NoiseModelHook.known['amplitude'|'anscombe'|'poisson']` in `phaser/plan.py:115-117`.

## Accepted state and returned value

Arrays passed to `calc_loss`/`calc_wave_update` (shapes as constructed by the calling engine
— `phaser/engines/gradient/run.py:457-461`, `phaser/engines/conventional/solvers.py:247-255`):

- `model_wave` — simulated exit wave in reciprocal space, complex, shape `(group, modes, y, x)`.
- `model_intensity` — `sum(abs(model_wave)**2, axis=modes)`, real, shape `(group, y, x)`
  (gradient engine) or `(group, 1, y, x)` (conventional engines, which keep a singleton
  mode axis via `keepdims=True`).
- `exp_patterns` — measured diffraction intensities in the same shape as `model_intensity`,
  in physical particle counts
  ([Intensity and count scaling](../state-and-conventions.md#intensity-and-count-scaling) —
  matters most for Poisson).
- `mask` — detector pattern mask, shape `(y, x)`, broadcasting against the group axis;
  `1.0` for valid detector pixels.
- `state` — the noise model's own carried state (`None` for every built-in; see below).

`calc_loss` returns `(loss, state)`: a scalar per call, summed over every axis of its
inputs (detector pixels and whatever positions are in the group). The engine, not the noise
model, later divides the iteration's accumulated loss by the total scan-position count
(`groups.n_pos`, `phaser/engines/gradient/run.py:306`) to report a per-position average — a
custom noise model's `calc_loss` should follow the same sum-per-call convention rather than
pre-averaging.

`calc_wave_update` returns `(chi, state)`: `chi` is the reciprocal-space update the same
shape as `model_wave`, back-propagated through the slice stack by the calling conventional
solver.

All three built-ins are stateless: `init_state` returns `None`, and `state` is threaded
through unchanged. A noise model with actual per-call state (rare) would return a non-`None`
value from `init_state` and update it inside `calc_loss`/`calc_wave_update`, exactly like a
[solver](solvers.md).

## Built-in implementations

Registered in `NoiseModelHook.known` (`phaser/plan.py:115-117`; implementations in
`phaser/engines/common/noise_models.py`):

| Name | Class | Loss form | Engine compatibility |
| --- | --- | --- | --- |
| `amplitude` | `AmplitudeNoiseModel` | Squared difference of offset square-root amplitudes, scaled by `1 / (1 + gaussian_variance)` | gradient and conventional |
| `anscombe` | `AnscombeNoiseModel` (subclasses `AmplitudeNoiseModel`, only changing the `offset` default to `0.375`) | Same form as `amplitude`, with the Anscombe-transform offset | gradient and conventional |
| `poisson` | `PoissonNoiseModel` | Poisson negative log-likelihood between counts and simulated intensity | **gradient only** — see restriction above |

`AnscombeNoiseModel.__init__` just calls `super().__init__(args, props)` — it is the same
implementation as `AmplitudeNoiseModel` with a different default `offset` and `name()`.

## Minimal custom implementation

A custom noise model implements `calc_loss` and `calc_wave_update` (and a trivial
`init_state` if it carries no state). The `NoiseModel` protocol is structural — a class
needs no base class, only the methods above. This example compares intensities directly,
with a fixed-variance Gaussian loss, instead of the built-ins' offset-amplitude form:

```python
import typing as t
import numpy
from numpy.typing import NDArray

from phaser.utils.num import get_array_module, Float
from phaser.state import ReconsState
from phaser.types import Dataclass


class GaussianNoiseProps(Dataclass):
    eps: float = 1.0e-3


class GaussianNoiseModel:
    """Compares intensities directly, with a fixed Gaussian variance."""

    @classmethod
    def name(cls) -> str:
        return "gaussian_intensity"

    def __init__(self, args: None, props: GaussianNoiseProps):
        self.eps: float = props.eps

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating],
        state: None,
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)
        loss = xp.sum(mask * (exp_patterns - model_intensity) ** 2)
        return (loss.astype(exp_patterns.dtype), state)

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating],
        state: None,
    ) -> t.Tuple[NDArray[numpy.complexfloating], None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)
        update = mask * 2.0 * (exp_patterns - model_intensity)
        return (update * model_wave, state)
```

This was executed directly (outside any engine) against small synthetic arrays to confirm
both methods run and return correctly-shaped values:

```python
model = GaussianNoiseModel(None, GaussianNoiseProps(eps=1e-3))

rng = numpy.random.default_rng(0)
shape = (3, 8, 8)  # (group, y, x)
model_wave = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(numpy.complex64)
model_intensity = numpy.abs(model_wave) ** 2
exp_patterns = model_intensity + rng.normal(scale=0.01, size=shape).astype(numpy.float32)
mask = numpy.ones(shape[-2:], dtype=numpy.float32)

state = model.init_state(None)
loss, state = model.calc_loss(model_wave, model_intensity, exp_patterns, mask, state)
# loss: 0.0191166, dtype float32

chi, state = model.calc_wave_update(model_wave, model_intensity, exp_patterns, mask, state)
# chi.shape == model_wave.shape == (3, 8, 8), complex64
```

Both calls completed and returned finite, correctly-shaped, correctly-typed results.

## YAML invocation

Built-in, by registered name:

```yaml
engines:
  - type: gradient
    noise_model:
      type: poisson
      eps: 2.0
    # ... solvers, regularizers, etc.
```

```yaml
engines:
  - type: conventional
    noise_model:
      type: anscombe
      eps: 1.0e-4
    solver:
      type: lsqml
```

External, by `package.module:function` reference — the properties dictionary that follows
is passed straight through to `GaussianNoiseModel.__init__` as a plain `dict` and is
**not** schema-validated (no type checking, no defaults filled in; a typo in a property
name surfaces only when the noise model is constructed, not at plan-parse time):

```yaml
engines:
  - type: gradient
    noise_model:
      type: "my_package.my_module:GaussianNoiseModel"
      eps: 1.0e-3
```

## Engine and backend restrictions

- **Poisson is gradient-engine only** (restriction above). Do not configure
  `noise_model: poisson` for a `conventional` engine.
- **Amplitude and Anscombe** work with both engine families and aren't backend-restricted
  by the noise-model mechanism itself; the gradient engine's own backend restriction (JAX
  or Torch) applies regardless of noise model
  ([Engine families and backends](../overview.md#engine-families-and-backends)).
- Both `calc_loss` and `calc_wave_update` must be traceable under JAX's `jit`/`grad` when
  used with the gradient engine on the JAX backend — no Python-level control flow depending
  on array *values* (as opposed to static shapes), and no side effects.

## Optional dependencies

None. All three built-in noise models depend only on NumPy-compatible array operations
already required by Phaser's core.

## Testing pattern

A noise model has no dependency on `ReconsState`, engines, or backends beyond whichever
array module its arrays already use — test it directly with small synthetic arrays, as
shown in [Minimal custom implementation](#minimal-custom-implementation) above:

1. Construct the model with its properties dataclass directly (bypass YAML/plan parsing).
2. Build small `model_wave`/`model_intensity`/`exp_patterns`/`mask` arrays of matching
   shape (a `(group, y, x)` triple is enough — no real object/probe/scan needed).
3. Call `init_state(None)` (a real `ReconsState` is only required if your model actually
   reads it), then `calc_loss` and `calc_wave_update`, and assert the returned loss is a
   finite scalar and the returned `chi` is complex-valued with the same shape as
   `model_wave`.
4. If the noise model is meant for the gradient engine, additionally confirm it runs
   under `jax.jit`/`jax.grad` (or the Torch equivalent) with the same synthetic arrays,
   since the engine always calls it inside a JIT-traced function
   (`phaser/engines/gradient/run.py:354-358`).

## Maintainer sources

- `phaser/hooks/solver.py`
- `phaser/plan.py`
- `phaser/engines/common/noise_models.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/conventional/solvers.py`
- `phaser/engines/common/simulation.py`
