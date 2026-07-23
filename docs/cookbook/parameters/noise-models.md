# Noise models

A [noise model](../../concepts/glossary.md#noise-model) tells a reconstruction how to
compare a simulated diffraction intensity against a measured pattern: it supplies the
scalar loss the gradient engine minimizes, and (for the conventional engines) the
reciprocal-space update that drives the object/probe correction. Every engine declares
exactly one noise-model hook — `noise_model:` is a single hook, never a list — so choosing
a noise model means choosing the statistical assumption your whole engine run makes about
detector noise.

!!! warning "Restriction — Poisson works with the gradient engine only"
    The built-in **`poisson`** noise model implements only half of the noise-model
    interface: its loss calculation works, but its conventional-engine wave-update method
    raises `NotImplementedError()` unconditionally
    (`phaser/engines/common/noise_models.py:104-112`). The gradient engine only ever calls
    the loss calculation, so `poisson` works there; the conventional engines (ePIE, LSQML)
    only ever call the wave-update method, so **a conventional engine configured with
    `noise_model: {type: poisson}` fails at runtime**, even though `phaser validate` accepts
    it — the plan schema does not encode this restriction. Use `amplitude` or `anscombe`
    with a conventional engine instead. See
    [Noise-model hooks](../../architecture/hooks/noise-models.md#callable-signature-and-property-schema)
    for the underlying interface.

## Prerequisite: patterns must be scaled to physical counts

All three noise models assume `exp_patterns` (the measured diffraction intensities) are in
physical particle counts (electrons or photons), not an arbitrary detector intensity scale.
Phaser warns during initialization if the mean total pattern intensity is below `5.0`,
suggesting the `scale` or `poisson` `post_load` hooks if the data needs rescaling
(`phaser/execute.py:367-374`; see
[Intensity and count scaling](../../architecture/state-and-conventions.md#intensity-and-count-scaling)).
This matters most for **Poisson**, whose loss is a correct statistical model only if the
compared values really are counts — but the effective strength of `amplitude`/`anscombe`
and of every [regularizer](regularization.md) also shifts with pattern scale, since a
regularizer's `cost` weight is fixed while the noise-model loss it competes against is not.

Do not confuse the `post_load` hook named `poisson` (`add_poisson_noise` — adds simulated
shot noise to already-noiseless synthetic patterns, useful when preparing test data) with
the noise-model hook of the same name documented below (`PoissonNoiseModel` — a loss
function comparing measured and simulated intensities). They share a registered name in two
unrelated hook families.

```yaml
post_load:
  - type: scale
    scale: 1.0e+6   # multiply patterns to bring them into physical-count units
```

## Choosing a noise model

| Name | Statistical assumption | Typical dose regime |
| --- | --- | --- |
| [`amplitude`](#amplitude-and-anscombe) | Fixed-variance Gaussian noise on the square-root (amplitude) of the intensity | Moderate-to-high count data, where the square-root transform is a reasonable variance-stabilizing approximation |
| [`anscombe`](#amplitude-and-anscombe) | Same fixed-variance Gaussian form as `amplitude`, using the classic Anscombe offset (`3/8`) instead of `0` | Lower-count data, where the plain square-root transform is a poorer approximation and the Anscombe offset corrects for it |
| [`poisson`](#poisson) | Exact Poisson negative log-likelihood between simulated intensity and measured counts | Any dose, but the model where a Gaussian approximation is least accurate is exactly where Poisson matters most: very low counts per pixel |

All three are registered in `NoiseModelHook.known`
(`phaser/plan.py:115-117`; implemented in `phaser/engines/common/noise_models.py`).

### `amplitude` and `anscombe`

**`amplitude`** (`AmplitudeNoiseModel.calc_loss`): a fixed-variance squared difference of
offset square roots,
`2 * sum(mask * (sqrt(patterns + offset) - sqrt(model_intensity + offset) - eps)^2) / (1 + gaussian_variance)`.
Comparing square roots rather than raw intensities is a standard variance-stabilizing
transform for shot noise (a Poisson variable's square root has approximately constant
variance regardless of its mean), and dividing by `1 + gaussian_variance` scales the loss
for a chosen fixed noise level.

**`anscombe`** (`AnscombeNoiseModel`): the exact same implementation as `amplitude`, with
only its `offset` default changed to `0.375` (`3/8`) — the standard Anscombe-transform
offset, a refinement of the plain square-root transform that is more accurate at low
counts.

### `poisson`

**`poisson`** (`PoissonNoiseModel.calc_loss`): the Poisson negative log-likelihood between
measured counts and simulated intensity,
`sum(mask * (model_intensity + eps + patterns * (log(patterns + eps) - log(model_intensity + eps) - 1)))`
— this is the standard Poisson NLL (`λ - k·log(λ) + log(k!)`) with `log(k!)` replaced by
its Stirling approximation, `eps` avoiding both logarithms' singularity at zero.

## `eps` and `offset`

| Property | Type | Default | Read by | Units |
| --- | --- | --- | --- | --- |
| `eps` | `float` | `0.001` (all three) | all three | dimensionless, same scale as a count |
| `offset` | `float` | `0.0` (`amplitude`); `0.375` (`anscombe`) | `amplitude`, `anscombe` only | dimensionless, same scale as a count |
| `gaussian_variance` | `float` | `0.1` | `amplitude`, `anscombe` only | dimensionless |

- **Lifecycle stage:** read once, when the noise model is constructed at the start of the
  engine (`props.noise_model(None)`); constant for the whole engine run — none of these
  fields are schedule-valued.
- **Valid range:** no enforced bound; all three default to small positive values that avoid
  division by, or the logarithm of, zero. Practical tuning ranges beyond the defaults shown
  in `examples/` below are guidance pending — no test or benchmark exercises a sweep of
  these values.
- **Engines/backends:** `eps`/`offset`/`gaussian_variance` are read identically regardless
  of engine or backend; the restriction that matters is which noise model you choose (see
  above), not these fields.
- **Interactions:** `offset` and `eps` both stabilize the same square-root transform for
  `amplitude`/`anscombe` — `offset` shifts the whole intensity scale before the square root
  and `eps` regularizes the division in the wave-update ratio; changing one does not require
  changing the other, but both matter most at low counts.

`PoissonNoisePlan` inherits `gaussian_variance` and `offset` from the same base class as
`AmplitudeNoisePlan` for schema reasons, but `PoissonNoiseModel.__init__` only reads
`props.eps` — setting `gaussian_variance` or `offset` under `noise_model: {type: poisson}`
has no effect and is not rejected at validation time.

See the [generated noise-model reference](../../generated/hooks/noise-model.md) for the
exact property tables (types, requiredness, defaults) this section summarizes.

## Minimal example

```yaml
engines:
  - type: gradient
    noise_model:
      type: poisson
      eps: 2.0
    # ... solvers, regularizers, etc.
```

evidenced by `examples/si_grad.yaml` and `examples/mos2_grad.yaml`, both of which pair
`poisson` with the gradient engine and an `eps` around `0.1`–`2.0`.

```yaml
engines:
  - type: conventional
    noise_model:
      type: anscombe
      eps: 1.0e-4
    solver:
      type: lsqml
```

evidenced by `examples/mos2_lsqml.yaml`, which pairs `anscombe` with a `conventional`
(LSQML) engine.

## Failure mode: Poisson under a conventional engine

Configuring `noise_model: {type: poisson}` under a `conventional` engine validates
successfully but raises `NotImplementedError` the first time the solver calls
`calc_wave_update`, at the start of the first group of the first iteration. If you see this,
switch to `amplitude` or `anscombe` for that engine, or move the `poisson` noise model to a
`gradient` engine instead.

## Maintainer sources

- `phaser/engines/common/noise_models.py`
- `phaser/plan.py`
- `phaser/execute.py`
- `docs/architecture/hooks/noise-models.md`
- `docs/architecture/state-and-conventions.md`
- `docs/generated/hooks/noise-model.md`
- `docs/generated/hooks/post-load.md`
- `examples/si_grad.yaml`
- `examples/mos2_grad.yaml`
- `examples/mos2_lsqml.yaml`
