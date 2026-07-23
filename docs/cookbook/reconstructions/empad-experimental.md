# Experimental reconstruction (Si, gradient descent)

| Metadata | Value |
| --- | --- |
| Reader level | Intermediate |
| Data origin | Experimental (EMPAD) |
| Data loader | [`empad`](../parameters/data-and-calibration.md) |
| Model | Multislice — 10 slices over 200 Å |
| Engine | [Gradient descent](../../concepts/glossary.md#engine) |
| Compute requirements | `jax` or `torch` backend (gradient engine requirement); GPU recommended. |
| Updated variables | [Object](../../concepts/glossary.md#object) and [probe](../../concepts/glossary.md#probe) (probe from iteration 5); scan positions (from iteration 10). Tilt is not refined. |
| Features | Poisson noise model; 4 [probe modes](../recipes/mixed-state-probe-modes.md); three [cost regularizers](../../concepts/glossary.md#regularizer); two [iteration constraints](../../concepts/glossary.md#constraint); per-variable Adam/SGD solvers; a `crop_data` preprocessing step. |
| Runtime class | Workstation/GPU run (500 iterations) |
| Verification | Plan ships as `examples/si_grad_exp.yaml`. Not re-run for this page — validate and run on the downloaded `sample_data/`. |
| Expected output | Per the plan's save settings: HDF5 states every 10 iterations, images every 2, plus a `finished` marker (see [Expected result](#expected-result)). |

## Goal

Reconstruct a real experimental 4D-STEM dataset (Si, EMPAD detector) with the
**gradient-descent** engine: a multislice, mixed-state reconstruction using the Poisson
noise model, cost regularizers, and scan-position refinement. It is the experimental
counterpart to the [simulated single-slice gradient
example](simulated-single-slice-gradient.md), and the companion to the
[LSQML PrScO₃ page](lsqml.md) on the same real-data path.

## When to use it

Use this template when you want the gradient engine's specific capabilities on your own
data: the **Poisson** noise model (correct for photon-counting detectors at low dose),
differentiable **cost regularizers**, and independent optimizers per variable. It assumes:

- a thick specimen needing [multislice](../../concepts/glossary.md#engine) propagation
  (here 10 slices);
- a photon-counting detector where Poisson statistics apply;
- a JAX or Torch install (the gradient engine requires one).

For a general-purpose first reconstruction with fewer moving parts, start from the
[LSQML page](lsqml.md) instead.

## Compatibility

- **Engine:** gradient descent (`phaser.engines.gradient.run:run_engine`).
- **Backend:** JAX or Torch **only** — NumPy and CuPy raise `ValueError` at engine start
  (`phaser/execute.py`, verified blocker B1). This plan uses `jax`.
- **Noise model:** `poisson`. This works **only** on the gradient engine; the conventional
  engines raise `NotImplementedError` for it (blocker B6). See [noise-model
  hooks](../../architecture/hooks/noise-models.md).
- **Refinable variables:** object, probe, and positions. Position refinement here is a
  gradient step on the `positions` variable (an `sgd` solver) — the gradient-engine path,
  which is not affected by the conventional-engine position blockers (B7/B8).
- **Cost regularizers:** available only on the gradient engine (the conventional engines
  have no `regularizers` field). See [cost-regularizer
  hooks](../../architecture/hooks/cost-regularizers.md).

## Input contract

- **Loader:** [`empad`](../../architecture/hooks/raw-data-loaders.md), reading
  `sample_data/experimental_si/acq12_20over.json` (download the `sample_data/` archive
  first — see [Your first
  reconstruction](../../get-started/first-reconstruction.md#get-the-sample-data)).
- **Diffraction origin:** corner-origin (`phaser/state.py`); `diffraction_align` recenters
  the measured diffraction during `post_init`.
- **Calibration and dose:** wavelength, detector sampling, and count scaling come from the
  EMPAD metadata; the Poisson noise model depends on patterns being in electron-count
  units. See [Data and calibration](../parameters/data-and-calibration.md).
- **`raw_data.path` resolves relative to the current working directory** — run from the
  repository root or use an absolute path.

## Complete plan

`examples/si_grad_exp.yaml`:

```yaml
--8<-- "examples/si_grad_exp.yaml"
```

Validate, then run (from the repository root):

```console
$ phaser validate examples/si_grad_exp.yaml
Validation of plan successful!
$ phaser run examples/si_grad_exp.yaml
```

## Execution flow

1. **Raw-data loading** (`empad`): reads the acquisition and produces corner-origin
   patterns with calibration from metadata.
2. **`post_load`:** `crop_data` with `crop: [50, -50, 50, -50]` trims 50 pixels from each
   edge of the detector, discarding the outer margin before reconstruction.
3. **Initialization:** builds the object (10 slices over 200 Å ≈ 20 Å/slice), probe, and
   scan from loader metadata and defaults (see
   [Initialization](../parameters/initialization.md)).
4. **`post_init`:** `drop_nans` removes mostly-NaN patterns; `diffraction_align` recenters
   the diffraction. See [Post-init hooks](../../architecture/hooks/post-init.md).
5. **Gradient engine** (500 iterations, groups of 128 positions):
   - a dry run rescales the initial probe intensity to match the data's dose (this always
     happens; see [lifecycle](../../architecture/lifecycle.md));
   - each group computes the Poisson loss (summed over the 4 probe modes),
     back-propagates gradients through the multislice forward model, and applies Adam
     updates to the object (every group) and probe (from iteration 5); positions update via
     SGD from iteration 10;
   - the three cost regularizers add differentiable penalties to the loss each group; the
     two iteration constraints run after each iteration;
   - the state saves every 10 iterations and images every 2.
6. **Finish:** a `finished` marker is written.

## Parameter walkthrough

Only the options this plan sets are covered. Types and defaults come from the [generated
plan reference](../../generated/plan/index.md); this page adds meaning and units.

**Preprocessing and geometry:**

- `post_load: [{type: crop_data, crop: [50, -50, 50, -50]}]` — crops 50 px from each
  detector edge (see [Post-load hooks](../../architecture/hooks/post-load.md)).
- `slices: {n: 10, total_thickness: 200}` — 10-slice multislice object over 200 Å.
- `probe_modes: 4`, `bwlim_frac: 1.0`.

**Noise model:** `poisson` with `eps: 2.0` — the Poisson (photon-counting) likelihood, with
`eps` a small stabilizing constant. Gradient-engine only (see [Compatibility](#compatibility)).

**Solvers** (one per updated variable; see [Solvers and learning
rates](../parameters/solvers-and-learning-rates.md)):

- `object: {type: adam, learning_rate: 7.0e-2, nesterov: true}`
- `probe: {type: adam, learning_rate: 0.1, nesterov: true}`
- `positions: {type: sgd, learning_rate: 0.5, momentum: 0.90, nesterov: true}` — the
  gradient-engine position refinement.

**Cost regularizers** (differentiable penalties added to the loss; gradient-engine only):

- `obj_l2: {cost: 0.3}` — L2 penalty on the object.
- `obj_tikh: {cost: 0.6}` — Tikhonov (smoothness) penalty on the object.
- `layers_tikh: {cost: 5.0e+3}` — strong Tikhonov coupling across slices, smoothing
  structure through the object's depth.

**Iteration constraints** (after each iteration):

- `limit_probe_support: {max_angle: 21.0}` (mrad) — low-passes the probe to its aperture.
- `clamp_object_amplitude: {amplitude: 1.0}` — clamps object amplitude to ≤ 1.

**Schedules and saving:**

- `update_probe: {after: 5}`, `update_positions: {after: 10}` — hold probe fixed for 5
  iterations and positions for 10 while the object settles.
- `save: {every: 10}`, `save_images: {every: 2}`, `save_options.images` — see [Expected
  result](#expected-result). Images every 2 iterations produces a lot of files across 500
  iterations; raise the interval if that is too much output.

## Expected result

Not re-run for this page — the values below describe what the plan's save configuration
produces, not measured numbers.

Each iteration logs a Poisson loss that should trend downward and plateau. On completion,
the output directory contains HDF5 states `iter010.h5` … `iter500.h5` (every 10
iterations), TIFF images every 2 iterations for `probe`, `probe_recip`,
`object_phase_sum`, `object_mag_sum`, `object_phase_stack`, and `object_mag_stack`, and a
`finished` marker.

**Success check:** the loss decreases and levels off, and `object_phase_sum` resolves the
silicon atomic columns; the `object_phase_stack` shows structure distributed sensibly
through the 10 slices. Read a saved state with `phaser.state.ReconsState.read_hdf5`.

## Variations

- **LSQML instead of gradient** (any backend, faster coarse pass): see the [LSQML
  page](lsqml.md); hand off from LSQML to this gradient stage by chaining engines.
- **Amplitude noise model:** switch `noise_model` to `amplitude` if your data is not in
  clean count units (Poisson assumes electron-count patterns).
- **Fewer iterations for a trial:** lower `niter` and raise `save_images.every` while
  tuning learning rates and regularizer weights.
- **Single-slice:** remove the `slices` block for a thin specimen.

## Failure modes

- **`ValueError` at engine start under NumPy or CuPy.** The gradient engine requires JAX or
  Torch (B1). Install one and set `backend: jax` (or `torch`).
- **`Couldn't find raw data at path ...`.** Run from the repository root, and confirm
  `sample_data/` was downloaded and unpacked there.
- **`NaN`/`inf` during optimization.** Usually too-large learning rates or count scaling
  issues for the Poisson model — lower the `object`/`probe` learning rates, raise the
  noise-model `eps`, or check ADU/count calibration (see
  [Troubleshooting](../troubleshooting.md) and [Data and
  calibration](../parameters/data-and-calibration.md)).
- **Loss decreases but the object looks wrong (phase ramp, layer artifacts).** Adjust the
  regularizer weights (`obj_tikh`, `layers_tikh`) and the iteration constraints; too little
  `layers_tikh` lets slices vary unphysically, too much oversmooths depth.
- **Positions drift too far.** Lower `positions.learning_rate` or delay `update_positions`.

## Maintainer sources

- `examples/si_grad_exp.yaml`
- `phaser/plan.py`
- `phaser/hooks/io/empad.py`
- `phaser/hooks/preprocessing.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/common/noise_models.py`
- `phaser/engines/common/regularizers.py`
- `phaser/execute.py`
- `phaser/state.py`
