# LSQML reconstruction (experimental PrScO₃)

| Metadata | Value |
| --- | --- |
| Reader level | Beginner–intermediate |
| Data origin | Experimental (EMPAD) |
| Data loader | [`empad`](../parameters/data-and-calibration.md) |
| Model | Multislice — 21 slices over 210 Å |
| Engine | [LSQML](../../concepts/glossary.md#engine) (conventional) |
| Compute requirements | `jax` backend; GPU recommended. No optional dependencies beyond a backend. |
| Updated variables | [Object](../../concepts/glossary.md#object) and [probe](../../concepts/glossary.md#probe) (probe from iteration 5); scan positions (from iteration 5). Tilt is not refined. |
| Features | 8 [mixed-state probe modes](../recipes/mixed-state-probe-modes.md); [expression schedule](../parameters/schedules-and-flags.md) on the LSQML step sizes; four [iteration constraints](../../concepts/glossary.md#constraint); `momentum` position refinement. |
| Runtime class | Workstation/GPU run (not a smoke test) |
| Verification | Plan ships as `examples/prsco3_lsqml.yaml`. Not re-run for this page — validate and run on the downloaded `sample_data/`. |
| Expected output | Per the plan's save settings: HDF5 states and TIFF images every 10 iterations, plus a `finished` marker (see [Expected result](#expected-result)). |

## Goal

Reconstruct a real experimental 4D-STEM dataset (PrScO₃, EMPAD detector) with the
**LSQML** conventional solver, as a multislice, mixed-state reconstruction with scan-position
refinement. This is the general-purpose starting point recommended in [Choosing a
reconstruction engine](../engine-selection.md): LSQML runs on any backend and converges
quickly.

## When to use it

Use this as the template when you have your own experimental detector data and want a
robust first reconstruction. It assumes:

- a thick specimen needing [multislice](../../concepts/glossary.md#engine) propagation
  (here 21 slices) — for a thin, single-slice specimen, drop the `slices` block;
- partial coherence modeled with several [probe modes](../../concepts/glossary.md#mode)
  (here 8);
- scan positions worth refining (experimental scans usually are).

It is not a ptychography-theory tutorial; see [Concepts](../../concepts/ptychography.md)
and the [parameter reference](../parameters/index.md).

## Compatibility

- **Engine:** LSQML (`phaser.engines.conventional.run:run_engine` with the `lsqml` solver).
- **Backend:** any — the conventional engines are not backend-restricted. This plan uses
  `jax`; NumPy, CuPy, and Torch also run it.
- **Noise model:** `amplitude`. **Do not switch a conventional engine to `poisson`** — it
  raises `NotImplementedError` at the first group (verified blocker B6); see the
  [noise-model hooks](../../architecture/hooks/noise-models.md) page.
- **Refinable variables:** object, probe, and positions. Conventional engines do not refine
  tilt (`update_tilt` is inert there).
- **Position refinement caveat:** this plan uses the `momentum` position solver, which
  correctly honors `max_step_size`. The conventional position-correction path is under
  code-owner review (blockers B7/B8); the `momentum` solver is the path unaffected by them.
  See [Solver hooks](../../architecture/hooks/solvers.md) and
  [engine selection](../engine-selection.md).

## Input contract

- **Loader:** [`empad`](../../architecture/hooks/raw-data-loaders.md), reading
  `sample_data/experimental_prsco3/PSO.json` (an EMPAD acquisition; download the
  `sample_data/` archive first — see [Your first
  reconstruction](../../get-started/first-reconstruction.md#get-the-sample-data)).
- **Diffraction origin:** corner-origin (zero-frequency in the array corner), the
  convention `phaser/state.py` documents; the `empad` loader produces this. `diffraction_align`
  (a `post_init` step) recenters the measured diffraction before reconstruction.
- **Calibration:** wavelength, detector angular step, and dose come from the EMPAD metadata
  the loader reads; see [Data and calibration](../parameters/data-and-calibration.md).
- **`raw_data.path` is resolved relative to the current working directory** at run time —
  run from the repository root, or use an absolute path.

## Complete plan

`examples/prsco3_lsqml.yaml`:

```yaml
--8<-- "examples/prsco3_lsqml.yaml"
```

Validate, then run (both from the repository root):

```console
$ phaser validate examples/prsco3_lsqml.yaml
Validation of plan successful!
$ phaser run examples/prsco3_lsqml.yaml
```

!!! danger "Trust warning"
    The `beta_object`/`beta_probe` expression schedules evaluate arbitrary Python via
    `eval` (`phaser/hooks/schedule.py`). A plan that uses one is equivalent to a script —
    only run plans you trust. See [Schedules and flags](../parameters/schedules-and-flags.md).

## Execution flow

1. **Raw-data loading** (`empad`): reads the acquisition, derives wavelength and detector
   sampling from metadata, and produces corner-origin patterns.
2. **Initialization:** builds the object, probe, and scan. With no explicit `init.*` block,
   these come from loader metadata where available, otherwise schema defaults, merged as
   described in [Initialization](../parameters/initialization.md). The object is created as
   21 slices spanning 210 Å (≈ 10 Å per slice).
3. **`post_init`:** `drop_nans` removes patterns that are mostly NaN; `diffraction_align`
   recenters the diffraction origin. See [Post-init hooks](../../architecture/hooks/post-init.md).
4. **LSQML engine** (100 iterations, groups of 16 positions):
   - each group updates the object and (from iteration 5) probe via the LSQML least-squares
     maximum-likelihood update, summing over the 8 probe modes before comparing to the
     measured intensity;
   - the `beta_object`/`beta_probe` step sizes follow `1 - exp(-i/3)`, ramping from ~0 at
     iteration 0 to ~0.95 by iteration 9 — a gentle start that avoids early instability;
   - from iteration 5, the `momentum` position solver nudges scan positions (capped at
     `max_step_size: 0.2`);
   - after every iteration, the four `iter_constraints` run in order (see [Parameter
     walkthrough](#parameter-walkthrough));
   - every 10 iterations, the state and images are saved.
5. **Finish:** a `finished` marker is written to the output directory.

## Parameter walkthrough

Only the options this plan sets are covered; everything else is the `phaser/plan.py`
default. Types and defaults come from the [generated plan
reference](../../generated/plan/index.md); this page adds meaning and units.

**Geometry and model:**

- `slices: {n: 21, total_thickness: 210}` — multislice object, 21 slices over 210 Å
  (thickness in Å). See [Simulation geometry](../parameters/simulation-geometry.md).
- `sim_shape: [128, 128]` — the reconstruction's simulation array size (pixels).
- `probe_modes: 8` — eight incoherent [probe modes](../recipes/mixed-state-probe-modes.md)
  to model partial coherence.
- `bwlim_frac: 1.0` — no extra band-width limiting of the simulation.

**LSQML solver** (`solver.type: lsqml`; see [Solvers and learning
rates](../parameters/solvers-and-learning-rates.md) and [Solver
hooks](../../architecture/hooks/solvers.md)):

- `beta_object`, `beta_probe` — object/probe step sizes, here the schedule `1 - np.exp(-i/3)`
  (`i` is the iteration index). Ramping up from ~0 lets the reconstruction settle before
  taking full steps.
- `gamma: 1.0e-4` — the LSQML step-size regularization (small).
- `illum_reg_object: 50.0`, `illum_reg_probe: 0.1` — illumination regularization
  stabilizing the object and probe updates where illumination is weak; the strong object
  value (50) damps object noise in poorly-illuminated regions.

**Position refinement** (`position_solver.type: momentum`):

- `momentum: 0.90`, `step_size: 1.0e-3`, `max_step_size: 0.2` — a momentum step on scan
  positions with a per-step cap. Active from iteration 5 (`update_positions: {after: 5}`).

**Noise model:** `amplitude` with `eps: 1.0` — the standard amplitude (Gaussian-in-amplitude)
detector model.

**Iteration constraints** (applied after each iteration, in order):

- `limit_probe_support: {max_angle: 22.0}` (mrad) — low-passes the probe to its aperture
  support, keeping high-angle noise out of the probe.
- `clamp_object_amplitude: {amplitude: 1.0}` — clamps object amplitude to ≤ 1 (a
  near-pure-phase object), removing a spurious amplitude degree of freedom.
- `layers: {sigma: 100.0, weight: 0.8}` — regularizes the object across slices, coupling
  adjacent layers to suppress unphysical layer-to-layer variation.
- `obj_gaussian: {sigma: 0.3, weight: 1.0e-2}` — a light real-space Gaussian smoothing of
  the object.

**Schedules and saving:**

- `update_probe: {after: 5}`, `update_positions: {after: 5}` — hold probe and positions
  fixed for the first 5 iterations while the object takes shape.
- `save: {every: 10}`, `save_images: {every: 10}`, `save_options.images` — see [Expected
  result](#expected-result).

## Expected result

Not re-run for this page — the values below describe what the plan's save configuration
produces, not measured numbers.

During the run, each iteration logs a detector error that should trend downward and level
off. On completion, the output directory contains:

```text
iter010.h5  iter020.h5  ...  iter100.h5     # full ReconsState every 10 iterations
probe_iter010.tiff            ... iter100
probe_recip_iter010.tiff      ... iter100
object_phase_sum_iter010.tiff ... iter100
object_mag_sum_iter010.tiff   ... iter100
object_phase_stack_iter010.tiff ... iter100
object_mag_stack_iter010.tiff   ... iter100
finished
```

Each `iterNNN.h5` is a full `ReconsState` (read with `phaser.state.ReconsState.read_hdf5`).
**Success check:** the detector error decreases and plateaus, and the summed object-phase
image (`object_phase_sum`) resolves the atomic-column structure of the specimen. The
per-slice `object_phase_stack` shows how structure is distributed through the 21 slices —
useful for judging whether the multislice depth is right.

## Variations

- **Thin / single-slice specimen:** remove the `slices` block for a 2D projected object.
- **Different specimen (Si) with the gradient engine:** see [Experimental reconstruction
  (Si, gradient descent)](empad-experimental.md).
- **Hand off to gradient descent** for a regularized or Poisson-noise refinement after this
  LSQML stage — add a second engine to the `engines` list (see [engine
  selection](../engine-selection.md)).
- **Fewer probe modes / faster trial:** lower `probe_modes` and `niter` to iterate on
  parameters quickly before a full run.

## Failure modes

- **`Couldn't find raw data at path ...`.** `raw_data.path` is resolved from the current
  working directory, not the plan's location. Run from the repository root, or make sure
  `sample_data/` was downloaded and unpacked there.
- **`NotImplementedError` after switching to `noise_model: poisson`.** Poisson is not
  supported on conventional engines (verified blocker B6) — keep `amplitude` (or
  `anscombe`), or move to the gradient engine.
- **Loss does not decrease / reconstruction diverges.** Lower the `beta_*` step sizes (or
  start the schedule from a smaller value), increase `illum_reg_*`, or check the detector
  calibration and centering — see [Troubleshooting](../troubleshooting.md).
- **Positions drift too far.** Lower `position_solver.step_size` or `max_step_size`, or
  delay `update_positions`. Position refinement on conventional engines is under review
  (B7/B8); the `momentum` solver used here is the unaffected path.

## Maintainer sources

- `examples/prsco3_lsqml.yaml`
- `phaser/plan.py`
- `phaser/hooks/io/empad.py`
- `phaser/hooks/preprocessing.py`
- `phaser/engines/conventional/run.py`, `phaser/engines/conventional/solvers.py`
- `phaser/engines/common/position_correction.py`
- `phaser/engines/common/regularizers.py`
- `phaser/hooks/schedule.py`
- `phaser/state.py`
