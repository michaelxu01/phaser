# Simulated single-slice reconstruction (gradient descent)

| Metadata | Value |
| --- | --- |
| Reader level | Beginner |
| Data origin | Simulated |
| Data loader | [manual](../../architecture/hooks/raw-data-loaders.md) (`.npy`) |
| Model | Single-slice |
| Engine | Gradient |
| Compute requirements | `jax` backend (CPU is enough; `torch` also works — see [Compatibility](#compatibility)). No optional dependencies beyond Phaser's core install. |
| Updated variables | [Object](../../concepts/glossary.md#object) (every iteration), [probe](../../concepts/glossary.md#probe) (from iteration 6). Positions and tilt are not updated. |
| Features | [Iteration constraints](../../concepts/glossary.md#constraint) (`clamp_object_amplitude`, `remove_phase_ramp`, `limit_probe_support`); a `post_load` Poisson-noise injection step. No probe modes, schedules, cost regularizers, or restart. |
| Runtime class | Small smoke test (256 scan positions, 64x64 detector, 200 iterations) |
| Verification | Executable smoke test: `phaser validate` passed, `phaser run` executed to completion, and the result was checked against the known synthetic object (see [Expected result](#expected-result)) |
| Expected output | `examples/smoke/output/`: four HDF5 state files, twelve TIFF images, a `finished` marker; detector error falling from 4.4e+02 to 2.5e+02 over 200 iterations; reconstructed object phase visually and quantitatively recovers the synthetic atom lattice |

## Goal

This page reconstructs a **synthesized** 4D-STEM dataset — not a real specimen — with
Phaser's gradient-descent [engine](../../concepts/glossary.md#engine), giving you a
complete, fast, self-contained example before touching real data. It also doubles as the
**pathfinder for Phaser's shared data-synthesis infrastructure**: no example dataset is
checked into the repository, so this page's companion script
(`examples/smoke/make_test_data.py`) builds one on demand from Phaser's own optics and
scan utilities, and every other simulated-data reconstruction page in this cookbook
reuses it.

## When to use it

Use this page to learn the shape of a minimal gradient-descent plan, to confirm your
Phaser install and backend work end to end, or as a starting point for your own simulated
smoke tests. It assumes:

- a single-slice (2D projected) [object](../../concepts/glossary.md#object) — no
  multislice propagation (see the [multislice sibling page](simulated-multislice-gradient.md)
  for that);
- a single coherent [probe mode](../../concepts/glossary.md#mode) (`probe_modes: 1`, the
  schema default) — good enough for a clean simulated probe with no partial coherence;
- fixed scan positions and no tilt — position and tilt refinement are separate recipes
  ([position refinement](../../cookbook/recipes/position-refinement.md),
  [tilt refinement](../../cookbook/recipes/tilt-refinement.md)).

It is not a tutorial on ptychography theory or on tuning a real reconstruction for
resolution — see [Variations](#variations) and the
[parameter reference](../parameters/index.md) for that.

## Compatibility

- **Engine:** gradient descent (`phaser.engines.gradient.run:run_engine`,
  `phaser/plan.py`).
- **Backend:** the gradient engine requires JAX or Torch — NumPy and CuPy raise
  `ValueError` at engine start (`phaser/execute.py:386-387`, verified checklist blocker
  B1). This page uses JAX; Torch was unavailable in the verification environment, so its
  behavior here is claimed from the code path only, not execution.
- **Noise model:** `amplitude` (`phaser/engines/common/noise_models.py`). The `poisson`
  noise model also works with the gradient engine (it implements `calc_loss`) but
  **only** there — it raises `NotImplementedError` under the conventional engines (ePIE,
  LSQML); see the [noise-model hook page](../../architecture/hooks/noise-models.md) and
  [Variations](#variations) below. This is distinct from the `post_load: poisson` step
  used in [Complete plan](#complete-plan), which injects Poisson shot noise into the
  synthesized patterns and is unrelated to the noise-model hook.
- **Refinable variables:** object and probe. Positions and tilt are left fixed
  (`update_positions: false`, `update_tilt` left at its default `false`).
- **Regularizers:** none (`regularizers: []`). Three **iteration constraints** are used
  instead — see [Parameter walkthrough](#parameter-walkthrough).

## Input contract

The reconstruction reads a raw-binary-free `.npy` array through the
[`manual` raw-data loader](../../architecture/hooks/raw-data-loaders.md)
(`phaser/hooks/io/manual.py`), produced by `examples/smoke/make_test_data.py`:

- **Patterns array:** `patterns.npy`, shape `(scan_ny, scan_nx, det_ny, det_nx)` =
  `(16, 16, 64, 64)`, `float32`, non-negative.
- **Diffraction origin:** corner-origin (zero-frequency sample at `[0, 0]`), matching the
  convention `phaser/state.py` documents for `Patterns.patterns`. The plan sets
  `fftshifted: true` so the `manual` loader does **not** re-shift the data (it would
  otherwise apply an `ifftshift`, which is only correct for detector-centered input).
- **Wavelength:** given as `kv: 300.0` (accelerating voltage, kV); the loader derives
  wavelength from it (`phaser.utils.physics.Electron`). 300 kV → wavelength ≈
  0.019687 Å.
- **Angular/real-space sampling:** `diff_step: 1.538085515954122` mrad (the detector's
  angular step). Together with the wavelength and the 64-pixel detector width, this
  fixes the real-space pixel size at exactly 0.20 Å/px — the same pixel size the
  synthesis script used to build the probe, object, and scan, so the two agree exactly.
  This number must be copied from `examples/smoke/data/manifest.json` if you regenerate
  the dataset with different arguments.
- **Dose/ADU:** `adu: 1.0`. The synthesized patterns are already in electron-count
  units (not raw detector ADU), so this disables the loader's ADU division
  (dividing by 1.0 is a no-op) and avoids its "ADU not supplied" warning.
- **Scan and probe metadata:** the `manual` loader always returns `scan_hook`,
  `tilt_hook`, and `probe_hook` as `None` (`phaser/hooks/io/manual.py:112-114`) — unlike,
  for example, the `empad` loader, which can carry metadata from an acquisition file.
  This plan therefore supplies `init.scan` (`raster`, shape `(16, 16)`, step `1.0` Å) and
  `init.probe` (`focused`, `conv_angle: 20.0` mrad, `defocus: 100.0` Å) explicitly; both
  must match the values `make_test_data.py` used to build the ground truth, or the
  simulated probe/scan used for reconstruction will not match the specimen the patterns
  were actually generated from.
- **Object initialization:** left at the schema default (`init.object` unset →
  `ObjectHook('random')`, `phaser/plan.py`), a near-uniform random-phase guess — the
  reconstruction starts from no prior knowledge of the specimen.

## Complete plan

First, synthesize the dataset (from the repository root, so the plan's relative paths
resolve):

```console
$ python examples/smoke/make_test_data.py
Wrote (16, 16, 64, 64) patterns to examples/smoke/data/patterns.npy
Mean pattern sum (1 electron-equivalent expected): 1.0000
Manifest: { ... }
```

This writes `examples/smoke/data/patterns.npy` (read by the plan below),
`examples/smoke/data/ground_truth.npz` (the synthetic probe/object/scan, for comparison
only — never read by Phaser), and `examples/smoke/data/manifest.json` (every physical
constant used, for cross-checking the plan). None are committed to the repository
(`examples/smoke/.gitignore`) — regenerate them before running the plan.
`examples/smoke/make_test_data.py`'s module docstring documents exactly how the sibling
multislice/EMPAD/ePIE/LSQML reconstruction pages reuse this same script.

The plan itself, `examples/smoke/single_slice_gradient.yaml`:

```yaml
--8<-- "examples/smoke/single_slice_gradient.yaml"
```

Validate it:

```console
$ phaser validate examples/smoke/single_slice_gradient.yaml
Validation of plan successful!
```

Run it:

```console
$ phaser run examples/smoke/single_slice_gradient.yaml
```

(Both commands run from the repository root. See [Expected result](#expected-result) for
the transcript this produced.)

## Execution flow

1. **Raw-data loading** (`manual`, `phaser/hooks/io/manual.py`): reads
   `patterns.npy`, derives wavelength from `kv`, builds a `Sampling` from `diff_step`,
   and (because `fftshifted: true`) leaves the corner-origin patterns untouched.
2. **`post_load`**: `poisson` (`phaser/hooks/preprocessing.py:60`) scales the
   noiseless synthetic patterns by `scale: 5.0e+4` (electrons/pattern) and draws Poisson
   shot noise — the same step `examples/si_grad.yaml` uses for its own simulated data.
3. **Initialization** (`phaser/execute.py:initialize_reconstruction`): builds the probe
   (`focused`), the scan (`raster`), and the object (default `random`, sized from the
   scan extent plus `obj_pad_px`, schema default `5.0`) — none of these come from raw-data
   metadata, since the `manual` loader never supplies any.
4. **`post_init`**: none configured.
5. **Gradient engine** (`phaser.engines.gradient.run:run_engine`):
   - a **dry run** over every group computes a per-group rescale factor
     (measured/simulated intensity) and rescales the initial probe's intensity to match
     the data's actual dose (`phaser/engines/gradient/run.py:209-226`) — this always
     happens, independent of any plan option, and is why the log reports "Rescaling
     initial probe intensity by 5.00e+04," matching the `post_load` dose scale almost
     exactly;
   - **200 iterations**, each split into 4 groups of 64 positions (`grouping: 64` over
     256 total positions); each group computes a loss via the `amplitude` noise model,
     back-propagates gradients through the forward model, and applies an Adam update to
     the object every group and to the probe every group starting at iteration 6
     (`update_probe: {after: 5}`);
   - after every iteration, the three `iter_constraints` run in order: clamp the object's
     amplitude, remove the object's linear phase ramp, and low-pass the probe to its
     aperture's support;
   - every 50th iteration (`save`/`save_images`), the current state and three images are
     written to `examples/smoke/output/`.
6. **Finish**: a `finished` marker file is touched in the output directory.

## Parameter walkthrough

Only the options this plan sets explicitly are covered — everything else is the
`phaser/plan.py` schema default.

**Required, not physically tunable:**

- `raw_data.type: manual`, `path`, `kv`, `diff_step` — described in
  [Input contract](#input-contract).
- `engines[0].noise_model.type: amplitude` — every engine plan requires exactly one
  [noise-model](../../concepts/glossary.md#noise-model) hook (no default).
- `engines[0].solvers.object`/`probe` — every variable a gradient engine updates needs an
  explicit solver (no default); both use `adam` here.

**Commonly adjusted:**

- `raw_data.adu: 1.0` and `fftshifted: true` — see [Input contract](#input-contract);
  both default to `None`/`false` and must be set for pre-normalized, corner-origin
  synthetic data like this.
- `post_load: [{type: poisson, scale: 5.0e+4}]` — the dose (electrons/pattern) applied to
  the noiseless synthetic patterns before adding shot noise. Higher values give a less
  noisy, easier reconstruction; this value was chosen to be visibly noisy without needing
  more iterations to average out.
- `engines[0].niter: 200` (schema default `10`) — chosen empirically: the detector error
  visibly plateaus by iteration ~150–200 at this dose and scan size (see
  [Expected result](#expected-result)).
- `engines[0].grouping: 64` (schema default `None`, which resolves to `64` internally
  anyway, `phaser/engines/common/simulation.py:31`) — set explicitly for clarity, not
  because it changes behavior from the default.
- `engines[0].solvers.*.learning_rate` (`2.0e-3` object, `5.0e-3` probe) — chosen small
  enough to avoid divergence at this noise model and offset (see the next bullet and
  [Failure modes](#failure-modes)); no default exists for this required field.
- `engines[0].noise_model.offset: 1.0` (schema default `0.0`) — **the single most
  important non-obvious setting on this page.** See
  [Failure modes](#failure-modes): without a nonzero `offset`, this exact plan produces
  `NaN`/`inf` at iteration 1.
- `engines[0].update_probe: {after: 5}` (schema default `true`, i.e. every iteration from
  the start) — delays probe updates by 5 iterations so the (initially random) object
  moves first; the probe was already rescaled to the correct intensity by the dry run, so
  it doesn't need to move immediately.
- `engines[0].update_positions: false` — identical to the schema default; set explicitly
  to make clear that position refinement is deliberately out of scope for this page (see
  [position refinement](../../cookbook/recipes/position-refinement.md)).
- `engines[0].iter_constraints` (schema default `[]`, i.e. none) — three constraints,
  explained together since they solve one problem (see
  [Failure modes](#failure-modes) for what happens without them):
  - `clamp_object_amplitude: {amplitude: 1.05}` (`ClampObjectAmplitudeProps.amplitude`
    default `1.1`) — the synthesized object is pure-phase (unit amplitude by
    construction), so clamping the reconstructed amplitude close to 1 removes a degree of
    freedom gradient descent would otherwise waste on amplitude drift instead of phase
    accuracy.
  - `remove_phase_ramp` (no properties) — removes an arbitrary linear phase gradient
    the object is free to acquire (a well-known ptychographic gauge freedom); without it,
    the reconstructed phase can wrap to ±π with no corresponding feature in the specimen.
  - `limit_probe_support: {max_angle: 40.0}` (mrad; required, no default) — low-pass
    filters the probe to twice the illumination aperture (`conv_angle: 20.0`) every
    iteration, preventing the probe from absorbing high-angle noise.
- `engines[0].save`/`save_images: {every: 50}` and `save_options.out_dir`/`images` —
  see [Expected result](#expected-result) for what these produce.

**Left at the default, worth knowing about:** `probe_modes: 1` (a single coherent probe
mode — see [mixed-state probe modes](../../cookbook/recipes/mixed-state-probe-modes.md)
to add more) and `buffer_n_groups` (schema default `2`, i.e. two groups of patterns are
prefetched onto the device at a time — irrelevant at this dataset size, but see
[grouping and memory](../parameters/grouping-and-memory.md) for its tri-state meaning).

## Expected result

Validation:

```console
$ phaser validate examples/smoke/single_slice_gradient.yaml
Validation of plan successful!
```

Execution (`phaser run examples/smoke/single_slice_gradient.yaml`, CPU, `jax` backend,
this page's verification run):

```text
INFO:phaser.hooks.preprocessing:Mean pattern intensity: 50015.2265625
INFO:root:Initialized reconstruction in 00:01.091
INFO:phaser.engines.gradient.run:Rescaling initial probe intensity by 5.00e+04
INFO:root:Finished iter   1/200 [00:02.060] Error: 4.425e+02
INFO:root:Finished iter   2/200 [00:00.023] Error: 4.195e+02
...
INFO:root:Finished iter 199/200 [00:00.028] Error: 2.466e+02
INFO:root:Finished iter 200/200 [00:00.035] Error: 2.466e+02
INFO:root:Engine finished!
INFO:root:Total engine time: 00:00:08.737
INFO:root:Total reconstruction time: 00:00:08.745
INFO:root:Reconstruction finished!
```

Total wall time for the whole command (including JIT compilation, `time phaser run ...`):
**≈10.5 s** on a CPU-only `jax` backend — comfortably inside a "small smoke test" budget.

**Output tree** (`examples/smoke/output/`, not committed — see
`examples/smoke/.gitignore`):

```text
examples/smoke/output/
  finished
  iter050.h5   iter100.h5   iter150.h5   iter200.h5
  probe_iter050.tiff              ... iter100/150/200
  object_phase_sum_iter050.tiff   ... iter100/150/200
  object_mag_sum_iter050.tiff     ... iter100/150/200
```

Each `iterNNN.h5` is a full `ReconsState` (readable with
`phaser.state.ReconsState.read_hdf5`); each `.tiff` is a 16-bit scaled image
(`save_options.img_dtype`, default `'16bit'`).

**Basic success check.** The detector error falls monotonically from `4.425e+02` at
iteration 1 to `2.466e+02` at iteration 200, then visibly flattens over the last ~50
iterations — consistent with the reconstruction reaching the statistical noise floor set
by the Poisson dose (`scale: 5.0e+4`), not a stalled or diverged optimization. To confirm
the reconstruction actually recovered the **known synthetic specimen** (not just that the
loss went down), this page's verification compared the final object phase
(`examples/smoke/output/iter200.h5`) against `examples/smoke/data/ground_truth.npz`:

- The reconstructed object phase, viewed as an image, visibly reproduces the synthesized
  5x5 grid of Gaussian "atom" phase bumps at the correct spacing and arrangement, inside
  the scanned region; outside the scanned region (where no probe ever illuminated the
  object) the phase is empty/noisy, as expected.
- A normalized cross-correlation between the reconstructed and ground-truth phase, cropped
  to a 100x100-pixel window centered on the atom lattice, is **0.30 at zero shift** and
  **0.75 at its best alignment**, found at a sub-pixel offset of only 2 object pixels
  (0.4 Å) in each direction — that small, sub-resolution offset is expected: the
  synthesis script and Phaser's own reconstruction independently compute the object's
  padding and center pixel, so their coordinate origins are not pixel-identical even
  though both use the same 0.20 Å pixel size.

This is example-demonstrated evidence from one run, not an asserted correctness test (see
checklist blocker B12) — no automated test in `tests/` exercises the gradient engine.

## Variations

- **Multislice:** run `examples/smoke/make_test_data.py --multislice` (writes to a
  separate output directory to avoid overwriting this page's dataset) and add a
  top-level `slices: {n: 2, total_thickness: 80.0}` to the plan — see the
  [multislice sibling page](simulated-multislice-gradient.md).
- **More probe modes:** set `probe_modes` above `1` to model partial coherence — see
  [mixed-state probe modes](../../cookbook/recipes/mixed-state-probe-modes.md).
- **Poisson noise model instead of amplitude:** set `noise_model: {type: poisson, offset:
  <nonzero>}` — it works with the gradient engine (unlike the conventional engines; see
  [Compatibility](#compatibility)) and needs the same nonzero `offset` protection
  discussed in [Failure modes](#failure-modes).
- **Higher or lower dose:** change `post_load[0].scale`; a lower dose needs a larger
  `noise_model.offset` (relative to typical per-pixel counts) to stay numerically stable,
  and will plateau at a higher detector error (a higher noise floor).
- **More scan positions / larger detector:** increase `--scan-shape`/`--det-shape` on
  `make_test_data.py` and the matching `init.scan.shape`/detector-dependent fields in the
  plan; runtime grows roughly linearly with the number of positions.

## Failure modes

- **`ValueError: NaN or inf encountered, iteration 1`, with the `amplitude` (or
  `poisson`) noise model.** Cause (verified by direct reproduction): the noise model's
  `calc_loss` differentiates `sqrt(model_intensity + offset)`
  (`phaser/engines/common/noise_models.py:36-41`); with `offset: 0.0` (the schema
  default) and a simulated probe whose far-field disk covers only a small fraction of
  the detector, a nontrivial fraction of detector pixels have `model_intensity` that
  underflows to exactly `0.0` in `float32` — and `d/dx sqrt(x)` diverges as `x -> 0`,
  producing `inf`/`NaN` gradients from the very first iteration even though the loss
  itself is finite. This reproduced on this exact plan with `offset: 0.0`; setting
  `offset: 1.0` (roughly the scale of a single detector count) fixed it immediately.
  This is a general risk whenever a simulated dataset has large true-zero regions on the
  detector (small convergence angle relative to the detector's angular range) — less
  likely with real experimental data, which almost always has some detector background.
- **Detector error plateaus early and the reconstructed phase wraps to ±π with no
  resemblance to the specimen.** Cause (reproduced): running this same plan with
  `iter_constraints: []` lets the object accumulate an unconstrained linear phase ramp
  and amplitude drift — gradient descent is free to spend effort there instead of on the
  specimen's actual phase. Fix: the three `iter_constraints` in this plan
  (`clamp_object_amplitude`, `remove_phase_ramp`, `limit_probe_support`); see
  [Parameter walkthrough](#parameter-walkthrough).
  - Confusingly, the raw error *number* by itself does not distinguish "reached the noise
    floor with a good reconstruction" from "stalled with a bad one" — always check the
    reconstructed object against independent knowledge of the specimen (or, with real
    data, plausibility) rather than the loss curve alone.
- **`ValueError: Couldn't find raw data at path ...`.** Cause: `raw_data.path` is
  resolved relative to the current working directory at run time
  (`Path(props.path).expanduser()`, `phaser/hooks/io/manual.py:20`), **not** relative to
  the plan YAML file's location. Fix: run `phaser validate`/`phaser run` from the
  repository root, as shown in [Complete plan](#complete-plan), or use an absolute path.
- **`raw_data.path` exists but the reconstruction looks wrong with no error raised.**
  Cause: `diff_step`/`kv` in the plan no longer match the values
  `examples/smoke/data/manifest.json` records for the actual data on disk — for example,
  after regenerating the dataset with different script arguments without updating the
  plan. Nothing in `phaser validate` or `phaser run` can detect this (the two files are
  independent), so a silent sampling mismatch is possible. Fix: after regenerating data,
  re-check `manifest.json`'s `diff_step_mrad`, `kv`, `conv_angle_mrad`, `defocus_angstrom`,
  and scan fields against the plan.
- **`WARNING: ADU not supplied for experimental dataset.`** Harmless but avoidable: this
  fires whenever `raw_data.adu` is unset, including for synthetic data already in
  electron-count units — set `adu: 1.0` (a no-op scale) to silence it, as this plan does.

## Maintainer sources

- `phaser/hooks/io/manual.py`
- `phaser/hooks/__init__.py`
- `phaser/hooks/probe.py`, `phaser/hooks/scan.py`, `phaser/hooks/object.py`
- `phaser/hooks/preprocessing.py`
- `phaser/plan.py`
- `phaser/execute.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/common/noise_models.py`
- `phaser/engines/common/regularizers.py`
- `phaser/engines/common/simulation.py`
- `phaser/utils/optics.py`, `phaser/utils/scan.py`, `phaser/utils/physics.py`
- `phaser/state.py`
- `examples/smoke/make_test_data.py`, `examples/smoke/single_slice_gradient.yaml` (this
  page's own verified artifacts)
- `examples/si_grad.yaml` (precedent for the `post_load: poisson` pattern)
