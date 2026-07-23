# Output and restart

This page covers what a reconstruction writes to disk (`save`, `save_images`,
`save_options`) and how to resume a later reconstruction from a saved state
(`init.state`). It builds on
[Serialization](../../architecture/state-and-conventions.md#serialization) for the HDF5
file format and
[Initialization merge semantics](../../architecture/lifecycle.md#raw-data-loading-and-the-initialization-merge)
for how a restart interacts with `init.*` — this page doesn't repeat that merge rule, only
how to use it for restart. All fields below are generated from
`ConventionalEnginePlan`/`GradientEnginePlan` and `SaveOptions` in the
[plan reference](../../generated/plan/index.md); this page adds units, cadence semantics,
and file-naming behavior.

## `save` and `save_images`: what gets written, and when

| Field | Type (generated) | Default | What it triggers |
| --- | --- | --- | --- |
| `save` | `bool \| SimpleFlag \| FlagHook` | `False` | Writing the full `ReconsState` to an HDF5 file (`output_state`, `phaser/engines/common/output.py:35-45`). |
| `save_images` | `bool \| SimpleFlag \| FlagHook` | `False` | Writing the image types listed in `save_options.images` (`output_images`, `phaser/engines/common/output.py:16-32`). |

Both are [flags](../../concepts/glossary.md#flag) — a plain `bool` (always/never), a
`SimpleFlag(after=, every=, before=)`, or a schedule/flag hook (see
[Schedules and flags](schedules-and-flags.md) for the full mechanism, once written). For a
`SimpleFlag`, the check is `state.iter.engine_iter > after`, `(engine_iter - after) % every
== 0`, and `engine_iter < before` if set (`SimpleFlag.__call__`, `phaser/types.py:220-226`)
— `after`/`every`/`before` count the *current engine's* iteration, not the reconstruction's
total. `SaveObserver` (the built-in observer that performs saving) checks the flag once per
iteration (`update_iteration`, `phaser/observer.py:260-270`) **and unconditionally saves
once more at the end of the engine** if the flag was ever true for any iteration of that
engine (`finish_engine`, `phaser/observer.py:272-281`, using `flag_any_true` to decide
whether any output is expected at all) — so the last iteration of an engine is always
captured, even if `every` would not otherwise have matched it.

**Lifecycle stage:** re-evaluated at the start of each engine (`init_engine`,
`phaser/observer.py:222-230`) and then checked every iteration through that engine's run.
**Engines/backends:** identical behavior in both engine families — `SaveObserver` is
engine-agnostic.

## `save_options`: image and file-naming configuration

Type and default are generated on the
[`SaveOptions` reference](../../generated/plan/index.md#saveoptions). Every field below is
verified against `phaser/plan.py` and `phaser/engines/common/output.py`; only these nine
fields exist — do not use names from older prose that don't appear here.

| Field | Type | Default | Units | Meaning |
| --- | --- | --- | --- | --- |
| `images` | `tuple` of `'probe'`, `'probe_mag'`, `'probe_recip'`, `'probe_recip_mag'`, `'object_phase_stack'`, `'object_phase_sum'`, `'object_mag_stack'`, `'object_mag_sum'`, `'scan'`, `'tilt'` | `('probe', 'object_phase_stack')` | — | Which image products `save_images` writes; each name maps to one save function (`_SAVE_FUNCS`, `phaser/engines/common/output.py:240-251`). `probe_recip`/`probe_recip_mag` are the probe's diffraction-space (Fourier) amplitude; `_stack` variants write one image per object slice, `_sum` variants write one image summed/averaged over slices. `scan` and `tilt` are plotted diagrams, not detector-shaped images (see `plot_ext` below). |
| `crop_roi` | `bool` | `True` | — | Whether object images are cropped to the scanned region of interest before saving, rather than the full (padded) object array (`_save_object_phase`/`_save_object_mag`, `phaser/engines/common/output.py:112-149`). |
| `unwrap_phase` | `bool` | `False` | — | Whether object phase images are phase-unwrapped before saving (affects `object_phase_stack`/`object_phase_sum` only). |
| `img_dtype` | `'float' \| '8bit' \| '16bit' \| '32bit'` | `'16bit'` | — | Output pixel encoding for saved images: `'float'` writes the raw float array; the integer options scale to that bit depth (`scale_to_integral_type`). Does not affect the HDF5 state file, which always stores full-precision arrays. |
| `plot_ext` | `str` | `'svg'` | — | File extension for the two **plotted** image types (`scan`, `tilt`; `_PLOT_FUNCS`, `phaser/engines/common/output.py:253`) — every other image type is always written as `'tiff'` regardless of this setting. |
| `plot_dpi` | `int` | `300` | dots per inch | Resolution of the `scan`/`tilt` plots specifically; irrelevant to `tiff`-written images. |
| `out_dir` | `str` (format string) | `'{name}'` | — | Output directory, formatted once per engine with `engine_num`, `name` (the reconstruction name), `group` (that engine's `grouping`), and `niter` (`SaveObserver.init_engine`, `phaser/observer.py:234-239`). An unrecognized `{key}` raises `ValueError` ("Invalid format string in 'out_dir'"). |
| `img_fmt` | `str` (format string) | `'{type}_iter{iter.total_iter:03}.{ext}'` | — | Per-image filename, formatted with `type` (the image name from `images`), `iter` (the current `IterState` — supports attribute access like `iter.total_iter`, `iter.engine_iter`, `iter.engine_num`), and `ext` (`plot_ext` for `scan`/`tilt`, else `'tiff'`). |
| `hdf5_fmt` | `str` (format string) | `'iter{iter.total_iter:03}.h5'` | — | Per-state-file filename, formatted with `iter` (same `IterState` object) only — no `type`/`ext` keys are available here. |

**Lifecycle stage:** `out_dir` is resolved once per engine (so it can depend on that
engine's `grouping`/`niter`/`engine_num`); `img_fmt`/`hdf5_fmt` are resolved once per saved
file. **Interactions:** if the reconstruction has more than one engine and `out_dir`'s
resolved value changes between engines, the previous directory is closed (its `finished`
marker touched) before the new one is opened (`SaveObserver.init_engine`,
`phaser/observer.py:246-248`) — see [HDF5 state contents](#hdf5-state-contents) below for
what `finished` means.

## HDF5 state contents

Each saved state file is a complete `ReconsState`
(`phaser/state.py`), written via `write_hdf5`/`hdf5_write_state`
(`phaser/utils/io.py`) — every field the reconstruction actually tracks, not a partial
snapshot:

| Field | Contents |
| --- | --- |
| `iter` | The `IterState` at the moment of saving: `engine_num`, `engine_iter`, `total_iter`, and (if set) `n_engine_iters`/`n_total_iters`. |
| `wavelength` | The electron wavelength, Å (scalar). |
| `probe` | `ProbeState`: sampling and `data`, shape `(modes, y, x)`. |
| `object` | `ObjectState`: sampling, `data` (shape `(z, y, x)`), and `thicknesses`. |
| `scan` | Scan positions, shape `(..., 2)`, `(y, x)`, length units. |
| `tilt` | Tilt angles if present, shape `(..., 2)`, `(y, x)`, mrad; otherwise absent. |
| `progress` | Named error-measurement series (`ProgressState`: parallel `iters`/`values` lists), if any were recorded. |

The file also carries a `type` marker (`'phaser_state'`) and a `version` marker, checked on
read — see
[Serialization](../../architecture/state-and-conventions.md#serialization). A `'finished'`
marker file (not part of the HDF5 state itself) is touched in `out_dir` once the engine (or
whole reconstruction) completes without an exception
(`SaveObserver.close`/`finish_engine`, `phaser/observer.py:272-286`) and removed at the
start of each new engine using that directory — its presence is a simple, file-based signal
that the run in that directory completed rather than being interrupted mid-way.

## Restarting a reconstruction

To resume from a saved state, point `init.state` at that HDF5 file:

```yaml
init:
  state: "output/iter100.h5"
```

`init.state` is read into a `PartialReconsState` before raw-data loading
(`phaser.execute.initialize_reconstruction`); each of `scan`, `tilt`, `probe`, and `object`
is then independently either **reused from that file** or **rebuilt from the merged
`init.*` hook**, following the rule in
[Initialization merge semantics](../../architecture/lifecycle.md#raw-data-loading-and-the-initialization-merge)
— leave the matching `init.*` field unset to reuse a component from the restart file, or
set it (to `{}` for `scan`/`tilt`/`probe`, or to an explicit hook for any of the four,
including `object`, which has no `{}` form) to rebuild it instead. See
[Initialization: Restart](initialization.md#restart-versus-leaving-a-field-unset) for the
full per-component table and a worked example.

A restart also re-reads `raw_data` from scratch (the loader hook always runs) — the
restart file supplies *state*, not diffraction patterns; you still need a valid `raw_data`
entry in the restarting plan, ordinarily the same one the original run used.

## Minimal example

Saving state and a small set of images every 25 iterations, to a per-reconstruction-name
directory:

```yaml
engines:
  - type: gradient
    niter: 100
    save: {every: 25}
    save_images: {every: 25}
    save_options:
      images: [probe, object_phase_sum, object_mag_sum]
      out_dir: "{name}"
      img_fmt: "{type}_iter{iter.total_iter:03}.{ext}"
      hdf5_fmt: "iter{iter.total_iter:03}.h5"
```

Restarting from that run's final saved state, keeping every component (scan, tilt, probe,
object) exactly as saved:

```yaml
init:
  state: "{name}/iter100.h5"
```

*Verification pending:* both snippets are drawn from field defaults and the observed
behavior of `examples/smoke/single_slice_gradient.yaml` (see
[Simulated single-slice gradient: Expected result](../reconstructions/simulated-single-slice-gradient.md#expected-result)
for a real, executed transcript); the restart snippet was not independently re-run for
this page.

## Maintainer sources

- `phaser/plan.py`
- `phaser/observer.py`
- `phaser/engines/common/output.py`
- `phaser/state.py`
- `phaser/utils/io.py`
- `phaser/types.py`
- `phaser/execute.py`
