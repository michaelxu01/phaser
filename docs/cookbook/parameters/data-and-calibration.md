# Data and calibration

Choose a `raw_data` loader and set the properties that make the loaded patterns physically
correct: detector geometry, accelerating voltage, dose scaling, and diffraction-pattern
orientation. Types and defaults come from the
[Raw Data reference](../../generated/hooks/raw-data.md); this page adds units, meaning, and
how the loaders differ. It assumes the corner-origin and intensity conventions from
[State and conventions](../../architecture/state-and-conventions.md) and the
[initialization merge](../../architecture/lifecycle.md#raw-data-loading-and-the-initialization-merge).

## Choosing a raw-data loader

`raw_data` is **required** and takes exactly one hook, run first. Pick `empad`/`gatan`/`nion`
when your acquisition software wrote a metadata file Phaser understands (so scan geometry is
read for you); pick `manual` for any other format or for full control. Verified against
`phaser/hooks/io/*.py`:

| Loader | Reads | Metadata it can supply | You must supply |
| --- | --- | --- | --- |
| `empad` | raw EMPAD binary, or a `.json` pointing at one | from `.json`: `probe_hook` (`focused`), `scan_hook` (`raster`); no tilt | `kv`/`diff_step` only if the metadata omits them, or if loading a raw binary |
| `gatan` | Gatan `.dm4` (needs optional `rsciio`) | `probe_hook` (`focused`), `scan_hook` (`raster`) from file metadata | `kv`/`diff_step` only if the file omits them |
| `nion` | Nion `.zip` bundle | `scan_hook` (`raster`); no probe, no tilt | `diff_step` always (required); `kv` read from the bundle |
| `manual` | `.npy`/`.npz`, TIFF, HDF5/EMD, or raw binary | none (`phaser/hooks/io/manual.py:112-114`) | `diff_step` always; `kv` or `wavelength` |

Supplied metadata is merged with `init.scan`/`init.tilt`/`init.probe`, not final — see the
[merge rules](../../architecture/lifecycle.md#raw-data-loading-and-the-initialization-merge)
and [Initialization](initialization.md).

!!! warning "Restriction"
    The `gatan` loader requires the optional `rsciio` dependency (`pip install rosettasciio`).

## Per-loader calibration properties

All read once at raw-data loading, before any state exists.

### Shared across loaders

| Property | Loaders | Units | Meaning |
| --- | --- | --- | --- |
| `path` | all | — | File path (`~`-expanded for `empad`/`gatan`/`nion`; must exist or raises `ValueError`). |
| `diff_step` | all | mrad | Detector angular step per pixel. Sets the real-space pixel size: `pixel_size = wavelength / (diff_step * 1e-3)` (`phaser/hooks/io/empad.py:83`). Wrong value silently mis-scales every reconstructed length. Required for `nion`/`manual`; optional (metadata fallback) for `empad`/`gatan`. |
| `kv` | `empad`, `gatan`, `manual` | kV | Accelerating voltage → wavelength via `Electron(voltage).wavelength`. Metadata fallback for `empad`/`gatan`; `manual` needs `kv` **or** `wavelength`, not both (`manual.py:24`). `nion` has none — voltage comes from the bundle. |
| `adu` | `empad`, `gatan` (schema only, see below), `manual` | counts per particle | Divides raw values to physical particle counts. See [Count scaling](#intensity-and-count-scaling). |
| `det_flips` | `empad`, `manual` | — | `(flip_y, flip_x, transpose)` applied on load. Metadata fallback for `empad`. |

### Loader-specific

| Loader | Property | Notes |
| --- | --- | --- |
| `empad` | (`.json` path) | Reads `scan_shape`, `scan_step`, `scan_correction`, rotation, `conv_angle`, `defocus`, and `is_simulated()`. EMPAD v2 (`empad_version > 1`) raises `ValueError` (`empad.py:27`). |
| `nion` | `detector_rotation_offset` | `float \| None`, default `None`; degrees, added to the bundle's scan rotation. |
| `manual` | `wavelength` | Å; alternative to `kv`. |
| `manual` | `det_shape` | `(ny, nx)`; required only for raw-binary loads. |
| `manual` | `dtype`, `gap`, `offset` | Raw-binary only: NumPy dtype string; byte gap between patterns; byte offset before the first. |
| `manual` | `key` | HDF5/EMD dataset path; if unset, tries `dp`, `data`, `datacube_root/datacube/data`. |
| `manual` | `fftshifted` | `bool`, default `False`; whether the source is already corner-origin (see [below](#corner-origin-diffraction-patterns)). |

!!! warning "Restriction"
    `gatan` accepts `adu` in the schema but never reads it — `gatan.py:40` is
    `adu = 1 #props.adu or meta.adu`, so every Gatan load uses `adu = 1`. Scaling instead
    comes from the file's `e_scaling`/`background_offset` (`gatan.py:76-78`). Treat
    `raw_data.adu` as a no-op for this loader.

*Verification pending: `nion` has no `adu` field and its dose-scaling path is commented out
(`nion.py:88-94`), so Nion data arrives in raw file units. Read from source, not exercised
against a real file.*

## Wavelength

`ReconsPlan.wavelength` (top-level, optional, Å) is only needed when your loader doesn't
compute one, or to override it. If unset, `load_raw_data` (`phaser/execute.py:170-172`) uses
the loader's reported value; if neither supplies one, `execute_plan` raises `ValueError`.
Every built-in loader except a raw-binary `manual` load without `wavelength`/`kv` supplies
its own. Resolved once, before scan/probe/object are built.

## Calibration-relevant `post_load` operations

`post_load` (optional, default `()`) is an ordered list of hooks that transform the whole
raw-data dict (`phaser/execute.py:216-218`) after loading, before state is built. Built-ins
relevant to calibration (full list in the
[Post Load reference](../../generated/hooks/post-load.md)):

| Hook | Property | What it does |
| --- | --- | --- |
| `scale` | `scale: float` | `patterns *= scale`, no noise (`preprocessing.py:37-39`). |
| `poisson` | `scale`, `gaussian` (default `0.001`) | Scales then **replaces** patterns with a Poisson draw, optional Gaussian read noise (`preprocessing.py:60-81`). Standard way to dose noiseless simulated data — `examples/si_grad.yaml:11`, `mos2_grad.yaml:10`. Distinct from the `noise_model: poisson` engine hook. |
| `crop_data` | `crop: (y_i, y_f, x_i, x_f)` | Crops scan axes; updates a `raster` `scan_hook`'s `shape` (`preprocessing.py:18-34`). Used in `examples/si_grad_exp.yaml:11`. |
| `offset` | `offset: float` | `patterns -= offset` (`preprocessing.py:41-43`); removes a known background. |
| `bin` | `bin: int` | Sums `bin×bin` detector blocks (`preprocessing.py:45-57`). |

## Intensity and count scaling

Patterns must be in particle counts, not arbitrary units:
`initialize_reconstruction` warns if the mean total intensity is below `5.0`
(`phaser/execute.py:367-374`). This matters because the **Poisson** loss is only a correct
negative log-likelihood in counts, and count scale changes a regularizer's *effective*
strength — see [Intensity and count scaling](../../architecture/state-and-conventions.md#intensity-and-count-scaling).

- `empad`/`gatan` scale experimental data toward counts automatically (via `adu`, or
  `gatan`'s file-derived scaling), and warn if they cannot.
- Simulated data normalized to peak/sum `1.0` needs an explicit `post_load: scale` or
  `poisson` step first.

## Corner-origin diffraction patterns

!!! warning "Restriction"
    Every built-in loader stores patterns **corner-origin** (zero-frequency in the array
    corner), applying `numpy.fft.ifftshift` to patterns and mask (`empad.py:74,91`,
    `gatan.py:70,92`, `nion.py:86,110`), matching `Patterns.patterns` in `phaser/state.py`.
    See [Diffraction-pattern origin](../../architecture/state-and-conventions.md#diffraction-pattern-origin).

`manual` is the only plan-level choice: it applies the same `ifftshift` unless
`fftshifted: true`, which loads the source as-is (`manual.py:103-105`). Set it only if your
file's zero-frequency sample is already at the corner; setting it wrong silently shifts
every pattern by half its width, with no error.

## Minimal example

EMPAD acquisition with metadata, cropped before reconstruction:

```yaml
raw_data:
  type: empad
  path: "sample_data/experimental_si/acq12_20over.json"
post_load:
  - type: crop_data
    crop: [50, -50, 50, -50]
```

Manually-normalized, corner-origin simulated `.npy`, dosed with `poisson`:

```yaml
raw_data:
  type: manual
  path: "examples/smoke/data/patterns.npy"
  kv: 300.0
  diff_step: 1.538085515954122
  adu: 1.0
  fftshifted: true
post_load:
  - type: poisson
    scale: 5.0e+4
```

*Drawn from `examples/si_grad_exp.yaml` and `examples/smoke/single_slice_gradient.yaml`;
run `phaser validate` on your copy before relying on exact syntax.*

## Maintainer sources

- `phaser/hooks/io/{empad,gatan,nion,manual}.py`
- `phaser/hooks/preprocessing.py`
- `phaser/hooks/__init__.py`
- `phaser/execute.py`
- `phaser/plan.py`
- `examples/si_grad.yaml`, `examples/si_grad_exp.yaml`, `examples/mos2_grad.yaml`, `examples/smoke/single_slice_gradient.yaml`
