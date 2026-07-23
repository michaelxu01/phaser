# Simulation geometry

The options that set the simulation's physical extent and resolution: `sim_shape`,
`resize_method`, `bwlim_frac`, `obj_pad_px`, `slices`, `probe_modes`, `base_mode_power`.
All are engine-plan fields (types in the [plan reference](../../generated/plan/index.md)),
except `slices`, which also exists on `ReconsPlan` for the object's *initial* geometry. Each
is re-evaluated at every engine boundary — the mechanism behind staged reconstructions, its
mechanics covered in
[Engine-boundary reshaping](../../architecture/lifecycle.md#engine-boundary-reshaping). This
page covers what each controls physically.

## `sim_shape` and `resize_method`

| Field | Type (generated) | Units | Lifecycle stage |
| --- | --- | --- | --- |
| `sim_shape` | `tuple[int, int] \| None`, default `None` ([plan reference](../../generated/plan/index.md#conventionalengineplan)) | detector pixels `(ny, nx)` | Engine boundary: compared against the probe's *current* shape when that engine starts. |
| `resize_method` | `'pad_crop' \| 'resample'`, default `'pad_crop'` | — | Same engine boundary, only consulted when `sim_shape` actually differs from the current shape. |

`sim_shape` sets the pixel dimensions of the simulated diffraction plane — and, since
probe and object share this array, of the probe and (transversely) the object too. If it
differs from the probe's current shape when an engine starts, `prepare_for_engine`
resamples the probe and loaded patterns (and mask) to it (`phaser/execute.py:391-404`).
`resize_method` decides *how* that resample changes the physical pixel size:

- **`pad_crop`** (the default) keeps the probe's physical extent (`Sampling(shape,
  extent=...)`) fixed and pads or crops pixels — the real-space pixel size changes
  inversely with `sim_shape`, growing `sim_shape` gives a *finer* real-space pixel over the
  same physical field of view.
- **`resample`** keeps the probe's real-space pixel size (`Sampling(shape,
  sampling=...)`) fixed and changes the physical extent — growing `sim_shape` covers *more*
  physical area at the same resolution.

Examples only ever grow `sim_shape` with `pad_crop` between `gradient` stages
(`examples/czo_grad.yaml:30,86,151` is `128→128→192`, `prsco3_grad.yaml:22,74` is `128→256`);
`resample` guidance is pending. Growing `sim_shape` raises compute and memory roughly with
pixel count — see [Grouping and memory](grouping-and-memory.md), and `czo_grad.yaml:151`
("256 exceed mem limit") for a memory-capped case.

## `bwlim_frac`

| Type (generated) | Default | Units |
| --- | --- | --- |
| `float \| None` | `2/3` (`0.6666...`) | fraction of the probe's maximum spatial frequency |

`bwlim_frac` band-limits the multislice free-space propagator: `make_propagators`
(`phaser/engines/common/simulation.py:163-193`, shared by both engine families) masks
spatial frequencies above `bwlim_frac * k_max` (the probe sampling's maximum spatial
frequency) before each inter-slice Fresnel propagator, suppressing the aliasing a
plane-wave propagator otherwise introduces at high angle. `bwlim_frac: null` disables
band-limiting entirely (an all-pass propagator).

**Lifecycle stage:** consulted once per engine, when that engine's propagators are built
(`phaser/engines/gradient/run.py:209`, `phaser/engines/conventional/run.py:78`) — not
re-evaluated per group or per iteration.

**Interactions:** `make_propagators` returns `None` (no propagation) when the object has
fewer than two slices (`len(delta_zs) == 0`, `phaser/engines/common/simulation.py:171-173`)
— **`bwlim_frac` has no effect on a single-slice object**; it matters only once `slices`
(below) adds more than one. `examples/czo_grad.yaml:32,89,153` narrows `bwlim_frac` from
`1.0` to `0.8` across engine stages of one multislice reconstruction (comment: "limit to ~2
alpha"); guidance beyond that one observed case is pending.

## `obj_pad_px`

| Type (generated) | Default | Units |
| --- | --- | --- |
| `float` | `5.0` | probe pixels |

`obj_pad_px` sets how far the object's field of view extends beyond the probe's own extent
at the edge of the scanned region: the object is padded (never shrunk) to cover the scan
extent plus `probe.sampling.extent / 2 + obj_pad_px * probe.sampling.sampling`
(`phaser/execute.py:413-414`), so the probe never illuminates past the reconstructed
object's edge even at the outermost scan positions. Used identically at initial object
construction (`phaser/execute.py:327-330`, the *first* engine's `obj_pad_px`) and at every
later engine boundary (that engine's own value). Practical valid range beyond
"non-negative" is not evidenced — every inspected example leaves it at the schema default;
selection guidance is pending.

## `slices`

| Location | Type (generated) | Notes |
| --- | --- | --- |
| `ReconsPlan.slices` (top level) | `SliceList \| SliceStep \| SliceTotal \| None`, default `None` ([plan reference](../../generated/plan/index.md#reconsplan)) | Passed to the object-initialization hook; the built-in `random` object hook treats `None` as single-slice. |
| `EnginePlan.slices` (per engine) | same union type, default `None` ([plan reference](../../generated/plan/index.md#conventionalengineplan)) | `None` means *keep the object's current slice count and thicknesses* at this engine boundary — it is not equivalent to "single slice." |

Three ways to specify slice thicknesses, all producing a list of per-slice thicknesses
(`Slices.thicknesses`, `phaser/types.py`):

| Variant | Fields | Thickness computation |
| --- | --- | --- |
| `SliceList` | `thicknesses: list[float]` (Å) | Used exactly as given — arbitrary, non-uniform slice thicknesses. |
| `SliceStep` | `n: int`, `slice_thickness: float` (Å) | `n` slices, each exactly `slice_thickness` thick. |
| `SliceTotal` | `n: int`, `total_thickness: float` (Å) | `n` slices, each `total_thickness / n` thick (uniform division of a known total specimen thickness). |

Every repository example using `slices` uses `SliceTotal` at the top level, with `n`
ranging 10–50 and `total_thickness` 190–210 Å (`examples/si_grad.yaml:17-19`,
`examples/prsco3_grad.yaml:16-18`, `examples/czo_grad.yaml:23-25`); none overrides `slices`
per engine. Selection guidance for choosing `n` versus a physical slice thickness, beyond
"matches the specimen's known total thickness," is pending.

**Interactions:** changing slice count or thicknesses at an engine boundary triggers
`resample_slices` (see
[Engine-boundary reshaping](../../architecture/lifecycle.md#engine-boundary-reshaping)),
and only a multislice object (more than one slice) is affected by `bwlim_frac` above.

## `probe_modes` and `base_mode_power`

| Field | Type (generated) | Default | Units |
| --- | --- | --- | --- |
| `probe_modes` | `int` | `1` | mode count |
| `base_mode_power` | `float` | `0.7` | fraction of total probe intensity (see valid range below) |

`probe_modes` sets how many orthogonal, incoherently-summed
[modes](../../concepts/glossary.md#mixed-state) the probe carries at this engine — a single
coherent probe (`1`, the default) versus a mixed-state probe modeling partial spatial
coherence. When an engine's `probe_modes` differs from the probe's current count at that
boundary: reducing it truncates to the first `probe_modes` modes (no intensity
redistribution — `phaser/execute.py:429-430` marks this a `# TODO`); increasing it sums the
existing modes in real space and recreates that many via `make_hermetian_modes`, assigning
`base_mode_power` of total intensity to the base mode and splitting the rest evenly among
the new higher-order modes (`phaser/utils/optics.py:80-98`).

**Valid range:** `base_mode_power` must satisfy `0.0 <= base_mode_power < 1.0` —
`make_hermetian_modes` raises `ValueError` otherwise (`phaser/utils/optics.py:91-92`); `1.0`
itself is invalid (no intensity left for higher-order modes). `probe_modes` has no
enforced upper bound; repository examples range 4–8 (`examples/si_grad.yaml:27`,
`examples/mos2_lsqml.yaml:22`, `examples/czo_grad.yaml:31,87,152`,
`examples/prsco3_grad.yaml:23,75`) — guidance for a specific count beyond "matches expected
partial coherence" is pending.

**Interactions:** see
[Mixed-state probe modes](../recipes/mixed-state-probe-modes.md) (once written) for a
worked staged reconstruction that increases `probe_modes` between engines — this is exactly
the engine-boundary mode-count-change case described above.

## Per-engine overrides

Every option here is per-engine, so a plan's `engines` list can set a different `sim_shape`,
`probe_modes`, or slice count per stage; `prepare_for_engine` reshapes the shared state
whenever a later value differs. This is what makes coarse-to-fine and
single-slice-then-multislice workflows work without hand-building arrays — the full ordered
reshaping list is in
[Engine-boundary reshaping](../../architecture/lifecycle.md#engine-boundary-reshaping).

## Minimal example

A two-stage plan that reconstructs coarsely first, then increases both simulated
resolution and probe-mode count for a refinement stage:

```yaml
slices:
  n: 10
  total_thickness: 200

engines:
  - type: gradient
    sim_shape: [128, 128]
    probe_modes: 1
    niter: 50
    # ...

  - type: gradient
    sim_shape: [192, 192]
    resize_method: pad_crop
    probe_modes: 4
    niter: 100
    # ...
```

*Verification pending:* illustrates the shape of a staged plan; not validated with
`phaser validate` (omits required fields like `noise_model`/`solvers` for brevity) — see
[Simulated single-slice gradient](../reconstructions/simulated-single-slice-gradient.md) for
a complete, validated plan.

## Maintainer sources

- `phaser/plan.py`
- `phaser/types.py`
- `phaser/execute.py`
- `phaser/engines/common/simulation.py`
- `phaser/engines/gradient/run.py`
- `phaser/engines/conventional/run.py`
- `phaser/utils/optics.py`
- `examples/czo_grad.yaml`, `examples/si_grad.yaml`, `examples/prsco3_grad.yaml`, `examples/mos2_lsqml.yaml`
