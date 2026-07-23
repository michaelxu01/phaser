# Initialization hooks

Four hook families build the non-diffraction parts of a reconstruction's initial
[state](../../concepts/glossary.md#state): `ProbeHook`, `ObjectHook`, `ScanHook`, and
`TiltHook` (`phaser/hooks/__init__.py`), configured under a plan's `init.probe`,
`init.object`, `init.scan`, and `init.tilt` fields (`InitPlan`, `phaser/plan.py:19-25`).
This page covers what each family receives and must return; the rules deciding *which*
configuration reaches them (loader-supplied metadata merged with `init.*`, restart reuse,
`{}`) are documented once, in
[Initialization merge semantics](../lifecycle.md#raw-data-loading-and-the-initialization-merge).

## Lifecycle point

All four hooks run **once, during state construction**, inside `initialize_reconstruction`
(`phaser/execute.py:227-381`), after [raw-data loading](raw-data-loaders.md), the
[initialization merge](../lifecycle.md#raw-data-loading-and-the-initialization-merge), and
[post-load hooks](post-load.md) — before [post-init hooks](post-init.md), which need a
complete `ReconsState` to run against. Order (`phaser/execute.py:288-345`): **probe**,
**scan**, **tilt**, **object**.

For each of probe, scan, tilt, and object independently, `initialize_reconstruction` first
decides whether to **reuse** that component from `init_state` (a restart file or a
Python-supplied `init_state=`) or **construct it fresh** by calling the merged hook:

- reused only if `init_state.X is not None` **and** `plan.init.X is None` — the plan does
  not configure that component at all;
- an empty mapping `plan.init.X = {}` still counts as configuring it, so it is *not* reused
  even on restart — it requests a fresh, metadata-derived build instead;
- **tilt** has one extra rule: if no tilt metadata hook is present (merged or
  loader-supplied) and `state.tilt` isn't being reused, `initialize_reconstruction` leaves
  `state.tilt` as `None` rather than calling the hook with nothing to build from
  (`phaser/execute.py:316-325`) — tilt can still be created later, as a zeroed array, at an
  [engine boundary](../lifecycle.md#engine-boundary-reshaping) that first enables tilt
  refinement.

See [Initialization merge semantics](../lifecycle.md#raw-data-loading-and-the-initialization-merge)
for the full merge/reuse/`{}` rule and its `tests/test_initialization.py` evidence.

## Callable signature and property schema

Each family is `Hook[Args, Return]` with its own argument `TypedDict`
(`phaser/hooks/__init__.py`):

```python
class ProbeHookArgs(t.TypedDict):
    sampling: 'Sampling'
    wavelength: float
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any

class ProbeHook(Hook[ProbeHookArgs, 'ProbeState']):
    known = {'focused': ('phaser.hooks.probe:focused_probe', FocusedProbeProps)}


class ObjectHookArgs(t.TypedDict):
    sampling: 'ObjectSampling'
    wavelength: float
    slices: t.Optional[Slices]
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any

class ObjectHook(Hook[ObjectHookArgs, 'ObjectState']):
    known = {'random': ('phaser.hooks.object:random_object', RandomObjectProps)}


class ScanHookArgs(t.TypedDict):
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any

class ScanHook(Hook[ScanHookArgs, NDArray[numpy.floating]]):
    known = {'raster': ('phaser.hooks.scan:raster_scan', RasterScanProps)}


class TiltHookArgs(t.TypedDict):
    dtype: DTypeLike
    xp: t.Any
    shape: t.Tuple[int, ...]  # matches the scan shape

class TiltHook(Hook[TiltHookArgs, NDArray[numpy.floating]]):
    known = {
        'global': ('phaser.hooks.tilt:generate_global_tilt', GlobalTiltProps),
        'custom': ('phaser.hooks.tilt:load_custom_tilt', CustomTiltProps),
    }
```

(`phaser/hooks/__init__.py:87-168`.) Every resolved hook is called with one `Args`
dictionary and its own bound `props` — `self.resolve()(args, props=...)`
(`phaser/hooks/hook.py:61-62`) — like every other two-step hook family (contrast
[raw-data loaders](raw-data-loaders.md#callable-signature-and-property-schema), whose `args`
is always `None`, and [flags/schedules](schedules-and-flags.md), not two-step at all).

Call sites, with the concrete arguments passed (`phaser/execute.py:301-342`):

```python
probe = pane.from_data(probe_hook, ProbeHook)(
    {'sampling': sampling, 'wavelength': wavelength, 'dtype': dtype, 'seed': seed, 'xp': xp}
)
scan = pane.from_data(scan_hook, ScanHook)({'dtype': dtype, 'seed': seed, 'xp': xp})
tilt = pane.from_data(tilt_hook, TiltHook)({'dtype': dtype, 'xp': xp, 'shape': scan.shape[:-1]})
obj = (plan.init.object or pane.from_data('random', ObjectHook))({
    'sampling': obj_sampling, 'slices': plan.slices, 'wavelength': wavelength,
    'dtype': dtype, 'seed': seed, 'xp': xp
})
```

`object` defaults to the built-in `random` hook when `plan.init.object` is unset (no
metadata-merge step for `object` — no loader supplies object metadata); `obj_sampling` is
derived from the scan's extent, not passed in directly:
`ObjectSampling.from_scan(scan, sampling.sampling, sampling.extent / 2. + obj_pad_px *
sampling.sampling)`, where `obj_pad_px` comes from **the first engine in `plan.engines`**
(`plan.engines[0].obj_pad_px`), falling back to `5.0` — `EnginePlan.obj_pad_px`'s own default
(`phaser/plan.py`) — if `plan.engines` is empty (`phaser/execute.py:327-330`).

## Accepted state and returned value

| Hook family | Input (`Args`) | Returns | Shape / units |
| --- | --- | --- | --- |
| `ProbeHook` | `sampling: Sampling`, `wavelength: float`, `seed`, `dtype`, `xp` | `ProbeState` | `data`: complex, `(modes, y, x)` — a 2D result is reshaped to `(1, y, x)` at the call site (`phaser/execute.py:304-305`) |
| `ObjectHook` | `sampling: ObjectSampling`, `wavelength: float`, `slices: Optional[Slices]`, `seed`, `dtype`, `xp` | `ObjectState` | `data`: complex, `(z, y, x)`; `thicknesses`: length units (Å), `(z,)` — a 2D result is reshaped to `(1, y, x)` with `thicknesses = []` (single slice, `phaser/execute.py:343-345`) |
| `ScanHook` | `seed`, `dtype`, `xp` | `NDArray[floating]` | `(..., 2)`, last axis `(y, x)`, length units (Å) |
| `TiltHook` | `dtype`, `xp`, `shape: Tuple[int, ...]` (the scan's leading shape) | `NDArray[floating]` | shape matching `shape + (2,)`, last axis `(y, x)`, **mrad** |

Units and axis order match
[State and scientific conventions](../state-and-conventions.md#units-and-axis-ordering)
throughout — `y, x` order, Å for length, mrad for tilt and probe convergence angle. After
all four are built, `state.to_xp(xp)` converts every array to the plan's selected backend
(`phaser/execute.py:355`), and `_normalize_scan_shape` (`phaser/execute.py:128-159`) reshapes
`state.scan`, `patterns.patterns`, and `state.tilt` (if present) to a common leading shape,
whichever of the scan or pattern shape has more dimensions.

## Built-in implementations

Full property schemas are generated from each family's `known` registry:
[Probe](../../generated/hooks/probe.md), [Object](../../generated/hooks/object.md),
[Scan](../../generated/hooks/scan.md), [Tilt](../../generated/hooks/tilt.md). This table adds
what's needed to choose between them:

| Hook | Family | Purpose | Key properties |
| --- | --- | --- | --- |
| `focused` | Probe | Builds a focused-probe wavefunction from convergence angle, defocus, and optional aberrations (`phaser.utils.optics.make_focused_probe`). Raises `ValueError` if `conv_angle` or `defocus` is still unset after the metadata merge. | `conv_angle` (mrad), `defocus` (Å, `+` is overfocus), `aberrations` (sequence of `Aberration`) |
| `random` | Object | Fills the object with a random-phase transmission function of small amplitude (`phaser.utils.object.random_phase_object`), sized from `ObjectSampling` and, if `slices` is set, one thickness-defined slice per entry. | `sigma` (phase noise standard deviation, dimensionless, default `1e-6`) |
| `raster` | Scan | Builds a regular raster grid (`phaser.utils.scan.make_raster_scan`), optionally rotated and/or affine-corrected. Raises `ValueError` if `shape` or `step_size` is still unset after the merge. | `shape` (`(ny, nx)`), `step_size` (Å, scalar or `(y, x)`), `rotation` (degrees, CCW), `affine` (`(2, 2)` array) |
| `global` | Tilt | Broadcasts one uniform `[ty, tx]` tilt (mrad) to every scan position. | `tilt`: `(2,)` array, mrad |
| `custom` | Tilt | Loads a per-position tilt array from a `.npy` file, accepting either `(ny, nx, 2)` (matching the scan shape) or `(n, 2)` (reshaped to match). | `path` |

## Minimal custom implementation

A custom initialization hook is a plain function matching its family's `Args`/return
contract above — no base class needed to use it as an external hook. This one replaces the
built-in `raster` scan with an Archimedean-spiral scan (the same pattern applies to a custom
probe/object/tilt hook, using that family's `Args`/return types):

```python
import typing as t

import numpy
from numpy.typing import NDArray


def spiral_scan(args: t.Mapping[str, t.Any], props: t.Mapping[str, t.Any]) -> NDArray[numpy.floating]:
    """
    Returns scan positions `(y, x)` in the same length units (A) as the built-in
    `raster` scan's `step_size`, with shape `(n_points, 2)`.
    """
    n_points = int(props.get('n_points', 16))
    spacing = float(props.get('spacing', 1.0))  # A per turn step

    xp = args['xp']
    dtype = args['dtype']

    theta = numpy.linspace(0, 4 * numpy.pi, n_points, dtype=dtype)
    r = spacing * theta / (2 * numpy.pi)
    y = r * numpy.sin(theta)
    x = r * numpy.cos(theta)

    scan = numpy.stack([y, x], axis=-1).astype(dtype)
    return xp.asarray(scan)
```

Called directly, with a synthetic `args` dict (`python custom_scan_hook.py`):

```text
scan shape: (8, 2) dtype: float32
[[ 0.0000000e+00  0.0000000e+00]
 [ 5.5710161e-01 -1.2715481e-01]
 [-4.9586713e-01 -1.0296786e+00]
 ...
 [ 1.3987644e-06  4.0000000e+00]]
OK: custom scan hook produces (n_points, 2) array
```

and through a real plan, replacing the loader-supplied scan metadata entirely (a different
hook `ref` than the loader's `raster`, so the merge rule
[replaces rather than merges](../lifecycle.md#raw-data-loading-and-the-initialization-merge)):

```python
from phaser.plan import ReconsPlan
from phaser.execute import initialize_reconstruction

plan = ReconsPlan.from_data({
    'name': 'test',
    'raw_data': {
        'type': 'custom_raw_loader:load_synthetic',
        'scan_shape': (4, 2), 'det_shape': (32, 32), 'wavelength': 0.0251,
    },
    'init': {
        'scan': {'type': 'custom_scan_hook:spiral_scan', 'n_points': 8, 'spacing': 2.0},
        'probe': {'type': 'focused', 'conv_angle': 20.0, 'defocus': 300.0},
    },
    'engines': [],
})

recons = initialize_reconstruction(plan, xp=numpy)
```

printing `state.scan shape: (4, 2, 2)` — the spiral hook's raw `(8, 2)` output reshaped by
`_normalize_scan_shape` to match the loader's `(4, 2)`-shaped patterns, per
[Accepted state and returned value](#accepted-state-and-returned-value) above — followed by
`OK: custom scan hook used for state initialization`.

## YAML invocation

Built-in short names:

```yaml
init:
  probe:
    type: focused
    conv_angle: 20.0
    defocus: 300.0
  scan:
    type: raster
    step_size: [1.0, 1.0]
  object:
    type: random
    sigma: 1.0e-6
```

External reference, using the custom scan hook above — its properties (`n_points`,
`spacing`) are **not** schema-validated (only a registered short name's properties go
through a `pane` dataclass; [hook anatomy](index.md#hook-anatomy)):

```yaml
name: test
raw_data:
  type: "custom_raw_loader:load_synthetic"
  scan_shape: [4, 2]
  det_shape: [32, 32]
  wavelength: 0.0251
init:
  scan:
    type: "custom_scan_hook:spiral_scan"
    n_points: 8
    spacing: 2.0
  probe:
    type: focused
    conv_angle: 20.0
    defocus: 300.0
engines: []
```

Verified with the packaged validator:

```console
$ phaser validate init_plan.yaml
Validation of plan successful!
```

## Engine and backend restrictions

None specific to initialization — all four hooks run once during state construction, before
any engine is prepared, and already receive the plan's selected `dtype`/`xp` directly
(unlike raw-data loaders, which return plain NumPy and convert later). The one
engine-shaped dependency is `obj_pad_px`, taken from **the first entry** in `plan.engines`
when sizing the object ([Callable signature](#callable-signature-and-property-schema)
above) — a plan with no engines falls back to the same default (`5.0`)
`EnginePlan.obj_pad_px` itself uses.

## Optional dependencies

None. `focused` (`phaser.utils.optics`), `random` (`phaser.utils.object`), `raster`
(`phaser.utils.scan`), and `global`/`custom` (`phaser.hooks.tilt`, using only `numpy.load`
for `custom`) all use core dependencies only (`pyproject.toml`).

## Testing pattern

Modeled on `tests/test_initialization.py`: call a hook directly with a synthetic `Args`
dict to unit-test its logic, and drive `initialize_reconstruction` (or `load_raw_data` for
the merge step alone) with a `ReconsPlan` to integration-test merge/reuse behavior:

```python
import numpy

def test_spiral_scan_shape_and_dtype():
    scan = spiral_scan(
        {'xp': numpy, 'dtype': numpy.float32, 'seed': None},
        {'n_points': 8, 'spacing': 2.0},
    )
    assert scan.shape == (8, 2)
    assert scan.dtype == numpy.float32


def test_spiral_scan_replaces_loader_metadata():
    # a different hook `ref` than the loader's 'raster' fully replaces it
    # (phaser/execute.py's merge() -- see lifecycle.md)
    from phaser.plan import ReconsPlan
    from phaser.execute import initialize_reconstruction

    plan = ReconsPlan.from_data({...})  # as in "Minimal custom implementation" above
    recons = initialize_reconstruction(plan, xp=numpy)
    assert recons.state.scan.shape == (4, 2, 2)
```

For restart/merge behavior specifically, follow `tests/test_initialization.py`'s pattern of
constructing a `PartialReconsState` with only some fields set and asserting which
components were reused versus rebuilt (`test_load_raw_data_prev_state`).

## Maintainer sources

- `phaser/hooks/__init__.py`
- `phaser/hooks/probe.py`
- `phaser/hooks/object.py`
- `phaser/hooks/scan.py`
- `phaser/hooks/tilt.py`
- `phaser/hooks/hook.py`
- `phaser/execute.py`
- `phaser/state.py`
- `phaser/utils/object.py`
- `tests/test_initialization.py`
