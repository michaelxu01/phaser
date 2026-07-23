# Raw-data loader hooks

A [**raw-data loader**](../../concepts/glossary.md#hook) runs first in every reconstruction:
it produces the raw diffraction patterns, detector mask, and coordinate system every later
step builds on, plus optional partial scan/tilt/probe metadata. It's the simplest complete
hook family — one call, no arguments, before any reconstruction state exists — so it's the
recommended first read if you're new to the [hook mechanism](index.md).

## Lifecycle point

A raw-data loader runs **once, first**, called as `plan.raw_data(None)` inside
`load_raw_data` (`phaser/execute.py:162-224`), itself called from
`initialize_reconstruction` (`phaser/execute.py:277`). By this point only
[backend/device selection and observer construction](../lifecycle.md#backend-and-device-selection)
have happened — no `ReconsState`, probe, object, scan, or tilt exists yet. This hook and
`plan.init.*` are the only two sources of that information; see
[Initialization](initialization.md).

Immediately after the hook returns, `load_raw_data`:

1. resolves `wavelength` from the hook's return value or `plan.wavelength` (raises
   `ValueError` if neither supplies it);
2. merges any `scan_hook`/`tilt_hook`/`probe_hook` the loader supplied with the plan's
   `init.scan`/`init.tilt`/`init.probe` — see
   [Initialization merge semantics](../lifecycle.md#raw-data-loading-and-the-initialization-merge)
   and [Initialization hooks](initialization.md);
3. runs every `post_load` hook, in order, over the returned patterns — see
   [Post-load hooks](post-load.md).

## Callable signature and property schema

```python
class RawData(t.TypedDict):
    patterns: NDArray[numpy.floating]
    mask: NDArray[numpy.floating]
    sampling: 'Sampling'
    wavelength: NotRequired[t.Optional[float]]
    scan_hook: NotRequired[t.Union[t.Dict[str, t.Any], None]]
    tilt_hook: NotRequired[t.Union[t.Dict[str, t.Any], None]]
    probe_hook: NotRequired[t.Union[t.Dict[str, t.Any], None]]
    seed: NotRequired[t.Optional[object]]


class RawDataHook(Hook[None, RawData]):
    known = {
        'empad': ('phaser.hooks.io.empad:load_empad', LoadEmpadProps),
        'gatan': ('phaser.hooks.io.gatan:load_gatan', LoadGatanProps, ('rsciio',)),
        'nion': ('phaser.hooks.io.nion:load_nion', LoadNionProps),
        'manual': ('phaser.hooks.io.manual:load_manual', LoadManualProps),
    }
```

(`phaser/hooks/__init__.py:19-27,78-84`.) A resolved raw-data hook is called with **no
argument but its bound properties** — `plan.raw_data(None)` — matching `Hook.__call__`'s
`self.resolve()(args, props=...)` (`phaser/hooks/hook.py:61-62`) with `args=None`; the only
hook family where `args` is always `None`. See [hook anatomy](index.md#hook-anatomy) for how
a built-in short name resolves to `(function, properties dataclass)` versus an external
`"package.module:function"` reference (properties unvalidated).

## Accepted state and returned value

**Input:** `None` — no `ReconsState`, patterns, or prior metadata available.

**Output:** a `RawData` dict:

- **`patterns`** (required) — `NDArray[floating]`, shape `(..., y, x)`: arbitrary leading
  scan-position shape (flat `(n,)`, raster `(ny, nx)`, or other — `_normalize_scan_shape`
  (`phaser/execute.py:128-159`) reconciles it against `state.scan` once both exist), then the
  detector's two trailing axes. **Corner-origin**: the zero-frequency sample must sit in the
  array corner, not centered — every built-in loader enforces this with
  `numpy.fft.ifftshift(..., axes=(-1, -2))`
  ([Diffraction-pattern origin](../state-and-conventions.md#diffraction-pattern-origin)).
- **`mask`** (required) — `NDArray[floating]`, shape `(y, x)`, broadcastable against
  `patterns`' trailing axes: valid detector pixels (excludes a beamstop or unreliable
  border); also corner-origin, shifted the same way as `patterns`.
- **`sampling`** (required) — a `Sampling` object (`phaser/utils/num.py`) describing the
  detector's real-space coordinate system, built from wavelength and detector geometry (e.g.
  `phaser/hooks/io/empad.py:83-84`: `a = wavelength / (diff_step * 1e-3)`,
  `Sampling(patterns.shape[-2:], extent=(a, a))`) — length units are Å throughout
  ([Units and axis ordering](../state-and-conventions.md#units-and-axis-ordering)).
- **`wavelength`** (optional) — `float`, Å. If omitted, `plan.wavelength` must supply it, or
  `load_raw_data` raises `ValueError` (`phaser/execute.py:170-172`).
- **`scan_hook`, `tilt_hook`, `probe_hook`** (all optional) — a plain `dict` shaped like that
  hook's YAML form (`{'type': ..., **properties}`), or `None`/absent. Not constructed hooks
  yet — metadata the initialization merge combines with `plan.init.*` before scan, tilt, or
  probe is built ([Initialization hooks](initialization.md)).
- **`seed`** (optional) — passed through unchanged for hooks needing randomness (e.g. the
  `poisson` post-load hook); every built-in loader sets it `None`, and `load_raw_data`
  overwrites it with the `seed` argument to `execute_plan`/`initialize_reconstruction`
  regardless (`phaser/execute.py:208`).

## Built-in implementations

Full property schemas are generated from `RawDataHook.known` in the
[generated raw-data hook reference](../../generated/hooks/raw-data.md); this table adds what
each loader contributes as metadata, and when.

| Hook | Purpose | Metadata contributed | Notes |
| --- | --- | --- | --- |
| `empad` | Loads a 4D EMPAD dataset from a `.raw` file or a `.json` metadata sidecar (`EmpadMetadata`, `phaser/io/empad.py`) that also supplies voltage, `diff_step`, ADU, and scan/probe geometry. | `probe_hook` (`focused`) and `scan_hook` (`raster`) **only via a `.json` metadata file**; both `None` for a bare `.raw` path (`phaser/hooks/io/empad.py:37-58`). | Scales patterns by ADU when `needs_scale` (`not meta.is_simulated()`) and `adu` is available; warns if not. EMPAD v2 metadata is rejected (`ValueError`, `phaser/hooks/io/empad.py:26-27`). |
| `gatan` | Loads a Gatan `.dm4` 4D dataset via `rsciio.digitalmicrograph` (`phaser/hooks/io/gatan.py`). | `probe_hook` (`focused`) and `scan_hook` (`raster`), always — read unconditionally from the file (`phaser/hooks/io/gatan.py:43-56`). | Requires the optional `rsciio` dependency ([Optional dependencies](#optional-dependencies)). Subtracts a background offset and scales by `e_scaling` when not simulated. |
| `nion` | Loads a Nion Swift `.zip` 4D dataset, reading `metadata.json` inside the archive for scan shape/step/rotation and instrument voltage (`phaser/hooks/io/nion.py`). | `scan_hook` (`raster`) only — `probe_hook` is commented out (`phaser/hooks/io/nion.py:113`, `# 'probe_hook': probe_hook,`) and never returned. | Applies a detector left-right flip when the file's `camera_processing_parameters` records a `flip_l_r` step. |
| `manual` | A generic loader for `.npy`/`.npz`, `.tif`/`.tiff`, `.h5`/`.hdf5`/`.emd`, or headerless raw binary files, with explicit `wavelength`/`kv`, `diff_step`, and (for raw binary) `det_shape`/`dtype`/`gap`/`offset`. | None — `probe_hook`, `scan_hook`, and `tilt_hook` are always `None` (`phaser/hooks/io/manual.py:112-114`); scan and probe must come from `plan.init.*`. | `fftshifted: bool` (default `False`): `False` applies the same `ifftshift` corner-origin normalization as other loaders; `True` assumes the source is already corner-origin and skips it. `.mat` files raise `NotImplementedError`. |

## Minimal custom implementation

A custom raw-data loader is a plain Python function matching `Hook[None, RawData]`'s call
shape — no base class or registration needed to use it as an **external** hook. This one
synthesizes a tiny dataset in-process and demonstrates the corner-origin contract:

```python
import typing as t

import numpy
from phaser.hooks import RawData
from phaser.utils.num import Sampling


def load_synthetic(args: None, props: t.Mapping[str, t.Any]) -> RawData:
    """A minimal external raw-data loader.

    `props` arrives as a plain `dict` (external-hook properties are not
    schema-validated -- see "YAML invocation" below), so this function reads
    it with `.get()` and applies its own defaults, unlike a built-in loader's
    `pane`-validated properties dataclass.
    """
    scan_shape = tuple(props.get('scan_shape', (4, 4)))
    det_shape = tuple(props.get('det_shape', (32, 32)))
    wavelength = props.get('wavelength', 0.0251)

    ny, nx = det_shape
    # a bright disk centered in the array -- as if the detector recorded a
    # centered diffraction pattern before any corner-origin normalization
    yy, xx = numpy.mgrid[:ny, :nx]
    yy, xx = yy - ny // 2, xx - nx // 2
    disk = (yy**2 + xx**2 < (min(ny, nx) // 8) ** 2).astype(numpy.float32)

    patterns = numpy.broadcast_to(disk, (*scan_shape, ny, nx)).copy()
    mask = numpy.ones((ny, nx), dtype=numpy.float32)

    # corner-origin normalization: every built-in loader does this (state.py's
    # `Patterns.patterns` docstring requires 0-frequency sample in the corner)
    patterns = numpy.fft.ifftshift(patterns, axes=(-2, -1))
    mask = numpy.fft.ifftshift(mask, axes=(-2, -1))

    return {
        'patterns': patterns,
        'mask': mask,
        'sampling': Sampling((ny, nx), extent=(50.0, 50.0)),
        'wavelength': wavelength,
        'scan_hook': {
            'type': 'raster',
            'shape': scan_shape,
            'step_size': (1.0, 1.0),
        },
    }
```

Run directly (`python custom_raw_loader.py`):

```text
patterns shape: (4, 4, 32, 32) dtype: float32
mask shape: (32, 32)
sampling: Sampling(shape=array([32, 32]), extent=array([50., 50.]), sampling=array([1.5625, 1.5625]))
wavelength: 0.0251
scan_hook: {'type': 'raster', 'shape': (4, 4), 'step_size': (1.0, 1.0)}
center pixel value: 0.0 corner pixel value: 1.0
OK: corner-origin normalization verified
```

and resolved and invoked through a real plan exactly as the engine would (`plan.raw_data(None)`,
`phaser/execute.py:168`):

```python
from phaser.plan import ReconsPlan

plan = ReconsPlan.from_data({
    'name': 'test',
    'raw_data': {
        'type': 'custom_raw_loader:load_synthetic',
        'scan_shape': (4, 4),
        'det_shape': (32, 32),
        'wavelength': 0.0251,
    },
    'engines': [],
})

raw_data = plan.raw_data(None)
assert raw_data['patterns'].shape == (4, 4, 32, 32)
```

printing `Loaded via plan.raw_data(None): patterns shape: (4, 4, 32, 32) ...` and
`OK: external hook resolved and invoked through the plan` — confirming it resolves through
`importlib.import_module` exactly as [hook anatomy](index.md#lazy-resolution-and-caching)
describes for an external reference.

## YAML invocation

Built-in short name:

```yaml
raw_data:
  type: empad
  path: sample_data/acquisition/metadata.json
  adu: 375.0
```

External reference, using the custom loader above — its properties (`scan_shape`,
`det_shape`, `wavelength`) are **not** schema-validated: only a built-in short name goes
through a properties class ([hook anatomy](index.md#hook-anatomy)):

```yaml
name: test
raw_data:
  type: "custom_raw_loader:load_synthetic"
  scan_shape: [4, 4]
  det_shape: [32, 32]
  wavelength: 0.0251
engines: []
```

Verified with the packaged validator:

```console
$ phaser validate raw_loader_plan.yaml
Validation of plan successful!
```

## Engine and backend restrictions

None. A raw-data loader runs before any engine is selected or backend applied — every
built-in loader returns plain NumPy arrays regardless of `backend`; conversion to the
selected backend (`state.to_xp(xp)`) happens later, once a full `ReconsState` exists
(`phaser/execute.py:355`, [State initialization](../lifecycle.md#state-initialization)). The
`empad` loader reads its `.raw` file with `memmap=True` (`phaser/hooks/io/empad.py:74`);
`load_raw_data` materializes it into memory right after `post_load` hooks run
(`phaser/execute.py:220-222`).

## Optional dependencies

| Loader | Declared dependency | Checked |
| --- | --- | --- |
| `empad` | none | — |
| `gatan` | `rsciio` (`('rsciio',)` on the hook's `known` entry, `phaser/hooks/__init__.py:81`) | `_resolve_ref` calls `check_dependencies` immediately before importing, the first time the hook runs (`phaser/hooks/hook.py:38-40`, [lazy resolution](index.md#lazy-resolution-and-caching)); a missing `rsciio` raises `RuntimeError` naming the hook and install instructions (`phaser/hooks/_dependencies.py`), not a bare `ImportError`. |
| `nion` | none declared | Uses only the standard-library `zipfile`/`json`. |
| `manual` | none declared | Its `.tif`/`.tiff` and `.h5`/`.hdf5`/`.emd` branches import `tifffile`/`h5py` lazily (`phaser/hooks/io/manual.py:41-45`), but both are core Phaser dependencies (`pyproject.toml`), always installed — no dependency check needed. |

## Testing pattern

Modeled on `tests/test_load.py`: build the hook through `pane.from_data({...}, RawDataHook)`
as `ReconsPlan` parsing would, call it with `hook(None)`, and assert on the returned
`RawData` fields — shape, dtype, corner-origin placement, and metadata keys:

```python
import pane
from phaser.hooks import RawDataHook

def test_custom_loader_shape_and_corner_origin():
    hook = pane.from_data(
        {'type': 'custom_raw_loader:load_synthetic', 'scan_shape': (4, 4), 'det_shape': (32, 32)},
        RawDataHook,
    )
    raw_data = hook(None)

    assert raw_data['patterns'].shape == (4, 4, 32, 32)
    assert raw_data['patterns'].dtype == numpy.float32
    # corner-origin: the synthesized disk's energy lands at (0, 0), not the center
    assert raw_data['patterns'][0, 0, 0, 0] == 1.0
    assert raw_data['patterns'][0, 0, 16, 16] == 0.0
```

For a loader that contributes `scan_hook`/`probe_hook`/`tilt_hook`, also assert on those
dict values directly (as `tests/test_initialization.py`'s `test_load_raw_data_override` does
downstream of the loader), and validate a built-in-registered plan with `phaser validate` as
shown above.

## Maintainer sources

- `phaser/hooks/__init__.py`
- `phaser/hooks/hook.py`
- `phaser/hooks/_dependencies.py`
- `phaser/hooks/io/empad.py`
- `phaser/hooks/io/gatan.py`
- `phaser/hooks/io/nion.py`
- `phaser/hooks/io/manual.py`
- `phaser/io/empad.py`
- `phaser/execute.py`
- `phaser/state.py`
- `phaser/utils/num.py`
- `tests/test_load.py`
- `tests/test_initialization.py`
