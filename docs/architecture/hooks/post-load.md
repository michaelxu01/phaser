# Post-load hooks

A **post-load hook** processes the raw `patterns`/`mask`/metadata dictionary returned by
the [raw-data loader](raw-data-loaders.md), before any reconstruction
[state](../../concepts/glossary.md#state) exists. Configured as a list under a plan's
`post_load` field, it is where cropping, offsetting, scaling, binning, or synthetic
Poisson noise are applied to raw patterns.

## Lifecycle point

Post-load hooks run **once, immediately after the raw-data loader and the
[initialization merge](../lifecycle.md#raw-data-loading-and-the-initialization-merge)**,
and before any [initialization hook](initialization.md) builds probe, scan, tilt, or
object state (`phaser/execute.py:216-218`):

```python
for p in plan.post_load:
    raw_data = p(raw_data)
```

Every hook in the list runs, **in the order given in the plan**, each receiving the
previous hook's output — so a `crop_data` before a `poisson` crops first and adds noise to
the cropped result; the reverse order adds noise to the full patterns and then crops.
Right after this loop, `load_raw_data` materializes a memory-mapped `patterns` array (if
the loader returned one, as `empad` does) into memory (`phaser/execute.py:220-222`) — a
post-load hook still sees (and may still return) a memmap.

## Callable signature and property schema

```python
class PostLoadHook(Hook[RawData, RawData]):
    known = {
        'crop_data': ('phaser.hooks.preprocessing:crop_data', CropDataProps),
        'poisson': ('phaser.hooks.preprocessing:add_poisson_noise', PoissonProps),
        'scale': ('phaser.hooks.preprocessing:scale_patterns', ScaleProps),
        'offset': ('phaser.hooks.preprocessing:offset_patterns', OffsetProps),
        'bin': ('phaser.hooks.preprocessing:bin_patterns', BinProps),
    }
```

(`phaser/hooks/__init__.py:179-217`.) A resolved post-load hook is called with the whole
`RawData` dict as its only argument and its own bound `props`
(`self.resolve()(args, props=...)`, `phaser/hooks/hook.py:61-62`) — the same two-step
shape as [initialization hooks](initialization.md#callable-signature-and-property-schema),
but with `RawData` as both input and output instead of a hook-specific `Args`/return pair.

## Accepted state and returned value

Input and output are both the same `RawData` `TypedDict` a
[raw-data loader](raw-data-loaders.md#accepted-state-and-returned-value) returns:
`patterns`, `mask`, `sampling`, and the optional `wavelength`/`scan_hook`/`tilt_hook`/
`probe_hook`/`seed`. A post-load hook may replace `patterns`/`mask` (changing shape, dtype,
or values) and, if it changes the scan-position count or arrangement, should update
`scan_hook` to match — the built-in `crop_data` does exactly this (see below). Every
built-in mutates and returns the same dict object it was given, rather than constructing a
new one; a custom hook may do either.

## Built-in implementations

Full property schemas are generated from `PostLoadHook.known` in the
[generated post-load hook reference](../../generated/hooks/post-load.md); this table adds
what each one actually does to the data:

| Hook | Purpose | Key properties | Notes |
| --- | --- | --- | --- |
| `crop_data` | Crops `patterns` along its two leading (scan) axes to `[y_i:y_f, x_i:x_f]`. | `crop`: `(y_i, y_f, x_i, x_f)`, each optional (`None` keeps that bound) | Requires `patterns.ndim == 4` (a 2D scan raster plus two detector axes), raising `ValueError` otherwise (`phaser/hooks/preprocessing.py:18-20`). If `scan_hook` is present and its type is `raster`, also updates `scan_hook['shape']` to the cropped shape (`phaser/hooks/preprocessing.py:27-32`) — so a subsequent `raster` scan build reflects the crop. |
| `poisson` | Adds synthetic Poisson (shot) noise, and optionally Gaussian read noise, to `patterns` — for simulated data being made to look experimental. | `scale`: optional pre-multiplier (`float`); `gaussian`: optional read-noise standard deviation (`float`, default `1.0e-3`) | Uses `create_rng(raw_data.get('seed', None), 'poisson_noise')` (`phaser/utils/misc.py`), so its result is deterministic given the plan's `seed`. Computes on NumPy regardless of the array's backend (`to_numpy(...)` before `rng.poisson`), then converts back with `xp.asarray` (`phaser/hooks/preprocessing.py:60-81`). |
| `scale` | Multiplies `patterns` by a constant. | `scale`: `float` | `raw_data['patterns'] *= props.scale`. |
| `offset` | Subtracts a constant from `patterns`. | `offset`: `float` | `raw_data['patterns'] -= props.offset`; does not clip — negative values are possible (a custom hook, like the one below, can clip if that matters for your noise model). |
| `bin` | Sums adjacent detector pixels in non-overlapping `bin x bin` blocks, reducing the two trailing (detector) axes. | `bin`: `int` | Reshapes `(..., Ny, Nx)` to `(..., Ny/bin, bin, Nx/bin, bin)` and sums over the two inserted axes (`phaser/hooks/preprocessing.py:45-57`); `Ny`/`Nx` must be evenly divisible by `bin`, or the reshape raises `ValueError`. |

## Minimal custom implementation

A custom post-load hook is a plain function matching `Hook[RawData, RawData]`'s call
shape — no base class or registration is required to use it as an external hook. This one
clips negative pattern values to zero, a common cleanup step for detector data with a small
negative offset:

```python
import typing as t

import numpy
from phaser.hooks import RawData


def clip_negative(raw_data: RawData, props: t.Mapping[str, t.Any]) -> RawData:
    """
    Signature matches every post-load hook: receives and returns the whole
    `RawData` dict, so it can read or replace `patterns`, `mask`, or any
    loader-supplied metadata key.
    """
    raw_data['patterns'] = numpy.clip(raw_data['patterns'], a_min=0, a_max=None)
    return raw_data
```

Run directly (`python custom_post_load.py`):

```text
clipped patterns:
 [[0. 2.]
 [3. 0.]]
OK: negative values clipped
```

and chained after a built-in hook through a real plan's `post_load` list, in declared
order (`load_raw_data`, `phaser/execute.py`):

```python
plan = ReconsPlan.from_data({
    'name': 'test',
    'raw_data': {'type': 'custom_raw_loader:load_synthetic', ...},
    'init': {'probe': {'type': 'focused', 'conv_angle': 20.0, 'defocus': 300.0}},
    'post_load': [
        {'type': 'scale', 'scale': -1.0},             # built-in: makes values negative
        {'type': 'custom_post_load:clip_negative'},   # external: clips them back to 0
    ],
    'engines': [],
})

raw_data = load_raw_data(plan, numpy)
assert raw_data['patterns'].min() == 0.0
```

which printed `min pattern value after scale+clip: 0.0` and
`OK: built-in and external post_load hooks ran in declared order` — confirming both the
built-in and the external hook ran, in the order listed in the plan.

## YAML invocation

Built-in short names, applied in order:

```yaml
post_load:
  - type: crop_data
    crop: [2, -2, null, null]
  - type: scale
    scale: 100.0
```

External reference, using the custom hook above — its properties are **not**
schema-validated (only a registered short name's properties go through a `pane` dataclass;
see [hook anatomy](index.md#hook-anatomy)):

```yaml
name: test
raw_data:
  type: "custom_raw_loader:load_synthetic"
  scan_shape: [4, 4]
  det_shape: [32, 32]
  wavelength: 0.0251
init:
  probe:
    type: focused
    conv_angle: 20.0
    defocus: 300.0
post_load:
  - type: scale
    scale: -1.0
  - type: "custom_post_load:clip_negative"
engines: []
```

Verified with the packaged validator:

```console
$ phaser validate post_load_plan.yaml
Validation of plan successful!
```

## Engine and backend restrictions

None. Post-load hooks run before any engine is selected and before reconstruction state is
converted to the plan's backend (`state.to_xp(xp)` happens later, once a full `ReconsState`
exists — `phaser/execute.py:355`). The `poisson` hook is the only built-in that explicitly
moves data to NumPy internally (for `numpy.random`-based sampling) before converting it
back to whatever array module `patterns` arrived as.

## Optional dependencies

None. `phaser/hooks/preprocessing.py` uses only `numpy` and Phaser's own utilities
(`phaser.utils.misc.create_rng`), all core dependencies (`pyproject.toml`).

## Testing pattern

No existing test exercises `phaser/hooks/preprocessing.py` directly (verified by search).
The pattern demonstrated above — call the hook function directly with a small, hand-built
`RawData` dict and assert on the returned `patterns`/metadata — is the same one used for
[raw-data loaders](raw-data-loaders.md#testing-pattern) and works without any `ReconsPlan`,
engine, or backend selection:

```python
import numpy
from phaser.hooks import RawData


def test_clip_negative():
    raw_data: RawData = {
        'patterns': numpy.array([[-1.0, 2.0], [3.0, -0.5]], dtype=numpy.float32),
        'mask': numpy.ones((2, 2), dtype=numpy.float32),
        'sampling': None,
    }
    out = clip_negative(raw_data, {})
    assert numpy.all(out['patterns'] >= 0)
```

For a hook that also updates metadata (as `crop_data` updates `scan_hook['shape']`),
additionally assert on the returned metadata dict, and for a hook registered as a
built-in, validate the full plan with `phaser validate` as shown above.

## Maintainer sources

- `phaser/hooks/__init__.py`
- `phaser/hooks/preprocessing.py`
- `phaser/hooks/hook.py`
- `phaser/execute.py`
- `phaser/utils/misc.py`
