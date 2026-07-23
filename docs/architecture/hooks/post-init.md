# Post-init hooks

A **post-init hook** processes the raw diffraction `Patterns` together with the
complete, just-built [`ReconsState`](../../concepts/glossary.md#state) — probe, object,
scan, and (if configured) tilt — before any [engine](engines.md) runs. Configured as a
list under a plan's `post_init` field, it is where invalid patterns can be dropped or the
diffraction-pattern origin can be aligned, now that a real scan and a real probe/object
sampling exist to check patterns against.

## Lifecycle point

Post-init hooks run **once, after state construction and backend conversion, and before
the first engine's `prepare_for_engine`** (`phaser/execute.py:355-363`):

```python
state = state.to_xp(xp)  # TODO: figure out why this isn't already the case
data, state = _normalize_scan_shape(data, state)

# process post_init hooks
for p in plan.post_init:
    (data, state) = p({
        'data': data, 'state': state,
        'dtype': dtype, 'seed': seed, 'xp': xp
    })
```

By this point every array in `state` has been converted to the plan's selected backend
(`state.to_xp(xp)`), and `_normalize_scan_shape` (`phaser/execute.py:128-159`) has reshaped
`state.scan`, `data.patterns`, and (if present) `state.tilt` to a common leading shape.
Unlike a [post-load hook](post-load.md) (which only sees the raw `patterns`/`mask`
dictionary before any state exists), a post-init hook can read the actual scan positions,
probe sampling, and object it will reconstruct against — what makes pattern-dropping and
diffraction-origin alignment possible here and not earlier.

Every hook in the list runs, **in the order given in the plan**, each receiving the
previous hook's `(data, state)` output. Immediately after the loop, `initialize_reconstruction`
computes the mean total pattern intensity and warns if it's below `5.0`
(`phaser/execute.py:367-374`) — since a post-init hook can change `patterns` (e.g. by
dropping some), this check runs against whatever the last post-init hook returned, not the
loader's original patterns. After `initialize_reconstruction` returns a `PreparedRecons`,
the first engine's `prepare_for_engine`
([Engine-boundary reshaping](../lifecycle.md#engine-boundary-reshaping)) is the next thing to
touch `state` — post-init hooks are the last processing step not specific to any one
engine.

## Callable signature and property schema

```python
class PostInitArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class PostInitHook(Hook[PostInitArgs, t.Tuple['Patterns', 'ReconsState']]):
    known = {
        'drop_nans': ('phaser.hooks.preprocessing:drop_nan_patterns', DropNanProps),
        'diffraction_align': ('phaser.hooks.preprocessing:diffraction_align', DiffractionAlignProps),
    }
```

(`phaser/hooks/__init__.py:171-176,220-224`.) A resolved post-init hook is called with the
whole `PostInitArgs` dict as its only argument and its own bound `props`
(`self.resolve()(args, props=...)`, `phaser/hooks/hook.py:61-62`) — the same two-step call
shape as every other configuration hook
([hook anatomy](index.md#a-resolved-hook-is-a-plain-callable)). Post-init differs from
[post-load](post-load.md#callable-signature-and-property-schema) in one respect: its input
(`PostInitArgs`, a five-key dict including `state`, `dtype`, `seed`) and output (a plain
`(Patterns, ReconsState)` tuple) aren't the same shape, whereas post-load's input and output
are both the same `RawData` dict.

## Accepted state and returned value

Input is the `PostInitArgs` dict shown above:

- `data: Patterns` — `patterns` (measured intensities) and `pattern_mask`
  (`phaser/state.py:18-27`), already shape-normalized against the scan by
  `_normalize_scan_shape`.
- `state: ReconsState` — the complete, backend-converted probe/object/scan/tilt state.
- `dtype`, `xp`, `seed` — the selected real dtype, backend module, and RNG seed for this
  run, passed straight through from `initialize_reconstruction`.

A post-init hook returns `(data, state)` as a plain tuple — not a dict — and may replace
either or both: `drop_nan_patterns` reassigns `state.scan`, `state.tilt`, and
`data.patterns` (removing whole scan positions); `diffraction_align` reassigns
`data.patterns` and `data.pattern_mask` in place. Both built-ins mutate the objects they
were given and return those same, mutated objects; a custom hook may do either.

## Built-in implementations

Full property schemas are generated from `PostInitHook.known` in the
[generated post-init hook reference](../../generated/hooks/post-init.md); this table adds
what each one does to the data:

| Hook | Purpose | Key properties | Notes |
| --- | --- | --- | --- |
| `drop_nans` | Drops whole scan positions whose pattern is at least `threshold` NaN. | `threshold`: `float`, default `0.9` | Flattens `state.scan`, `state.tilt` (if present), and `data.patterns` to one leading axis, computes `fraction_nan = sum(isnan(pattern)) / pixel_count` per position, and drops every position where that fraction exceeds `threshold` (`phaser/hooks/preprocessing.py:84-119`). Raises `ValueError` if the (flattened) scan-position count does not match the pattern count before *or* after filtering — a defensive check against a scan/pattern mismatch introduced earlier in the pipeline. Logs how many positions were dropped, or is silent if none were. |
| `diffraction_align` | Shifts every diffraction pattern (and the pattern mask) to center the intensity-weighted centroid at the zero-frequency corner. | none | Takes no properties (`DiffractionAlignProps` declares no fields). Computes the mean pattern over sparse groups of up to 128 positions (`create_sparse_groupings`, `phaser/utils/misc.py`), finds its intensity-weighted centroid in a `(1.0, 1.0)`-extent reciprocal grid, and shifts every pattern (and the mask once) via a backend-generic bilinear `affine_transform` (`phaser.utils.image.affine_transform`, `phaser/hooks/preprocessing.py:122-159`) — a Torch-specific kernel under the Torch backend, a shared implementation otherwise (`phaser/utils/image.py:195-201`). Logs the pixel shift applied. |

Both built-ins operate on the corner-origin convention documented in
[Diffraction origin and normalization](../state-and-conventions.md#intensity-and-count-scaling);
`diffraction_align`'s `fftshift`/`ifftshift` pair inside `bilinear_shift`
(`phaser/hooks/preprocessing.py:147-151`) shifts a corner-origin pattern by treating it as
centered only for the duration of the interpolation.

## Minimal custom implementation

A custom post-init hook is a plain function matching `Hook[PostInitArgs, t.Tuple[Patterns,
ReconsState]]`'s call shape — no base class or registration needed to use it as an external
hook. This one drops any scan position whose pattern's total intensity is exactly zero (a
distinct failure mode from `drop_nans`'s NaN-fraction threshold — detector dead-time or a
masked-out acquisition might record all-zero rather than all-NaN):

```python
import typing as t

import numpy
from phaser.hooks import PostInitArgs
from phaser.state import Patterns, ReconsState


def drop_zero_patterns(
    args: PostInitArgs, props: t.Mapping[str, t.Any]
) -> t.Tuple[Patterns, ReconsState]:
    """
    Signature matches every post-init hook: receives the whole PostInitArgs
    dict (data, state, dtype, seed, xp) and returns a (Patterns, ReconsState)
    tuple, so it can read or replace patterns, the pattern mask, or any part
    of state (scan, tilt, probe, object).
    """
    data, state = args['data'], args['state']
    xp = args['xp']

    scan = state.scan.reshape(-1, 2)
    patterns = data.patterns.reshape(-1, *data.patterns.shape[-2:])

    total_intensity = xp.sum(patterns, axis=(-1, -2))
    keep = total_intensity > 0

    if not bool(xp.all(keep)):
        patterns = patterns[keep]
        scan = scan[keep]
        if state.tilt is not None:
            state.tilt = state.tilt.reshape(-1, 2)[keep]

    state.scan = scan
    data.patterns = patterns
    return (data, state)
```

!!! note "Verification pending: example not yet executed"
    Not run against a constructed `ReconsState` (cross-checked against `drop_nan_patterns`,
    `phaser/hooks/preprocessing.py:84-119`, and the `PostInitArgs`/`Patterns`/`ReconsState`
    definitions); execute via the [testing pattern](#testing-pattern) below before relying
    on it as confirmed-working example code.

## YAML invocation

Built-in short names, applied in order:

```yaml
post_init:
  - type: drop_nans
    threshold: 0.9
  - type: diffraction_align
```

External reference, using the custom hook above — its properties are **not**
schema-validated (only a registered short name's properties go through a `pane` dataclass;
see [hook anatomy](index.md#hook-anatomy)):

```yaml
name: test
raw_data:
  type: manual
  path: patterns.h5
  det_shape: [64, 64]
  diff_step: 0.5
  kv: 300.0
init:
  probe:
    type: focused
    conv_angle: 20.0
    defocus: 300.0
post_init:
  - type: diffraction_align
  - type: "custom_post_init:drop_zero_patterns"
engines: []
```

!!! note "Verification pending"
    Constructed from the schema (`phaser/plan.py`, `phaser/hooks/__init__.py`) but not run
    through `phaser validate`; validate before relying on exact syntax.

## Engine and backend restrictions

None specific to the post-init mechanism itself. Post-init hooks run once, before any
engine's `prepare_for_engine`, and after `state` has been converted to the plan's selected
backend (`state.to_xp(xp)`, immediately before the post-init loop, `phaser/execute.py:355`)
— so a post-init hook always sees arrays on the reconstruction's actual backend, unlike a
post-load hook, which may still see NumPy-loaded, not-yet-converted data.
`diffraction_align`'s `phaser.utils.image.affine_transform` does branch internally on
backend (a dedicated Torch kernel path, `phaser/utils/image.py:197-201`), but this is an
implementation detail of that built-in, not a restriction on the hook family.

## Optional dependencies

None. `phaser/hooks/preprocessing.py` uses only `numpy` and Phaser's own utilities
(`phaser.utils.misc.create_sparse_groupings`, `phaser.utils.image.affine_transform`), all
core dependencies (`pyproject.toml`).

## Testing pattern

No existing test exercises `phaser/hooks/preprocessing.py` directly (same gap noted on
[post-load's testing pattern](post-load.md#testing-pattern)). The pattern is the same one
used there and for [raw-data loaders](raw-data-loaders.md#testing-pattern), extended to
include a real `state`: build a small `ReconsState` directly (a tiny `object`/`probe`/`scan`,
optionally `tilt`), a small `Patterns` with a known NaN, zero, or off-center pattern, call the
hook function directly with a hand-built `PostInitArgs` dict, and assert on the returned
`patterns`/`state` fields:

```python
import numpy
from phaser.hooks import PostInitArgs


def test_drop_zero_patterns():
    # Build state.scan, state.tilt (optional), and data.patterns/pattern_mask
    # directly -- the same pattern tests/test_initialization.py uses to build a
    # minimal ReconsState without a full plan or raw-data loader.
    args: PostInitArgs = {
        'data': ...,   # Patterns with one all-zero pattern among several nonzero ones
        'state': ...,  # ReconsState whose scan has one entry per pattern
        'dtype': numpy.float32,
        'seed': None,
        'xp': numpy,
    }
    data, state = drop_zero_patterns(args, {})
    assert numpy.all(numpy.sum(data.patterns, axis=(-1, -2)) > 0)
    assert state.scan.shape[0] == data.patterns.shape[0]
```

This test skeleton was not run (see the
[verification-pending note](#minimal-custom-implementation) above). For a hook that changes
pattern *values* rather than dropping positions (as `diffraction_align` does), also assert
the returned centroid (recomputed the same way) is closer to the origin than the input's
was.

## Maintainer sources

- `phaser/hooks/__init__.py`
- `phaser/hooks/preprocessing.py`
- `phaser/hooks/hook.py`
- `phaser/execute.py`
- `phaser/state.py`
- `phaser/utils/misc.py`
- `phaser/utils/image.py`
