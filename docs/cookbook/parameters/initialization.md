# Initialization

Decide what to put in `init.scan`, `init.tilt`, `init.probe`, and `init.object` — and when
to leave one unset so a loader or prior reconstruction supplies it. The merge and reuse
mechanics live in
[Initialization merge semantics](../../architecture/lifecycle.md#raw-data-loading-and-the-initialization-merge);
this page is the decision-making view plus the initializer hooks' physical properties. Types
and defaults are in the [`InitPlan` reference](../../generated/plan/index.md#initplan).

## What each `init` field controls

| Field | Type (generated) | Built-in hooks | Governs |
| --- | --- | --- | --- |
| `init.state` | `Path \| None`, default `None` | — (not a hook) | Path to an HDF5 file to restart from — see [Output and restart](output-and-restart.md#restarting-a-reconstruction). |
| `init.scan` | `{} \| ScanHook \| None`, default `None` | [`raster`](../../generated/hooks/scan.md#raster) | Scan-position geometry: `ReconsState.scan`, shape `(..., 2)`, `(y, x)`, in length units (Å). |
| `init.tilt` | `{} \| TiltHook \| None`, default `None` | [`global`](../../generated/hooks/tilt.md#global), [`custom`](../../generated/hooks/tilt.md#custom) | Per-position beam tilt: `ReconsState.tilt`, shape `(..., 2)`, `(y, x)`, in mrad. |
| `init.probe` | `{} \| ProbeHook \| None`, default `None` | [`focused`](../../generated/hooks/probe.md#focused) | The initial probe wavefunction: `ProbeState.data`, shape `(modes, y, x)`. |
| `init.object` | `ObjectHook \| None`, default `None` (resolves to `ObjectHook('random')`) | [`random`](../../generated/hooks/object.md#random) | The initial object guess: `ObjectState.data`, shape `(z, y, x)`. |

All four are read once, at the start of a reconstruction
(`phaser.execute.initialize_reconstruction`) — never per-engine or per-iteration. Object
initialization additionally needs the first engine's `obj_pad_px` (or `5.0` if the plan has
no engines yet) to size the object's field of view from the scan extent
(`phaser/execute.py:327-330`); see [Simulation geometry](simulation-geometry.md#obj_pad_px).

## Trusting loader metadata versus supplying values yourself

Whether you write anything under `init` depends on what your
[loader](data-and-calibration.md#choosing-a-raw-data-loader) supplies:

- **Metadata loaders** (`empad` from `.json`, `gatan`, `nion`) supply `scan_hook` and, for
  `empad`/`gatan`, `probe_hook` — leave the matching `init.*` unset to accept it.
- **No-metadata loads** (`manual`, or a raw binary) supply none — you must set `init.scan`
  and `init.probe`, or the merge raises `ValueError` ("`scan` must be specified by raw data,
  previous state, or manually", `phaser/execute.py:196-199`).
- **Partial correction:** an `init.*` hook of the *same* type as the loader's metadata merges
  field by field (fix a wrong scan rotation without re-entering shape/step); a *different*
  type replaces it wholesale.

Prefer loader metadata where you have it; use an explicit `init.*` hook only for what the
loader can't supply, or the one field you're correcting.

## Built-in initializer hooks

### Scan — `raster`

Generated schema: [Scan hooks](../../generated/hooks/scan.md#raster).

| Property | Units | Notes |
| --- | --- | --- |
| `shape` | scan positions `(ny, nx)` | Required (by this hook, at the point it actually runs) if not supplied by loader metadata. |
| `step_size` | Å | A single float (isotropic step) or `(y, x)` tuple. Required if not supplied by metadata. |
| `rotation` | degrees, counterclockwise | Scan-to-detector rotation offset. |
| `affine` | dimensionless, shape `(2, 2)` | An additional linear correction applied to the raster grid (for example, a scan-coil non-orthogonality correction supplied by acquisition software). |

**Interactions:** the merged scan shape and the loaded patterns' leading shape are
reconciled by `_normalize_scan_shape` after state construction
(`phaser/execute.py:158-159`) — a scan hook and the patterns array need not match
dimensionality going in.

### Probe — `focused`

Generated schema: [Probe hooks](../../generated/hooks/probe.md#focused).

| Property | Units | Notes |
| --- | --- | --- |
| `defocus` | Å, positive is overfocus | Required if not supplied by metadata. See [Units and axis ordering](../../architecture/state-and-conventions.md#units-and-axis-ordering). |
| `conv_angle` | mrad | Semiconvergence angle of the illumination cone. Required if not supplied by metadata. |
| `aberrations` | dimensionless (each a complex or polar/Cartesian coefficient) | Sequence of named aberration coefficients (`phaser/hooks/__init__.py`); empty by default (an ideal, aberration-free probe apart from the specified defocus). |

**Interactions:** `focused_probe` builds a single-mode probe from these properties; if the
first engine's `probe_modes` is greater than `1`, `prepare_for_engine` expands it into that
many modes at the first engine boundary (see
[Engine-boundary reshaping](../../architecture/lifecycle.md#engine-boundary-reshaping) and
[Simulation geometry](simulation-geometry.md#probe_modes-and-base_mode_power)) — `init.probe`
cannot initialize a multi-mode probe directly.

### Object — `random`

Generated schema: [Object hooks](../../generated/hooks/object.md#random).

| Property | Units | Notes |
| --- | --- | --- |
| `sigma` | dimensionless (phase-noise standard deviation) | Default `1e-6` — an almost-flat, weakly perturbed random-phase starting guess. |

**Interactions:** the object's slice count comes from the top-level `slices` field
(`ReconsPlan.slices`, not from this hook's own properties) — `None` gives a single-slice
object; a `SliceList`/`SliceStep`/`SliceTotal` value adds a slice axis sized to
`len(slices.thicknesses)` (`phaser/hooks/object.py:11-18`; see
[Simulation geometry](simulation-geometry.md#slices)). The object's real-space extent comes
from the scan's spatial extent plus `obj_pad_px` (see above), not from any property here.

### Tilt — `global` and `custom`

Generated schema: [Tilt hooks](../../generated/hooks/tilt.md).

| Hook | Property | Units | Notes |
| --- | --- | --- | --- |
| `global` | `tilt` (required, shape `(2,)`) | mrad, `[ty, tx]` | Broadcasts one fixed tilt value to every scan position — for simulating or correcting a uniform beam-tilt offset. |
| `custom` | `path` (required) | — (mrad values, read from file) | Loads a per-position tilt array from a `.npy` file, shape `(ny, nx, 2)` or `(N, 2)` matching the scan. |

Unlike scan/probe, unset `init.tilt` does **not** always mean "no tilt": loader-supplied
tilt metadata would still be used (no built-in loader currently supplies any — see
[Choosing a raw-data loader](data-and-calibration.md#choosing-a-raw-data-loader)). With no
metadata and no `init.tilt`, `state.tilt` starts `None` (no correction) unless a later
gradient-engine solver targets `tilt`, which creates a zeroed tilt map at that engine
boundary (see
[Engine-boundary reshaping](../../architecture/lifecycle.md#engine-boundary-reshaping)).

## Restart: `{}` versus leaving a field unset

When `init.state` names a restart HDF5 file (see
[Output and restart](output-and-restart.md#restarting-a-reconstruction)), each of `scan`,
`tilt`, `probe`, `object` is independently either reused from the saved state or rebuilt —
never mixed within one component:

- **`init.X` unset (`None`)** — reuse from the restart file ("resume where I left off").
- **`init.X: {}`** (empty mapping; only `scan`/`tilt`/`probe` accept it) — rebuild that one
  component from loader metadata instead of the restart file, leaving the rest untouched
  (e.g. re-derive scan geometry fresh rather than keep a converged scan refinement).
- **`init.X: {type: ...}`** — replace with a fresh component from your config, ignoring both
  restart file and loader metadata.

`init.object` has no `{}` variant, so rebuilding just the object on restart needs an explicit
`ObjectHook`. Worked recipe:
[Restart, overriding a component](../recipes/restart-overriding-a-component.md).

## Minimal example

Restarting from a saved state, but re-deriving the scan from the original loader metadata
instead of reusing the restart file's scan, while everything else (probe, object, tilt) is
reused unchanged:

```yaml
init:
  state: "output/iter100.h5"
  scan: {}
```

Fully manual initialization for a loader that supplies no metadata (as used by the
[simulated single-slice gradient reconstruction](../reconstructions/simulated-single-slice-gradient.md#input-contract)):

```yaml
init:
  scan:
    type: raster
    shape: [16, 16]
    step_size: 1.0
  probe:
    type: focused
    conv_angle: 20.0
    defocus: 100.0
```

## Maintainer sources

- `phaser/execute.py`
- `phaser/plan.py`
- `phaser/hooks/scan.py`, `phaser/hooks/probe.py`, `phaser/hooks/object.py`, `phaser/hooks/tilt.py`
- `phaser/hooks/__init__.py`
- `tests/test_initialization.py`
