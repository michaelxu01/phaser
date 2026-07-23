#!/usr/bin/env python
"""
Synthesize a tiny simulated 4D-STEM dataset for Phaser reconstruction smoke tests.

This script is the shared data-synthesis infrastructure for the "minimal complete
reconstruction" cookbook pages under ``docs/cookbook/reconstructions/`` (work packages
WP-5a through WP-5e). No example dataset is checked into the Phaser repository
(see checklist blocker B13), so every simulated-data cookbook page synthesizes its own
tiny dataset on demand instead of shipping a binary fixture.

WHAT IT DOES
------------
It builds a synthetic weak-phase object (a smooth "amorphous" background plus a small
grid of Gaussian "atom" bumps), a focused circular-aperture STEM probe
(`phaser.utils.optics.make_focused_probe`), and a raster scan
(`phaser.utils.scan.make_raster_scan`), then computes noiseless far-field diffraction
patterns with the same convention Phaser's own reconstruction engines use: an
orthonormal, unshifted 2D FFT of the exit wave
(`numpy.fft.fft2(exit_wave, norm='ortho')`), so the zero-frequency sample lands in the
corner of the array exactly as `phaser/state.py` documents for `Patterns.patterns`
and as the built-in raw-data loaders enforce. This lets the output be read directly by
Phaser's `manual` raw-data loader (`phaser/hooks/io/manual.py`) with `fftshifted: true`
and no further reshuffling.

The object is a **pure phase object** (unit amplitude, `exp(i * phase)`): this is a
deliberate simplification for a fast, dependency-free smoke test, not a claim about
real specimens. Scan positions are chosen to land on exact pixel offsets of the object
grid (both the scan step and the object pixel size are chosen so the ratio is an
integer), so no sub-pixel interpolation is needed when cutting out patches — this
keeps the synthesis script simple and self-contained. It does NOT affect how Phaser
itself reconstructs the data; Phaser's own forward model still performs proper
sub-pixel shifts during reconstruction.

OUTPUT
------
Running this script writes, into ``--out-dir`` (default ``examples/smoke/data``,
resolved relative to the current working directory):

- ``patterns.npy``  -- float32 array, shape ``(scan_ny, scan_nx, det_ny, det_nx)``,
  corner-origin (zero-frequency in `[0, 0]`), noiseless, each pattern summing to
  approximately 1.0 (one incident-electron equivalent; see "Complete plan" in the
  cookbook page for how a `post_load: poisson` hook scales this to a chosen dose and
  adds shot noise -- exactly as `examples/si_grad.yaml` does for real data).
- ``ground_truth.npz`` -- the synthetic object (`object_phase`, `object_amplitude`,
  shape ``(n_slices, obj_ny, obj_nx)``), the probe (`probe`, shape ``(det_ny, det_nx)``,
  complex), the scan positions (`scan`, shape ``(scan_ny, scan_nx, 2)``, in angstrom),
  and the object pixel size (`obj_pixel_size`, angstrom) and slice thicknesses
  (`thicknesses`, angstrom). This file is for comparing a reconstruction's result
  against the known input -- it is never read by Phaser itself.
- ``manifest.json`` -- every physical constant needed to write a matching Phaser plan
  (wavelength, diff_step, kv, conv_angle, defocus, scan step, slice thicknesses, ...),
  so a plan file's `raw_data`/`init` sections can be written to match this data exactly
  without recomputing anything by hand.

This dataset is intentionally NOT committed to the repository -- run this script
before validating or executing any plan that reads its output, and see
``examples/smoke/.gitignore``.

HOW SIBLING RECONSTRUCTION PAGES SHOULD REUSE THIS SCRIPT
-----------------------------------------------------------
Run from the repository root so relative paths in the companion plan YAML resolve:

    python examples/smoke/make_test_data.py

This produces the single-slice dataset used by
``docs/cookbook/reconstructions/simulated-single-slice-gradient.md``
(``examples/smoke/single_slice_gradient.yaml``).

- **Multislice gradient sibling (WP-5b):** run with ``--multislice`` (equivalent to
  ``--n-slices 2``) and point the plan's `raw_data.path` at the same `--out-dir` (use a
  distinct one, e.g. ``--out-dir examples/smoke/data_multislice``, to avoid clobbering
  the single-slice dataset if both are regenerated in the same checkout). The object
  gains a second slice (alternating atoms assigned to each slice) and `manifest.json`
  gains a `thicknesses` list of length `n_slices`; the plan's top-level `slices: {n: ...,
  total_thickness: ...}` should match `sum(thicknesses)` and `n_slices` from the
  manifest.
- **EMPAD experimental / ePIE / LSQML siblings (WP-5c-e):** these can reuse this same
  single-slice dataset (regenerate with default arguments) and simply write a different
  `engines:` section in their own plan -- the `raw_data`/`init` sections in
  ``examples/smoke/single_slice_gradient.yaml`` are engine-agnostic. Only change
  `--seed` if a sibling page wants a visibly different (but equally reproducible)
  specimen.
- All physical constants (kv, pixel size, detector shape, scan shape/step, aperture,
  defocus, seed) are defaults on the CLI below and are also echoed to
  ``manifest.json`` -- read that file rather than re-deriving the numbers, to guarantee
  the plan YAML and the synthesized data agree.

Runtime: well under a minute on CPU for the default 16x16 scan / 64x64 detector shape.
Deterministic: fixed default `--seed`; identical arguments always produce identical
output (verify with two runs and `numpy.array_equal`, as this script's report does).

# Maintainer sources
See `phaser/hooks/io/manual.py`, `phaser/utils/optics.py`, `phaser/utils/scan.py`,
`phaser/utils/physics.py`, `phaser/state.py`, and `phaser/engines/gradient/run.py`
(for the `fft2(..., shift=False)` / `abs2` forward-model convention this script mirrors).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy

from phaser.utils.physics import Electron
from phaser.utils.optics import make_focused_probe, fresnel_propagator


# ---------------------------------------------------------------------------
# Fixed physical constants shared between this script and the companion plan YAML.
# Changing any of these requires updating examples/smoke/single_slice_gradient.yaml
# to match (or regenerating documentation with the new manifest.json values).
# ---------------------------------------------------------------------------

KV = 300.0
"""Accelerating voltage [kV]."""
PIXEL_SIZE = 0.20
"""Real-space sampling of the probe/detector/object grid [angstrom/pixel]."""
DET_SHAPE = (64, 64)
"""Detector (and simulation) shape (ny, nx) [pixels]."""
SCAN_SHAPE = (16, 16)
"""Scan shape (ny, nx) [positions]."""
SCAN_STEP = 1.0
"""Scan step size [angstrom]. Chosen as an exact multiple (5x) of PIXEL_SIZE so
scan positions land on exact object-grid pixels."""
CONV_ANGLE = 20.0
"""Probe convergence semi-angle [mrad]."""
DEFOCUS = 100.0
"""Probe defocus [angstrom] (positive = overfocus)."""
SLICE_THICKNESS = 40.0
"""Per-slice thickness used when --multislice / --n-slices > 1 [angstrom]."""
OBJ_MARGIN_PX = 24
"""Extra object-grid margin beyond the scan+probe footprint [pixels]."""
SEED = 20260722
"""Default deterministic seed."""


def wavelength_angstrom(kv: float = KV) -> float:
    """Electron wavelength [angstrom] at the given accelerating voltage [kV]."""
    return Electron(kv * 1e3).wavelength


def diff_step_mrad(kv: float = KV, det_shape=DET_SHAPE, pixel_size: float = PIXEL_SIZE) -> float:
    """
    Detector angular step [mrad] implied by the chosen real-space pixel size.

    Inverts the same relation the `manual` raw-data loader uses to build its
    `Sampling` (`phaser/hooks/io/manual.py`): extent = wavelength / (diff_step * 1e-3).
    """
    extent = det_shape[0] * pixel_size
    return wavelength_angstrom(kv) * 1000.0 / extent


def make_atom_lattice(n_atoms_per_side: int, spacing_px: float, rng: numpy.random.Generator):
    """Return (y, x) pixel offsets (relative to the object center) for a small
    jittered square lattice of "atoms", representing a tiny crystalline patch."""
    idx = numpy.arange(n_atoms_per_side) - (n_atoms_per_side - 1) / 2.0
    yy, xx = numpy.meshgrid(idx * spacing_px, idx * spacing_px, indexing='ij')
    jitter = rng.uniform(-1.5, 1.5, size=(*yy.shape, 2))
    ys = (yy + jitter[..., 0]).ravel()
    xs = (xx + jitter[..., 1]).ravel()
    return ys, xs


def make_object(
    obj_shape: tuple, n_slices: int, pixel_size: float, rng: numpy.random.Generator,
    atom_amplitude: float = 0.6, atom_sigma_px: float = 1.2,
    background_sigma: float = 0.05,
) -> numpy.ndarray:
    """
    Build a synthetic weak-phase object of shape (n_slices, *obj_shape).

    Each slice is `exp(i * phase)` with unit amplitude (a pure phase object): a smooth
    "amorphous" background (low-pass-filtered random phase) plus a lattice of Gaussian
    "atom" bumps, split roughly evenly across slices when `n_slices > 1`.
    """
    ny, nx = obj_shape
    yy, xx = numpy.meshgrid(
        numpy.arange(ny) - ny // 2, numpy.arange(nx) - nx // 2, indexing='ij'
    )

    atoms_y, atoms_x = make_atom_lattice(5, spacing_px=12.0, rng=rng)
    # deterministically split atoms across slices (round-robin)
    atom_slice = numpy.arange(len(atoms_y)) % n_slices

    phase = numpy.zeros((n_slices, ny, nx), dtype=numpy.float64)
    for i, (ay, ax) in enumerate(zip(atoms_y, atoms_x)):
        s = atom_slice[i]
        phase[s] += atom_amplitude * numpy.exp(
            -((yy - ay) ** 2 + (xx - ax) ** 2) / (2 * atom_sigma_px ** 2)
        )

    # smooth "amorphous" background: low-pass-filtered white noise, same on every slice's
    # share of the specimen so every slice carries some structure.
    noise = rng.normal(0.0, 1.0, size=(ny, nx))
    freq_y = numpy.fft.fftfreq(ny)
    freq_x = numpy.fft.fftfreq(nx)
    fy, fx = numpy.meshgrid(freq_y, freq_x, indexing='ij')
    lowpass = numpy.exp(-(fy ** 2 + fx ** 2) / (2 * (2.0 / ny) ** 2))
    background = numpy.fft.ifft2(numpy.fft.fft2(noise) * lowpass).real
    background *= background_sigma / (background.std() + 1e-12)

    for s in range(n_slices):
        phase[s] += background / n_slices

    return numpy.exp(1.0j * phase)


def simulate_patterns(
    probe: numpy.ndarray, obj: numpy.ndarray, scan_px: numpy.ndarray,
    propagator: 'numpy.ndarray | None',
) -> numpy.ndarray:
    """
    Compute noiseless corner-origin diffraction patterns for every scan position.

    `probe`: complex, shape (det_ny, det_nx).
    `obj`: complex, shape (n_slices, obj_ny, obj_nx).
    `scan_px`: integer pixel offsets (relative to the object center), shape (..., 2).
    `propagator`: Fresnel propagator, shape (n_slices - 1, det_ny, det_nx), or None
    for a single slice.

    Uses the same convention as `phaser/engines/gradient/run.py`: an orthonormal,
    unshifted FFT (`norm='ortho'`, no fftshift) so the zero-frequency sample is in the
    array corner, matching `Patterns.patterns` (`phaser/state.py`).
    """
    det_ny, det_nx = probe.shape
    obj_ny, obj_nx = obj.shape[-2:]
    cy, cx = obj_ny // 2, obj_nx // 2

    flat_scan = scan_px.reshape(-1, 2)
    patterns = numpy.empty((flat_scan.shape[0], det_ny, det_nx), dtype=numpy.float32)

    for i, (py, px) in enumerate(flat_scan):
        y0, x0 = cy + int(py) - det_ny // 2, cx + int(px) - det_nx // 2
        psi = probe.copy()
        for slice_i in range(obj.shape[0]):
            patch = obj[slice_i, y0:y0 + det_ny, x0:x0 + det_nx]
            psi = psi * patch
            if propagator is not None and slice_i < obj.shape[0] - 1:
                psi = numpy.fft.ifft2(
                    numpy.fft.fft2(psi, norm='ortho') * propagator[slice_i], norm='ortho'
                )
        wave = numpy.fft.fft2(psi, norm='ortho')
        patterns[i] = numpy.abs(wave) ** 2

    return patterns.reshape(*scan_px.shape[:-1], det_ny, det_nx)


def build_dataset(
    n_slices: int = 1, seed: int = SEED,
    scan_shape=SCAN_SHAPE, det_shape=DET_SHAPE, pixel_size: float = PIXEL_SIZE,
    scan_step: float = SCAN_STEP, kv: float = KV,
    conv_angle: float = CONV_ANGLE, defocus: float = DEFOCUS,
    slice_thickness: float = SLICE_THICKNESS,
):
    """Build and return (patterns, ground_truth_dict, manifest_dict)."""
    rng = numpy.random.default_rng(seed)
    wavelength = wavelength_angstrom(kv)

    # --- probe (focused circular aperture) ---
    det_ny, det_nx = det_shape
    ky = numpy.fft.fftfreq(det_ny, pixel_size)
    kx = numpy.fft.fftfreq(det_nx, pixel_size)
    kyy, kxx = numpy.meshgrid(ky, kx, indexing='ij')
    probe = make_focused_probe(kyy, kxx, wavelength, conv_angle, defocus=defocus)

    # --- scan (raster, centered on the object) ---
    step_px = scan_step / pixel_size
    if not float(step_px).is_integer():
        raise ValueError(
            f"scan_step ({scan_step} A) must be an exact multiple of pixel_size ({pixel_size} A) "
            f"so scan positions land on integer object-grid pixels; got step_px={step_px}"
        )
    step_px = int(round(step_px))
    yy_idx = numpy.arange(scan_shape[0]) - scan_shape[0] / 2.0
    xx_idx = numpy.arange(scan_shape[1]) - scan_shape[1] / 2.0
    scan_py, scan_px_grid = numpy.meshgrid(yy_idx * step_px, xx_idx * step_px, indexing='ij')
    scan_px = numpy.stack([scan_py, scan_px_grid], axis=-1).astype(numpy.int64)
    scan_angstrom = scan_px.astype(numpy.float64) * pixel_size

    # --- object grid, sized to safely contain every scan position's cutout ---
    max_offset_px = int(numpy.max(numpy.abs(scan_px))) + max(det_ny, det_nx) // 2 + OBJ_MARGIN_PX
    obj_side = 2 * max_offset_px
    obj = make_object((obj_side, obj_side), n_slices, pixel_size, rng)

    # --- propagator (multislice only) ---
    propagator = None
    thicknesses = []
    if n_slices > 1:
        thicknesses = [slice_thickness] * n_slices
        propagator = numpy.stack([
            fresnel_propagator(kyy, kxx, wavelength, slice_thickness)
            for _ in range(n_slices - 1)
        ], axis=0)

    patterns = simulate_patterns(probe, obj, scan_px, propagator)

    ground_truth = {
        'object_phase': numpy.angle(obj).astype(numpy.float32),
        'object_amplitude': numpy.abs(obj).astype(numpy.float32),
        'probe': probe.astype(numpy.complex64),
        'scan': scan_angstrom.astype(numpy.float32),
        'obj_pixel_size': numpy.float32(pixel_size),
        'thicknesses': numpy.array(thicknesses, dtype=numpy.float32),
    }

    manifest = {
        'seed': seed,
        'kv': kv,
        'wavelength_angstrom': wavelength,
        'det_shape': list(det_shape),
        'scan_shape': list(scan_shape),
        'scan_step_angstrom': scan_step,
        'pixel_size_angstrom': pixel_size,
        'diff_step_mrad': diff_step_mrad(kv, det_shape, pixel_size),
        'conv_angle_mrad': conv_angle,
        'defocus_angstrom': defocus,
        'n_slices': n_slices,
        'slice_thickness_angstrom': slice_thickness if n_slices > 1 else None,
        'total_thickness_angstrom': slice_thickness * n_slices if n_slices > 1 else None,
        'obj_shape_px': [obj_side, obj_side],
        'mean_pattern_sum': float(numpy.mean(numpy.sum(patterns, axis=(-1, -2)))),
    }

    return patterns, ground_truth, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else None)
    parser.add_argument('--out-dir', type=Path, default=Path('examples/smoke/data'),
                         help="Output directory (default: examples/smoke/data, relative to cwd)")
    parser.add_argument('--multislice', action='store_true',
                         help="Shorthand for --n-slices 2")
    parser.add_argument('--n-slices', type=int, default=None,
                         help="Number of object slices (default: 1, or 2 if --multislice)")
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--scan-shape', type=int, nargs=2, default=list(SCAN_SHAPE))
    parser.add_argument('--det-shape', type=int, nargs=2, default=list(DET_SHAPE))
    args = parser.parse_args()

    n_slices = args.n_slices if args.n_slices is not None else (2 if args.multislice else 1)

    patterns, ground_truth, manifest = build_dataset(
        n_slices=n_slices, seed=args.seed,
        scan_shape=tuple(args.scan_shape), det_shape=tuple(args.det_shape),
    )

    # determinism check: rebuild once more and compare
    patterns2, _, _ = build_dataset(
        n_slices=n_slices, seed=args.seed,
        scan_shape=tuple(args.scan_shape), det_shape=tuple(args.det_shape),
    )
    if not numpy.array_equal(patterns, patterns2):
        raise RuntimeError("Synthesis is not deterministic for the given arguments!")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    numpy.save(args.out_dir / 'patterns.npy', patterns)
    numpy.savez(args.out_dir / 'ground_truth.npz', **ground_truth)
    with open(args.out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {patterns.shape} patterns to {args.out_dir / 'patterns.npy'}")
    print(f"Mean pattern sum (1 electron-equivalent expected): {manifest['mean_pattern_sum']:.4f}")
    print(f"Manifest: {json.dumps(manifest, indent=2)}")


if __name__ == '__main__':
    main()
