import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
import typing as t

import click
import numpy
from numpy.typing import NDArray
import pane
from rich.console import Console

from phaser.cli.utils import CONSOLE_THEME, natural_key
from phaser.hooks import RawDataHook


@dataclass
class Aperture:
    name: str
    inner: float
    """Inner radius (mrad, or px if uncalibrated)"""
    outer: float
    """Outer radius (mrad, or px if uncalibrated)"""


@dataclass
class Previews:
    mean_pattern: NDArray[numpy.floating]
    """Mean pattern (k_y, k_x), centered"""
    coords: NDArray[numpy.int_]
    """Sampled scan positions (n, 2), as (y, x) indices"""
    patterns: NDArray[numpy.floating]
    """Patterns at `coords` (n, k_y, k_x), centered"""
    virtual: t.List[t.Tuple[Aperture, NDArray[numpy.floating]]]
    """Virtual images (scan_y, scan_x)"""
    center: t.Tuple[float, float]
    """Center of mass of the mean pattern (y, x), in diffraction units"""
    extent: t.Tuple[float, float, float, float]
    """Matplotlib extent of the diffraction patterns"""
    units: str


def default_apertures(bf_radius: float) -> t.List[Aperture]:
    return [
        Aperture('Center pixel', 0., 0.),
        Aperture('BF', 0., bf_radius),
        Aperture('ADF', 1.2 * bf_radius, numpy.inf),
    ]


def compute_previews(
    patterns: NDArray[numpy.floating],
    mask: NDArray[numpy.floating],
    k_step: float,
    apertures: t.Optional[t.Sequence[Aperture]] = None,
    grid: t.Sequence[float] = (0.25, 0.75),
    units: str = 'mrad',
    conv_angle: t.Optional[float] = None,
) -> Previews:
    """
    Compute preview images from a 4D dataset.

    `patterns` and `mask` are fftshifted (zero frequency in the corner), as returned by the raw data hooks.
    `k_step` is the diffraction pixel size, in the units of `apertures`.

    Apertures are centered on the center of mass of the mean pattern. Radii smaller than a pixel
    select the nearest pixel, and an infinite outer radius extends to the edge of the detector.
    If `apertures` isn't specified, defaults to center pixel, BF, and ADF, with the BF radius given
    by `conv_angle` (or estimated from the mean pattern).
    """
    if not all(0. <= f <= 1. for f in grid):
        raise ValueError("Grid fractions must be between 0 and 1")

    if patterns.ndim != 4:
        raise ValueError(f"Expected 4D patterns (scan_y, scan_x, k_y, k_x), got shape {patterns.shape}")
    (n_y, n_x, k_y, k_x) = patterns.shape

    mean_pattern = numpy.fft.fftshift(patterns.mean(axis=(0, 1)) * mask)

    # diffraction coordinates, centered
    ky = (numpy.arange(k_y) - k_y // 2) * k_step
    kx = (numpy.arange(k_x) - k_x // 2) * k_step
    kyy, kxx = numpy.meshgrid(ky, kx, indexing='ij')

    weights = numpy.clip(mean_pattern, 0., None)
    total = weights.sum()
    center = (
        (float((kyy * weights).sum() / total), float((kxx * weights).sum() / total))
        if total > 0 else (0., 0.)
    )
    k = numpy.sqrt((kyy - center[0])**2 + (kxx - center[1])**2)

    if apertures is None:
        if conv_angle is None:
            # estimate BF disc radius from its area (pixels above half max)
            vmax = float(mean_pattern.max())
            area = numpy.count_nonzero(mean_pattern > 0.5 * vmax) if vmax > 0 else 1
            conv_angle = float(numpy.sqrt(area / numpy.pi)) * k_step
        apertures = default_apertures(conv_angle)

    # radius of the largest circle inside the (unmasked) detector
    rows = numpy.nonzero(numpy.fft.fftshift(mask).any(axis=1))[0]
    cols = numpy.nonzero(numpy.fft.fftshift(mask).any(axis=0))[0]
    edge = float(min(
        center[0] - (ky[rows[0]] - k_step / 2.), (ky[rows[-1]] + k_step / 2.) - center[0],
        center[1] - (kx[cols[0]] - k_step / 2.), (kx[cols[-1]] + k_step / 2.) - center[1],
    )) if len(rows) and len(cols) else 0.

    virtual = []
    k_min = float(k.min())
    for aperture in apertures:
        aperture = Aperture(aperture.name, aperture.inner, min(aperture.outer, edge))
        # aperture applied in the fftshifted frame, so patterns aren't copied
        aperture_mask = numpy.fft.ifftshift((k >= aperture.inner) & (k <= max(aperture.outer, k_min))) * mask
        img = numpy.tensordot(patterns, aperture_mask.astype(patterns.dtype), axes=((-2, -1), (0, 1)))
        virtual.append((aperture, img))

    ys = numpy.around(numpy.array(grid) * (n_y - 1)).astype(int)
    xs = numpy.around(numpy.array(grid) * (n_x - 1)).astype(int)
    coords = numpy.array([(y, x) for y in ys for x in xs])
    sampled = numpy.fft.fftshift(patterns[coords[:, 0], coords[:, 1]] * mask, axes=(-2, -1))

    extent = (
        float(kx[0] - k_step / 2.), float(kx[-1] + k_step / 2.),
        float(ky[-1] + k_step / 2.), float(ky[0] - k_step / 2.),
    )

    return Previews(mean_pattern, coords, sampled, virtual, center, extent, units)


def _pattern_norm(pattern: NDArray[numpy.floating], log: bool):
    from matplotlib.colors import LogNorm, Normalize

    vmax = float(numpy.nanmax(pattern))
    if log and vmax > 0:
        return LogNorm(vmin=vmax * 1e-4, vmax=vmax, clip=True)
    return Normalize()


def _aperture_label(aperture: Aperture, units: str) -> str:
    if aperture.inner == aperture.outer == 0.:
        return aperture.name
    return f"{aperture.name} {aperture.inner:.3g}–{aperture.outer:.3g} {units}"


def save_previews(previews: Previews, out_dir: Path, *, log: bool = False, dpi: int = 200):
    # use Figure directly (not pyplot), so figures are freed and no display is needed
    from matplotlib.figure import Figure

    out_dir.mkdir(exist_ok=True)
    units = previews.units

    def pattern_axes(ax, pattern, title):
        ax.imshow(pattern, cmap='gray', interpolation='none', norm=_pattern_norm(pattern, log), extent=previews.extent)
        ax.set_title(title, fontsize='small')
        ax.set_xlabel(f"k_x [{units}]")
        ax.set_ylabel(f"k_y [{units}]")

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot()
    pattern_axes(ax, previews.mean_pattern, 'Mean pattern')
    ax.plot([previews.center[1]], [previews.center[0]], '+r')
    fig.savefig(out_dir / 'mean_pattern.png', dpi=dpi, bbox_inches='tight')

    ys = numpy.unique(previews.coords[:, 0])
    xs = numpy.unique(previews.coords[:, 1])
    fig = Figure(figsize=(3 * len(xs), 3 * len(ys)), layout='constrained')
    axs = numpy.atleast_2d(fig.subplots(len(ys), len(xs), sharex=True, sharey=True, squeeze=False))
    for (i, ((y, x), pattern)) in enumerate(zip(previews.coords, previews.patterns)):
        ax = axs.flat[i]
        pattern_axes(ax, pattern, f"{i + 1}: (x={x}, y={y})")
        ax.label_outer()
    fig.savefig(out_dir / 'pattern_grid.png', dpi=dpi, bbox_inches='tight')

    n = len(previews.virtual)
    fig = Figure(figsize=(4 * n, 4), layout='constrained')
    axs = fig.subplots(1, n, squeeze=False)[0]
    for (ax, (aperture, img)) in zip(axs, previews.virtual):
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.set_title(_aperture_label(aperture, units), fontsize='small')
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        for (i, (y, x)) in enumerate(previews.coords):
            ax.annotate(str(i + 1), (x, y), color='red', ha='center', va='center', fontsize='small')
    fig.savefig(out_dir / 'virtual_images.png', dpi=dpi, bbox_inches='tight')


def preview_dir(meta_path: Path) -> Path:
    return meta_path.parent / 'previews'


LOADERS: t.Dict[str, str] = {
    'empad_metadata': 'empad',
    'pyMultislicer_metadata': 'empad',
    'gatan_file': 'gatan',
    'nion_metadata': 'nion',
}
"""Maps metadata `file_type` to the raw data hook used to load it"""


def read_metadata(meta_path: Path) -> t.Optional[t.Dict[str, t.Any]]:
    """Read JSON metadata, returning `None` if it isn't a known dataset type."""
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get('file_type') not in LOADERS:
        return None
    return meta


def preview_one(meta_path: Path, console: Console, *, grid: t.Sequence[float] = (0.25, 0.75), log: bool = False):
    meta = read_metadata(meta_path)
    if meta is None:
        raise ValueError(f"'{meta_path}' isn't a known metadata file (missing or unknown 'file_type').")

    calibrated = meta.get('diff_step') is not None
    if not calibrated:
        console.print("[warning]'diff_step' missing from metadata, using pixel units.")

    # diffraction pixel size in mrad.
    # placeholder of 1 mrad/px when uncalibrated, so diffraction units are pixels
    k_step = float(meta['diff_step']) if calibrated else 1.0

    # the metadata alone determines how the dataset is loaded
    hook = pane.from_data({
        'type': LOADERS[meta['file_type']], 'path': str(meta_path),
        **({} if calibrated else {'diff_step': k_step}),
    }, RawDataHook)  # type: ignore
    raw = hook(None)

    conv_angle = (raw.get('probe_hook') or {}).get('conv_angle')
    previews = compute_previews(
        raw['patterns'], raw['mask'], k_step, grid=grid,
        units='mrad' if calibrated else 'px',
        # without calibration, BF radius is estimated from the mean pattern
        conv_angle=conv_angle if calibrated else None,
    )
    out_dir = preview_dir(meta_path)
    save_previews(previews, out_dir, log=log)
    return out_dir


def parse_grid(ctx: click.Context, param: click.Parameter, value: str) -> t.List[float]:
    try:
        grid = [float(v) for v in value.split(',')]
    except ValueError:
        raise click.BadParameter(f"Expected a comma-separated list of numbers, got '{value}'")
    if not all(0. <= f <= 1. for f in grid):
        raise click.BadParameter(f"Grid fractions must be between 0 and 1, got '{value}'")
    return grid


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option('--include', type=str, multiple=True,
              help="Glob of filenames to include. If not specified, include all '.json' files")
@click.option('--exclude', type=str, multiple=True,
              help="Glob of filenames to exclude.")
@click.option('--skip-existing/--no-skip-existing', default=True,
              help="Whether to skip datasets which already have previews. Defaults to true.")
@click.option('--grid', type=str, default='0.25,0.75', callback=parse_grid,
              help="Fractional scan positions to sample patterns at (along both x and y). Defaults to '0.25,0.75'.")
@click.option('--log/--linear', default=False, help="Display patterns on a log scale. Defaults to linear.")
def preview(path: t.Union[str, Path], include: t.Sequence[str] = (), exclude: t.Sequence[str] = (),
            skip_existing: bool = True, grid: t.Sequence[float] = (0.25, 0.75), log: bool = False):
    """
    Generate preview images (mean pattern, sampled patterns, and virtual images) for datasets.

    PATH should be the path to a JSON metadata file, or to a directory
    which will be searched for JSON metadata files. The metadata's `file_type`
    determines how the dataset is loaded.

    Previews are written to a `previews` folder alongside each metadata file.
    By default, datasets which already have this folder are skipped. This behavior can be changed
    using the `--no-skip-existing` option.
    """
    console = Console(theme=CONSOLE_THEME)
    path = Path(path)

    if path.is_file():
        paths = [path]
    else:
        exclude = (*exclude, "*_calib.json", '._*')
        paths = list(path.glob('**/*.json'))
        if len(include):
            paths = list(filter(lambda path: any(fnmatch.fnmatch(path.name, pat) for pat in include), paths))
        paths = list(filter(lambda path: not any(fnmatch.fnmatch(path.name, pat) for pat in exclude), paths))

        if skip_existing:
            paths = list(filter(lambda path: not preview_dir(path).exists(), paths))

        # skip unrelated JSON files
        paths = list(filter(lambda path: read_metadata(path) is not None, paths))
        paths.sort(key=natural_key)

    console.print(f"{len(paths)} file(s) to process.")

    for meta_path in paths:
        console.print(f"Loading '{meta_path}'...")
        try:
            out_dir = preview_one(meta_path.resolve(), console, grid=grid, log=log)
        except Exception:
            console.print(f"[error]Failed to make previews for '{meta_path}'. Skipping.")
            console.print_exception()
            continue
        console.print(f"Wrote previews to '{out_dir}'.")
