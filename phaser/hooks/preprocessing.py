import logging
import typing as t

from phaser.utils.num import get_array_module, to_numpy, fft2, ifft2, at
from phaser.utils.misc import create_rng, create_sparse_groupings
from phaser.utils.optics import fourier_shift_filter
from phaser.state import Patterns, ReconsState
from . import PreprocessingArgs, PoissonProps, ScaleProps, DropNanProps, ROICropProps

logger = logging.getLogger(__name__)


def scale_patterns(args: PreprocessingArgs, props: ScaleProps) -> t.Tuple[Patterns, ReconsState]:
    args['data'].patterns *= props.scale
    return (args['data'], args['state'])


def add_poisson_noise(args: PreprocessingArgs, props: PoissonProps) -> t.Tuple[Patterns, ReconsState]:
    xp = get_array_module(args['data'].patterns)
    dtype = args['dtype']

    if props.scale is not None:
        logger.info(f"Adding poisson noise to raw patterns, after scaling by {props.scale:.2e}")
        args['data'].patterns *= props.scale
    else:
        logger.info(f"Adding poisson noise to raw patterns")

    rng = create_rng(args['seed'], 'poisson_noise')

    patterns = rng.poisson(to_numpy(args['data'].patterns)).astype(dtype)

    if props.gaussian is not None:
        patterns += rng.normal(scale=props.gaussian, size=patterns.shape)

    args['data'].patterns = xp.array(patterns)
    return (args['data'], args['state'])


def drop_nan_patterns(args: PreprocessingArgs, props: DropNanProps) -> t.Tuple[Patterns, ReconsState]:
    xp = get_array_module(args['data'].patterns)

    # flatten scan and patterns
    scan = args['state'].scan.reshape(-1, 2)
    patterns = args['data'].patterns.reshape(-1, *args['data'].patterns.shape[-2:])

    fraction_nan = xp.sum(xp.isnan(args['data'].patterns), axis=(-1, -2)) / xp.prod(patterns.shape[-2:])

    mask = fraction_nan > props.threshold

    if (n := int(xp.sum(mask))):
        logger.info(f"Dropping {n}/{scan.shape[0]} patterns which are at least {props.threshold:.1%} NaN values")

        scan = scan[mask]
        patterns = patterns[mask]

    args['state'].scan = scan
    args['data'].patterns = patterns

    return (args['data'], args['state'])


def diffraction_align(args: PreprocessingArgs, props: t.Any = None) -> t.Tuple[Patterns, ReconsState]:
    patterns, state = args['data'], args['state']
    xp = get_array_module(patterns.patterns)

    mean_pattern = xp.nanmean(patterns.patterns * patterns.pattern_mask, axis=tuple(range(args['data'].patterns.ndim - 2)))
    ky, kx = state.probe.sampling.recip_grid(dtype=patterns.patterns.dtype, xp=xp)
    yy, xx = state.probe.sampling.real_grid(dtype=patterns.patterns.dtype, xp=xp)

    ky_shift = xp.nansum(ky * mean_pattern) / xp.nansum(mean_pattern)
    kx_shift = xp.nansum(kx * mean_pattern) / xp.nansum(mean_pattern)

    logging.info(f"Shifting diffraction patterns, ({kx_shift * state.probe.sampling.extent[1]}, {ky_shift * state.probe.sampling.extent[0]}) px...")
    shift = fourier_shift_filter(yy, xx, (ky_shift, kx_shift))

    for group in create_sparse_groupings(patterns.patterns.shape[:-2], 128):
        patterns.patterns = at(patterns.patterns, group).set(
            fft2(ifft2(patterns.patterns[*group]) * shift).real
        )

    # fftshift mask as well
    patterns.pattern_mask = fft2(ifft2(patterns.pattern_mask) * shift).real

    return (patterns, state)
