import logging
import math
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import get_array_module, cast_array_module, to_numpy, fft2, ifft2, at
from phaser.utils.misc import create_rng, create_sparse_groupings
from phaser.utils.optics import fourier_shift_filter
from phaser.utils.image import affine_transform
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

    xp = cast_array_module(args['xp'])
    grouping = 128
    groups = create_sparse_groupings(patterns.patterns.shape[:-2], grouping)

    sum_pattern = xp.zeros(patterns.patterns.shape[-2:], dtype=patterns.patterns.dtype)

    for group in groups:
        pats = xp.array(patterns.patterns[*group]) * xp.array(patterns.pattern_mask)
        sum_pattern += t.cast(NDArray[numpy.floating], xp.nansum(pats, axis=tuple(range(pats.ndim - 2))))

    mean_pattern = sum_pattern / math.prod(patterns.patterns.shape[:-2])

    ky, kx = state.probe.sampling.recip_grid(dtype=patterns.patterns.dtype, xp=xp)
    #yy, xx = state.probe.sampling.real_grid(dtype=patterns.patterns.dtype, xp=xp)

    shift = xp.array([
        xp.nansum(ky * mean_pattern), xp.nansum(kx * mean_pattern)
    ]) / xp.nansum(mean_pattern)
    # 1/A -> px
    shift *= xp.array(state.probe.sampling.extent)

    logging.info(f"Shifting diffraction patterns by ({shift[1]}, {shift[0]}) px")

    def bilinear_shift(arr: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
        return to_numpy(xp.fft.ifftshift(affine_transform(
            xp.fft.fftshift(xp.array(arr), axes=(-2, -1)), [1., 1.], shift,
            output_shape=arr.shape[-2:], order=1
        ), axes=(-2, -1)))

    for group in groups:
        patterns.patterns[*group] = bilinear_shift(patterns.patterns[*group])

    # fftshift mask as well
    patterns.pattern_mask = bilinear_shift(patterns.pattern_mask)

    return (patterns, state)
