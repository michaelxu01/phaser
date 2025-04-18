from pathlib import Path
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
import tifffile
from matplotlib import pyplot

from phaser.utils.num import fft2, ifft2, Sampling, get_array_module, at
from phaser.utils.object import ObjectSampling
from phaser.utils.image import remove_linear_ramp
from phaser.state import ObjectState


def _cross_correlate(x: NDArray[numpy.floating], y: NDArray[numpy.floating], max_shift: float) -> t.Tuple[float, float]:
    xp = get_array_module(x, y)

    samp = Sampling(tuple(x.shape), sampling=(1.0, 1.0))  # type: ignore
    yy, xx = samp.real_grid(xp=xp)

    cross_corr = ifft2(fft2(x) * xp.conj(fft2(y))).real
    # limit shift to corner size
    cross_corr = at(cross_corr, yy**2 + xx**2 > max_shift**2).set(numpy.nan)  # type: ignore

    max_i = xp.nanargmax(cross_corr)
    y = yy.ravel()[max_i]
    x = xx.ravel()[max_i]
    return (float(y), float(x))


def align_object_to_ground_truth(
    object: ObjectState,
    ground_truth: NDArray[numpy.floating],
    ground_truth_sampling: ArrayLike,
) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    """
    ground_truth: Ground truth phase (in radians, or radians/angstrom for multislice data)

    Returns a tuple (object, ground_truth)
    """
    xp = get_array_module(object.data, ground_truth)

    # preprocess object
    object_sampling = object.sampling
    object_phase = xp.angle(object.data)
    # normalize multislice objects to radians/angstrom
    if len(object.thicknesses):
        object_phase /= object.thicknesses[:, None, None]

    # remove linear ramp
    object_roi = object_sampling.get_region_mask(xp=xp)
    object_phase = remove_linear_ramp(object_phase, object_roi)
    object_phase -= xp.nanquantile(object_phase[..., object_roi], 0.01, axis=-1)[:, None, None]
    # and get average
    object_mean = xp.mean(object_phase, axis=0)

    # initially center ground truth on object roi
    ground_truth_corner = object.sampling.get_region_center() - numpy.array(ground_truth.shape) * ground_truth_sampling / 2.
    ground_truth_samp = ObjectSampling(
        t.cast(t.Tuple[int, int], ground_truth.shape),
        sampling=ground_truth_sampling, corner=ground_truth_corner,
        region_min=object_sampling.region_min, region_max=object_sampling.region_max,
    )

    # perform initial resampling
    upsamp_obj = object_sampling.resample(object_mean, ground_truth_samp, cval=0.)
    crop = ground_truth_samp.get_region_crop()

    # cross correlate
    max_shift = float(numpy.min(
        (ground_truth_samp.extent - object_sampling.get_region_extent()) / (3.0 * ground_truth_samp.sampling)
    ))
    if max_shift < 0:
        raise ValueError("Error: Ground truth extent smaller than object extent")
    #print(f"max_shift: {max_shift:.3f} px ({max_shift * ground_truth_samp.sampling[0]:.3f} angstrom)")

    shift = numpy.array(_cross_correlate(upsamp_obj[tuple(crop)], ground_truth[tuple(crop)], max_shift))

    # shift ground truth to match experiment
    ground_truth_samp = ObjectSampling(
        tuple(ground_truth_samp.shape),
        sampling=ground_truth_sampling, corner=ground_truth_corner + shift * ground_truth_sampling,
        region_min=object_sampling.region_min, region_max=object_sampling.region_max, 
    )

    # perform final upsampling
    upsamp_obj = object_sampling.resample(object_phase, ground_truth_samp, cval=0.)
    crop = ground_truth_samp.get_region_crop()

    return upsamp_obj[(slice(None), *crop)], ground_truth[tuple(crop)]