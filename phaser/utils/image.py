"""
Utilities for image processing & filtering
"""

import warnings
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

from .num import get_array_module, get_scipy_module, to_numpy, at, is_jax, abs2


NumT = t.TypeVar('NumT', bound=numpy.number)


@t.overload
def remove_linear_ramp(  # pyright: ignore[reportOverlappingOverload]
    data: NDArray[NumT], mask: t.Optional[NDArray[numpy.bool_]] = None
) -> NDArray[NumT]:
    ...

@t.overload
def remove_linear_ramp(
    data: ArrayLike, mask: t.Optional[NDArray[numpy.bool_]] = None
) -> NDArray[numpy.float64]:
    ...

def remove_linear_ramp(
    data: ArrayLike, mask: t.Optional[NDArray[numpy.bool_]] = None
) -> NDArray[numpy.number]:
    """
    Removes a linear 'ramp' from an image or stack of images.
    """

    xp = get_array_module(data)
    output = xp.empty_like(data)

    data = xp.array(data)

    (yy, xx) = (arr.flatten() for arr in xp.indices(data.shape[-2:], dtype=float))
    pts = xp.stack((xp.ones_like(xx), xx, yy), axis=-1)

    if mask is None:
        mask = xp.ones(len(yy), dtype=numpy.bool_)
    else:
        mask = mask.flatten()

    for idx in numpy.ndindex(data.shape[:-2]):
        layer = data[tuple(idx)].astype(numpy.float64)
        p, residues, rank, singular = xp.linalg.lstsq(pts[mask], layer.flatten()[mask], rcond=None)
        output = at(output, idx).set((layer - (p @ pts.T).reshape(layer.shape)).astype(output.dtype))

    return output


def colorize_complex(vals: ArrayLike, magnitude_only=False) -> NDArray[numpy.floating]:
    """Colorize a ndarray of complex values as rgb."""
    from matplotlib.colors import hsv_to_rgb
    xp = get_array_module(vals)

    vals = xp.asarray(vals, dtype=numpy.complexfloating)
    mag = abs2(vals)
    arg = xp.angle(vals) 
    max_mag = xp.max(mag)

    if magnitude_only:
        return mag

    h = (arg + numpy.pi) / (2*numpy.pi)
    s = 0.85 * xp.ones_like(mag)
    v = mag / max_mag 
    return hsv_to_rgb(to_numpy(xp.stack((h, s, v), axis=-1)))


def scale_to_integral_type(
    arr: NDArray[numpy.floating],
    ty: t.Literal['8bit', '16bit', '32bit', '64bit'],
    mask: t.Optional[NDArray[numpy.bool_]] = None,
    min_range: t.Optional[float] = None,
) -> NDArray[numpy.unsignedinteger]:
    xp = get_array_module(arr)

    dtype = {
        '8bit': numpy.uint8,
        '16bit': numpy.uint16,
        '32bit': numpy.uint32,
        '64bit': numpy.uint64,
    }[ty]

    imax = numpy.iinfo(dtype).max

    arr_crop = arr[..., mask] if mask is not None else arr
    # TODO: cupy doesn't support nanquantile
    vmax = xp.nanquantile(arr_crop, 0.999)
    vmin = xp.nanquantile(arr_crop, 0.001)

    if min_range is not None and (delta := min_range - (vmax - vmin)) > 0:
        # expand max and min to cover min_range
        vmax += delta/2
        vmin -= delta/2

    return (xp.clip((imax + 1) / (vmax - vmin) * (arr - vmin), 0, imax)).astype(dtype)


_BoundaryMode: t.TypeAlias = t.Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap', 'grid-mirror', 'grid-wrap', 'grid-constant']


def affine_transform(
    input: NDArray[NumT],
    matrix: ArrayLike,
    offset: t.Optional[ArrayLike] = None,
    output_shape: t.Optional[t.Tuple[int, ...]] = None,
    order: int = 1,
    mode: _BoundaryMode = 'grid-constant',
    cval: t.Union[NumT, float] = 0.0,
) -> NDArray[NumT]:
    if mode in ('constant', 'wrap'):
        # these modes aren't supported by jax
        raise ValueError(f"Resampling mode '{mode}' not supported (try 'grid-constant' or 'grid-wrap' instead)")
    if order > 1:
        raise ValueError(f"Interpolation order {order} not supported (currently only support order=0, 1)")

    xp = get_array_module(input, matrix, offset)
    scipy = get_scipy_module(input, matrix, offset)

    if is_jax(input):
        from ._jax_kernels import affine_transform, jax
        return t.cast(NDArray[NumT], affine_transform(
            t.cast(jax.Array, input), matrix, offset,
            output_shape, order, mode, cval
        ))

    if offset is None:
        offset = 0.
    if output_shape is None:
        output_shape = t.cast(t.Tuple[int, ...], input.shape)
    n_axes = len(output_shape)  # num axes to transform over

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message="The behavior of affine_transform with a 1-D array")

        output = xp.empty((*input.shape[:-n_axes], *output_shape), dtype=input.dtype)

        for idx in numpy.ndindex(input.shape[:-n_axes]):  # TODO: parallelize this on CUDA?
            scipy.ndimage.affine_transform(
                input[tuple(idx)], xp.array(matrix), offset=offset,
                output_shape=output_shape, output=output[*idx],
                order=order, mode=mode, cval=cval,
            )

        return output


__all__ = [
    'remove_linear_ramp', 'colorize_complex', 'scale_to_integral_type',
    'affine_transform',
]