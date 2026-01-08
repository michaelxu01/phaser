"""
Utilities for probe positions/scan
"""

import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from phaser.utils.tree import tree_dataclass

from .num import get_array_module, cast_array_module, NumT

## FIXME: output to Tuple? importance of array number types

@t.overload
def make_raster_scan(shape: t.Tuple[int, int], scan_step: ArrayLike,  # pyright: ignore[reportOverlappingOverload]
                     rotation: float = 0., affine: t.Union[None, ArrayLike] = None, *, dtype: NumT, xp: t.Any = None) -> t.Tuple[NDArray[NumT], NDArray[NumT], NDArray[NumT]]:
    ...

@t.overload
def make_raster_scan(shape: t.Tuple[int, int], scan_step: ArrayLike,
                     rotation: float = 0., affine: t.Union[None, ArrayLike] = None, *, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating]]:
    ...

def make_raster_scan(shape: t.Tuple[int, int], scan_step: ArrayLike,
                     rotation: float = 0., affine: t.Union[None, ArrayLike] = None, *, dtype: t.Any = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number], NDArray[numpy.number]]:
    """
    Make a raster scan, centered around the origin.

    Returns an array of shape `(n_y, n_x, 2)`, with the last dimension corresponding to `(y, x)` pairs.

    # Parameters

    - `shape`: Shape `(n_y, n_x)` of scan to create
    - `scan_step`: Scan step size `(s_y, s_x)`
    - `rotation`: Scan rotation to add (degrees CCW). Rotation is applied
      around the center of the scan.
    - `dtype`: Datatype of positions to return. Defaults to `numpy.float64`.
    - `xp`: Array module
    """
    xp2 = get_array_module(shape, scan_step) if xp is None else cast_array_module(xp)

    if dtype is None:
        dtype = numpy.float64

    # TODO actually center this around (0, 0)
    yy = xp2.arange(shape[0], dtype=dtype) - xp2.asarray(shape[0] / 2., dtype=dtype)
    xx = xp2.arange(shape[1], dtype=dtype) - xp2.asarray(shape[1] / 2., dtype=dtype)
    pts = xp2.stack(xp2.meshgrid(yy, xx, indexing='ij'), axis=-1)
    pts *= xp2.broadcast_to(xp2.asarray(scan_step, dtype=dtype), (2,))

    yy_ind = xp2.arange(shape[0], dtype=dtype)
    xx_ind = xp2.arange(shape[1], dtype=dtype)
    grid_inds = xp2.stack(xp2.meshgrid(yy_ind, xx_ind, indexing='ij'), axis=-1)
    yy_grid = grid_inds[..., 0]
    xx_grid = grid_inds[..., 1]

    if affine is not None:
        affine = xp2.asarray(affine, dtype=dtype)
        pts = (pts @ affine.T)

    if rotation != 0.:
        theta = rotation * numpy.pi/180.
        mat = xp2.asarray([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]], dtype=dtype)
        pts = (pts @ mat.T)

    return t.cast(NDArray[numpy.number], pts), t.cast(NDArray[numpy.number], yy_grid), t.cast(NDArray[numpy.number], xx_grid)

# @tree_dataclass(frozen=True, init=False)
# class RasterScanMetadata:
#     shape: NDArray[numpy.int_]
#     """Sampling shape `(n_y, n_x)`"""
#     sampling: NDArray[numpy.float64]
#     """Sample spacing `(s_y, s_x)`"""
#     corner: NDArray[numpy.float64]
#     """Corner of sampling `(y_min, x_min)`"""

#     region_min: t.Optional[NDArray[numpy.float64]]
#     region_max: t.Optional[NDArray[numpy.float64]]

#     @property
#     def min(self) -> NDArray[numpy.float64]:
#         """Minimum object pixel position (y, x). Alias for `corner`."""
#         return self.corner

#     @property
#     def max(self) -> NDArray[numpy.float64]:
#         """Maximum pixel position (y, x)."""
#         return (self.corner + (self.shape - 1) * self.sampling).astype(numpy.float64)

#     @property
#     def extent(self) -> NDArray[numpy.float64]:
#         return (self.shape * self.sampling).astype(numpy.float64)

#     def __init__(self, shape: t.Tuple[int, int], sampling: ArrayLike, corner: t.Optional[ArrayLike] = None,
#                  region_min: t.Optional[ArrayLike] = None, region_max: t.Optional[ArrayLike] = None):
#         object.__setattr__(self, 'shape', numpy.broadcast_to(as_numpy(shape).astype(numpy.int_), (2,)))
#         object.__setattr__(self, 'sampling', numpy.broadcast_to(as_numpy(sampling).astype(numpy.float64), (2,)))
#         object.__setattr__(self, 'region_min', numpy.broadcast_to(as_numpy(region_min).astype(numpy.float64), (2,)) if region_min is not None else None)
#         object.__setattr__(self, 'region_max', numpy.broadcast_to(as_numpy(region_max).astype(numpy.float64), (2,)) if region_max is not None else None)

#         if corner is None:
#             corner = -self.extent / 2. + self.sampling/2. #* (self.shape % 2)
#         else:
#             corner = numpy.broadcast_to(as_numpy(corner).astype(numpy.float64), (2,))

#         object.__setattr__(self, 'corner', corner)

#     def __eq__(self, other: t.Any) -> bool:
#         if type(self) is not type(other):
#             return False
#         xp = get_array_module(self.sampling, other.sampling)
#         return (
#             xp.array_equal(self.shape, other.shape) and
#             xp.array_equal(self.sampling, other.sampling) and
#             xp.array_equal(self.corner, other.corner)
#         )

#     @staticmethod
#     def _scan_extent(scan_positions: NDArray[numpy.floating]) -> t.Tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
#         xp = get_array_module(scan_positions)
#         scan_min = numpy.array(tuple(float(xp.nanmin(scan_positions[..., i])) for i in range(2)))
#         scan_max = numpy.array(tuple(float(xp.nanmax(scan_positions[..., i])) for i in range(2)))
#         return (scan_min, scan_max)

#     @classmethod
#     def from_scan(cls: t.Type[Self], scan_positions: NDArray[numpy.floating], sampling: ArrayLike, pad: ArrayLike = 0) -> Self:
#         """Create an ObjectSampling around the given scan positions, padded by at least a radius `pad` in real-space."""
#         sampling = as_numpy(sampling).astype(numpy.float64)
#         pad = numpy.broadcast_to(pad, (2,)).astype(numpy.float64)

#         (scan_min, scan_max) = cls._scan_extent(scan_positions)
#         n = numpy.ceil((2.*pad + scan_max - scan_min) / sampling).astype(numpy.int_) + 1

#         return cls((n[0], n[1]), sampling, scan_min - pad, scan_min, scan_max)

#     def expand_to_scan(self, scan_positions: NDArray[numpy.floating], pad: ArrayLike = 0.) -> Self:
#         pad = numpy.broadcast_to(pad, (2,)).astype(numpy.float64)

#         (scan_min, scan_max) = self._scan_extent(scan_positions)
#         pad_min = numpy.ceil(numpy.maximum(0, self.min - scan_min + pad) / self.sampling).astype(numpy.int_)
#         pad_max = numpy.ceil(numpy.maximum(0, scan_max - self.max + pad) / self.sampling).astype(numpy.int_)

#         if numpy.all(pad_min == 0) and numpy.all(pad_max == 0):
#             return self

#         region_min = numpy.minimum(self.region_min, scan_min) if self.region_min is not None else None
#         region_max = numpy.maximum(self.region_max, scan_max) if self.region_max is not None else None

#         return self.__class__(
#             t.cast(t.Tuple[int, int], tuple(self.shape + pad_min + pad_max)),
#             self.sampling,
#             self.corner - pad_min * self.sampling,
#             region_min, region_max
#         )


__all__ = [
    'make_raster_scan',
    # 'RasterScanMetadata'
]