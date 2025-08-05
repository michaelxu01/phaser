import functools
import itertools
import operator
from types import ModuleType
import typing as t

import numpy
from numpy.typing import ArrayLike
import torch

from phaser.utils.num import _PadMode
from phaser.utils.image import _InterpBoundaryMode


def get_cutouts(obj: torch.Tensor, start_idxs: torch.Tensor, cutout_shape: t.Tuple[int, int]) -> torch.Tensor:
    out_shape = (*start_idxs.shape[:-1], *obj.shape[:-2], *cutout_shape)

    # vmap version (broken)
    #out = torch.vmap(lambda idx: obj[..., idx[0]:idx[0]+cutout_shape[0], idx[1]:idx[1]+cutout_shape[1]])(
    #    start_idxs.reshape(-1, 2)
    #).reshape(out_shape)

    out = torch.stack([
        obj[..., i:i+cutout_shape[0], j:j+cutout_shape[1]]
        for (i, j) in start_idxs.reshape(-1, 2)
    ], dim=0).reshape(out_shape)

    return out


class _MockModule:
    def __init__(self, module: ModuleType, rewrites: t.Dict[str, t.Callable], wrap: t.Callable):
        self._inner: ModuleType = module
        self._rewrites: t.Dict[str, t.Callable] = rewrites
        self._wrap: t.Callable = wrap

        self.__name__ = module.__name__
        """
        self.__spec__ = module.__spec__
        self.__package__ = module.__package__
        self.__loader__ = module.__loader__
        self.__path__ = module.__path__
        self.__doc__ = module.__doc__
        self.__annotations__ = module.__annotations__
        if hasattr(module, '__file__') and hasattr(module, '__cached__'):
            self.__file__ = module.__file__
            self.__cached__ = module.__cached__
        """

        self.__setattr__ = lambda name, val: setattr(self._inner, name, val)

    def __getattr__(self, name: t.Any) -> t.Any:
        fullpath = f"{self.__name__}.{name}"
        if (rewrite := self._rewrites.get(fullpath, None)):
            if (val := getattr(self._inner, name, None)) is not None:
                return functools.update_wrapper(rewrite, val)
            return rewrite

        val = getattr(self._inner, name)

        if isinstance(val, ModuleType):
            return _MockModule(val, self._rewrites, self._wrap)

        if hasattr(val, '__call__') and not isinstance(val, type):
            def inner(*args, **kwargs):
                return self._wrap(val, *args, **kwargs)

            return functools.update_wrapper(inner, val)

        return val


class _MockTensor(torch.Tensor):
    #@property
    #def dtype(self) -> t.Type[numpy.generic]:
    #    return to_numpy_dtype(super().dtype)

    @property
    def T(self) -> '_MockTensor': # pyright: ignore[reportIncompatibleVariableOverride]
        if self.ndim == 2:
            return _MockTensor(super().T)
        return t.cast(_MockTensor, self.permute(*range(self.ndim - 1, -1, -1)))

    def astype(self, dtype: t.Union[str, torch.dtype, t.Type[numpy.generic]]) -> '_MockTensor':
        return t.cast(_MockTensor, self.to(to_torch_dtype(dtype)))


_TORCH_TO_NUMPY_DTYPE: t.Dict[torch.dtype, t.Type[numpy.generic]] = {
    torch.bool       : numpy.bool,
    torch.uint8      : numpy.uint8,
    torch.int8       : numpy.int8,
    torch.int16      : numpy.int16,
    torch.int32      : numpy.int32,
    torch.int64      : numpy.int64,
    torch.float16    : numpy.float16,
    torch.float32    : numpy.float32,
    torch.float64    : numpy.float64,
    torch.complex64  : numpy.complex64,
    torch.complex128 : numpy.complex128,
}

_NUMPY_TO_TORCH_DTYPE: t.Dict[t.Type[numpy.generic], torch.dtype] = {
    numpy.bool       : torch.bool,
    numpy.uint8      : torch.uint8,
    numpy.int8       : torch.int8,
    numpy.int16      : torch.int16,
    numpy.int32      : torch.int32,
    numpy.int64      : torch.int64,
    numpy.float16    : torch.float16,
    numpy.float32    : torch.float32,
    numpy.float64    : torch.float64,
    numpy.complex64  : torch.complex64,
    numpy.complex128 : torch.complex128,
}


def to_torch_dtype(dtype: t.Union[str, torch.dtype, numpy.dtype, t.Type[numpy.generic]]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, numpy.dtype):
        dtype = dtype.type
    elif not isinstance(dtype, type) or not issubclass(dtype, numpy.generic):
        dtype = numpy.dtype(dtype).type

    try:
        return _NUMPY_TO_TORCH_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"Can't convert dtype '{dtype}' to a PyTorch dtype")


def to_numpy_dtype(dtype: t.Union[str, torch.dtype, numpy.dtype, t.Type[numpy.generic]]) -> t.Type[numpy.generic]:
    if isinstance(dtype, str):
        return numpy.dtype(dtype).type
    if isinstance(dtype, numpy.dtype):
        return dtype.type
    if isinstance(dtype, torch.dtype):
        return _TORCH_TO_NUMPY_DTYPE[dtype]
    return dtype


def _mirror(idx: torch.Tensor, size: int) -> torch.Tensor:
    s = size -1
    return torch.abs((idx + s) % (2 * s) - s)


_BOUNDARY_FNS: t.Dict[str, t.Callable[[torch.Tensor, int], torch.Tensor]] = {
    'nearest': lambda idx, size: torch.clip(idx, 0, size - 1),
    'grid-wrap': lambda idx, size: idx % size,
    'reflect': lambda idx, size: torch.floor_divide(_mirror(2*idx+1, 2*size+1), 2),
    'mirror': _mirror,
}

_PAD_MODE_MAP: t.Dict[_PadMode, str] = {
    'constant': 'constant',
    'edge': 'replicate',
    'reflect': 'reflect',
    'wrap': 'circular',
}

def min(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    if axis is None:
        if keepdims:
            return torch.min(arr).reshape((1,) * arr.ndim)
        return torch.min(arr)
    return torch.amin(arr, axis, keepdim=keepdims)


def max(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    if axis is None:
        if keepdims:
            return torch.max(arr).reshape((1,) * arr.ndim)
        return torch.max(arr)
    return torch.amax(arr, axis, keepdim=keepdims)


def nanmin(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    return min(torch.nan_to_num(arr, nan=torch.inf), axis, keepdims=keepdims)


def nanmax(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    return max(torch.nan_to_num(arr, nan=-torch.inf), axis, keepdims=keepdims)


def minimum(
    x1: ArrayLike, x2: ArrayLike
) -> torch.Tensor:
    if not isinstance(x1, torch.Tensor):
        x1 = _MockTensor(torch.asarray(x1))
    if not isinstance(x2, torch.Tensor):
        x2 = _MockTensor(torch.asarray(x2))

    return torch.minimum(x1, x2)


def maximum(
    x1: ArrayLike, x2: ArrayLike
) -> torch.Tensor:
    if not isinstance(x1, torch.Tensor):
        x1 = _MockTensor(torch.asarray(x1))
    if not isinstance(x2, torch.Tensor):
        x2 = _MockTensor(torch.asarray(x2))

    return torch.maximum(x1, x2)


def split(
    arr: torch.Tensor, sections: int, *, axis: int = 0 
) -> t.Tuple[torch.Tensor, ...]:
    if arr.shape[axis] % sections != 0:
        raise ValueError("array split does not result in an equal division")
    return torch.split(arr, arr.shape[axis] // sections, axis)


def pad(
    arr: torch.Tensor, pad_width: t.Union[int, t.Tuple[int, int], t.Sequence[t.Tuple[int, int]]], /, *,
    mode: _PadMode = 'constant', cval: float = 0.
) -> torch.Tensor:
    if mode not in ('constant', 'edge', 'reflect', 'wrap'):
        raise ValueError(f"Unsupported padding mode '{mode}'")

    pad = (pad_width, pad_width) if isinstance(pad_width, int) else pad_width

    if isinstance(pad[0], int):
        pad = (pad,)

    if len(pad) == 1:
        pad = tuple(pad) * arr.ndim
    elif len(pad) != arr.ndim:
        raise ValueError(f"Invalid `pad_width` '{pad_width}'.")

    pad = tuple(itertools.chain.from_iterable(t.cast(t.Sequence[t.Tuple[int, int]], reversed(pad))))

    kwargs = {'value': cval} if mode == 'constant' else {}
    return _MockTensor(torch.nn.functional.pad(arr, pad, mode=_PAD_MODE_MAP[mode], **kwargs))


def unwrap(arr: torch.Tensor, discont: t.Optional[float] = None, axis: int = -1, *,
           period: float = 2.*numpy.pi) -> torch.Tensor:
    if discont is None:
        discont = period / 2

    diff = torch.diff(arr, dim=axis)
    dtype = torch.result_type(diff, period)

    if dtype.is_floating_point:
        interval_high = period / 2
        boundary_ambiguous = True
    else:
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0

    interval_low = -interval_high
    diffmod = torch.remainder(diff - interval_low, period) + interval_low
    if boundary_ambiguous:
        diffmod[(diffmod == interval_low) & (diff > 0)] = interval_high

    phase_correct = diffmod - diff
    phase_correct[abs(diff) < discont] = 0.

    prepend_shape = list(arr.shape)
    prepend_shape[axis] = 1
    return arr + torch.cat([torch.zeros(prepend_shape, dtype=dtype), torch.cumsum(phase_correct, axis)], dim=axis)


def indices(
    shape: t.Tuple[int, ...], dtype: t.Union[str, None, t.Type[numpy.generic], torch.dtype] = None, sparse: bool = False
) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]:
    dtype = to_torch_dtype(dtype) if dtype is not None else torch.int64

    n = len(shape)

    if sparse:
        return tuple(
            _MockTensor(torch.arange(s, dtype=dtype).reshape((1,) * i + (s,) + (1,) * (n - i - 1)))
            for (i, s) in enumerate(shape)
        )

    arrs = tuple(torch.arange(s, dtype=dtype) for s in shape)
    return _MockTensor(torch.stack(torch.meshgrid(*arrs, indexing='ij'), dim=0))


def size(arr: torch.Tensor, axis: t.Optional[int]) -> int:
    return arr.size(axis) if axis is not None else arr.numel()


def asarray(
    arr: t.Any, dtype: t.Union[str, torch.dtype, numpy.dtype, t.Type[numpy.generic], None] = None, *,
    copy: t.Optional[bool] = None,
) -> _MockTensor:
    dtype = to_torch_dtype(dtype) if dtype is not None else None
    requires_grad = arr.requires_grad if isinstance(arr, torch.Tensor) else False

    return _MockTensor(torch.asarray(arr, dtype=dtype, requires_grad=requires_grad, copy=copy))


def affine_transform(
    input: torch.Tensor, matrix: ArrayLike,
    offset: t.Optional[ArrayLike] = None,
    output_shape: t.Optional[t.Tuple[int, ...]] = None,
    order: int = 1, mode: _InterpBoundaryMode = 'grid-constant',
    cval: ArrayLike = 0.0,
) -> torch.Tensor:

    if output_shape is None:
        output_shape = input.shape
    n_axes = len(output_shape)  # num axes to transform over

    idxs = t.cast(torch.Tensor, indices(output_shape, dtype=torch.float64))

    matrix = asarray(matrix)
    if matrix.size() == (n_axes + 1, n_axes + 1):
        # homogenous transform matrix
        coords = torch.tensordot(
            matrix, torch.stack((*idxs, torch.ones_like(idxs[0])), dim=0), dims=1
        )[:-1]
    elif matrix.size() == (n_axes,):
        coords = (idxs.T * matrix + asarray(offset)).T
    else:
        raise ValueError(f"Expected matrix of shape ({n_axes + 1}, {n_axes + 1}) or ({n_axes},), instead got shape {matrix.shape}")

    return _MockTensor(torch.vmap(
        lambda a: map_coordinates(a, coords, order=order, mode=mode, cval=cval)
    )(input.reshape(-1, *input.shape[-n_axes:])).reshape((*input.shape[:-n_axes], *output_shape)))


def map_coordinates(
    arr: torch.Tensor, coordinates: torch.Tensor,
    order: int = 1, mode: _InterpBoundaryMode = 'grid-constant',
    cval: ArrayLike = 0.0
) -> torch.Tensor:
    from phaser.utils.num import to_real_dtype
    if arr.ndim != coordinates.shape[0]:
        raise ValueError("invalid shape for coordinate array")

    if order not in (0, 1):
        raise ValueError(f"Interpolation order {order} not supported (torch currently only supports order=0, 1)")

    if mode == 'grid-constant':
        return _map_coordinates_constant(
            arr, coordinates, order=order, cval=cval
        )

    remap_fn = _BOUNDARY_FNS.get(mode)
    if remap_fn is None:
        raise ValueError(f"Interpolation mode '{mode}' not supported (torch supports one of "
                         "('constant', 'nearest', 'reflect', 'mirror', 'grid-wrap'))")

    weight_dtype = to_torch_dtype(to_real_dtype(to_numpy_dtype(arr.dtype)))

    ax_nodes: t.List[t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], ...]] = []

    for ax_coords, size in zip(coordinates, arr.shape):
        if order == 1:
            lower = torch.floor(ax_coords)
            upper_weight = ax_coords - lower
            lower_idx = lower.type(torch.int32)
            ax_nodes.append((
                (remap_fn(lower_idx, size), 1.0 - upper_weight),
                (remap_fn(lower_idx + 1, size), upper_weight),
            ))
        else:
            idx = torch.round(ax_coords).type(torch.int32)
            ax_nodes.append(((remap_fn(idx, size), torch.ones((), dtype=weight_dtype)),))

    outputs = []
    for corner in itertools.product(*ax_nodes):
        idxs, weights = zip(*corner)
        outputs.append(arr[idxs] * functools.reduce(operator.mul, weights))

    result = functools.reduce(operator.add, outputs)
    return _MockTensor(result.type(arr.dtype))


def _map_coordinates_constant(
    arr: torch.Tensor, coordinates: torch.Tensor,
    order: int = 1, cval: ArrayLike = 0.0
) -> torch.Tensor:
    from phaser.utils.num import to_real_dtype
    weight_dtype = to_torch_dtype(to_real_dtype(to_numpy_dtype(arr.dtype)))
    cval = torch.tensor(cval)

    is_valid = lambda idx, size: (0 <= idx) & (idx < size)  # noqa: E731
    clip = lambda idx, size: torch.clip(idx, 0, size - 1)   # noqa: E731

    ax_nodes: t.List[t.Tuple[t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]] = []

    for ax_coords, size in zip(coordinates, arr.shape):
        if order == 1:
            lower = torch.floor(ax_coords)
            upper_weight = ax_coords - lower
            lower_idx = lower.type(torch.int32)
            ax_nodes.append((
                (clip(lower_idx, size), is_valid(lower_idx, size), 1.0 - upper_weight),
                (clip(lower_idx + 1, size), is_valid(lower_idx + 1, size), upper_weight),
            ))
        else:
            idx = torch.round(ax_coords).type(torch.int32)
            ax_nodes.append(((clip(idx, size), is_valid(idx, size), torch.ones((), dtype=weight_dtype)),))

    outputs = []
    for corner in itertools.product(*ax_nodes):
        idxs, valids, weights = zip(*corner)
        val = torch.where(functools.reduce(operator.and_, valids), arr[idxs], cval)
        outputs.append(val * functools.reduce(operator.mul, weights))

    result = functools.reduce(operator.add, outputs)
    return result.type(arr.dtype)



def _wrap_call(f, *args: t.Any, **kwargs: t.Any) -> t.Any:
    try:
        kwargs['dtype'] = to_torch_dtype(kwargs['dtype'])
    except KeyError:
        pass

    try:
        kwargs['dim'] = kwargs.pop('axes')
    except KeyError:
        try:
            kwargs['dim'] = kwargs.pop('axis')
        except KeyError:
            pass

    if f is torch.asarray and isinstance(args[0], numpy.ndarray):
        if not args[0].flags['W']:
            raise ValueError()

    result = f(*args, **kwargs)
    # TODO: deal with tuples of output, pytrees, etc. here
    # this will result in some nasty bugs
    if isinstance(result, torch.Tensor):
        return _MockTensor(result)
    return result


mock_torch = _MockModule(torch, {
    'torch.array': functools.update_wrapper(lambda *args, **kwargs: _MockTensor(_wrap_call(torch.asarray, *args, **kwargs)), torch.asarray),  # type: ignore
    'torch.asarray': asarray,
    'torch.mod': functools.update_wrapper(lambda *args, **kwargs: _MockTensor(_wrap_call(torch.remainder, *args, **kwargs)), torch.remainder),  # type: ignore
    'torch.split': split,
    'torch.pad': pad,
    'torch.min': min, 'torch.max': max,
    'torch.nanmin': nanmin, 'torch.nanmax': nanmax,
    'torch.minimum': minimum, 'torch.maximum': maximum,
    'torch.unwrap': unwrap,
    'torch.indices': indices,
    'torch.size': size,
    'torch.iscomplexobj': lambda arr: torch.is_complex(arr),
    'torch.isrealobj': lambda arr: not torch.is_complex(arr),
}, _wrap_call)

mock_torch._MockTensor = _MockTensor  # type: ignore