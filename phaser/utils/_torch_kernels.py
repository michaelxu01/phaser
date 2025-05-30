import functools
import itertools
from types import ModuleType
import typing as t

import numpy
from numpy.typing import ArrayLike
import torch

from phaser.utils.num import _PadMode


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

    result = f(*args, **kwargs)
    # TODO: deal with tuples of output, pytrees, etc. here
    # this will result in some nasty bugs
    if isinstance(result, torch.Tensor):
        return _MockTensor(result)
    return result


mock_torch = _MockModule(torch, {
    'torch.array': functools.update_wrapper(lambda *args, **kwargs: _MockTensor(_wrap_call(torch.asarray, *args, **kwargs)), torch.asarray),  # type: ignore
    'torch.mod': functools.update_wrapper(lambda *args, **kwargs: _MockTensor(_wrap_call(torch.remainder, *args, **kwargs)), torch.remainder),  # type: ignore
    'torch.split': split,
    'torch.pad': pad,
    'torch.min': min, 'torch.max': max,
    'torch.nanmin': nanmin, 'torch.nanmax': nanmax,
    'torch.minimum': minimum, 'torch.maximum': maximum,
    'torch.unwrap': unwrap,
    'torch.indices': indices,
    'torch.size': size,
}, _wrap_call)

mock_torch._MockTensor = _MockTensor  # type: ignore