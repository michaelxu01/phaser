import functools
import itertools
from types import ModuleType
import typing as t

import numpy
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


def to_torch_dtype(dtype: t.Union[str, torch.dtype, t.Type[numpy.generic]]) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = numpy.dtype(dtype).type
    if isinstance(dtype, torch.dtype):
        return dtype

    try:
        return _NUMPY_TO_TORCH_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"Can't convert dtype '{dtype}' to a PyTorch dtype")


def to_numpy_dtype(dtype: t.Union[str, torch.dtype, t.Type[numpy.generic]]) -> t.Type[numpy.generic]:
    if isinstance(dtype, str):
        return numpy.dtype(dtype).type
    if isinstance(dtype, torch.dtype):
        return _TORCH_TO_NUMPY_DTYPE[dtype]
    return dtype


_PAD_MODE_MAP: t.Dict[_PadMode, str] = {
    'constant': 'constant',
    'edge': 'replicate',
    'reflect': 'reflect',
    'wrap': 'circular',
}


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
    'torch.pad': pad,
    'torch.indices': indices,
}, _wrap_call)

mock_torch._MockTensor = _MockTensor  # type: ignore