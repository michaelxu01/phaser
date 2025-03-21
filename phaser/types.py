import typing as t

import numpy
import pane

if t.TYPE_CHECKING:
    from phaser.hooks import FlagArgs
    from phaser.plan import FlagLike


BackendName: t.TypeAlias = t.Literal['cuda', 'cupy', 'jax', 'cpu', 'numpy']
ReconsVar: t.TypeAlias = t.Literal['object', 'probe', 'scan']


class Cancelled(BaseException):
    ...


class Dataclass(pane.PaneBase, kw_only=True, allow_extra=False):
    ...


class SliceList(Dataclass):
    thicknesses: t.List[float]

class SliceStep(Dataclass):
    n: int
    slice_thickness: float

    @property
    def zs(self) -> t.List[float]:
        return [float(z) for z in numpy.arange(self.n) * self.slice_thickness]

    @property
    def thicknesses(self) -> t.List[float]:
        return [self.slice_thickness] * self.n

class SliceTotal(Dataclass):
    n: int
    total_thickness: float

    @property
    def zs(self) -> t.List[float]:
        return [float(z) for z in numpy.arange(self.n) * self.total_thickness / self.n]

    @property
    def thicknesses(self) -> t.List[float]:
        return [self.total_thickness / self.n] * self.n

Slices: t.TypeAlias = t.Union[SliceList, SliceStep, SliceTotal]


class Flag(Dataclass):
    after: int = 0
    every: int = 1
    before: t.Optional[int] = None

    def any_true(self, niter: int) -> bool:
        return (
            self.after < niter
            and (self.before is None or self.before > 0)
        )

    def __call__(self, args: 'FlagArgs') -> bool:
        i = args['state'].iter.engine_iter
        return (
            (self.before is None or i < self.before)
            and i > self.after
            and (i - self.after) % self.every == 0
        )

    def resolve(self) -> t.Self:
        return self


class _ConstFlag:
    def __init__(self, val: bool):
        self.val = val

    def __call__(self, args: 'FlagArgs') -> bool:
        return self.val


def process_flag(flag: 'FlagLike') -> t.Callable[['FlagArgs'], bool]:
    if isinstance(flag, bool):
        return _ConstFlag(flag)
    else:
        return flag.resolve()


def flag_any_true(flag: t.Callable[['FlagArgs'], bool], niter: int) -> bool:
    if isinstance(flag, Flag):
        return flag.any_true(niter)
    elif isinstance(flag, _ConstFlag):
        return flag.val
    # assume flag will return true
    return True


__all__ = [
    'BackendName', 'Dataclass', 'Slices', 'Flag',
    'process_flag', 'flag_any_true',
]