import logging
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike

from phaser.utils.num import get_array_module, to_real_dtype, to_complex_dtype, fft2, ifft2, is_jax, to_numpy
from phaser.utils.misc import FloatKey, jax_dataclass
from phaser.utils.optics import fresnel_propagator, fourier_shift_filter
from phaser.state import ReconsState
from phaser.hooks.solver import NoiseModel, GroupConstraint, IterConstraint, StateT

logger = logging.getLogger(__name__)


@jax_dataclass(init=False, static_fields=('xp', 'dtype', 'noise_model', 'group_constraints', 'iter_constraints'), drop_fields=('ky', 'kx'))
class SimulationState:
    state: ReconsState
    patterns: NDArray[numpy.floating]
    pattern_mask: NDArray[numpy.floating]

    ky: NDArray[numpy.floating]
    kx: NDArray[numpy.floating]

    noise_model: NoiseModel
    group_constraints: t.Tuple[GroupConstraint[t.Any], ...]
    iter_constraints: t.Tuple[IterConstraint[t.Any], ...]

    noise_model_state: t.Any
    group_constraint_states: t.Tuple[t.Any, ...]
    iter_constraint_states: t.Tuple[t.Any, ...]

    xp: t.Any
    dtype: DTypeLike
    start_iter: int

    def __init__(
        self, *,
        state: ReconsState,
        noise_model: NoiseModel[t.Any],
        group_constraints: t.Tuple[GroupConstraint[t.Any], ...],
        iter_constraints: t.Tuple[IterConstraint[t.Any], ...],
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        xp: t.Any,
        dtype: DTypeLike,
        noise_model_state: t.Optional[t.Any] = None,
        group_constraint_states: t.Optional[t.Tuple[t.Any, ...]] = None,
        iter_constraint_states: t.Optional[t.Tuple[t.Any, ...]] = None,
        start_iter: t.Optional[int] = None,
    ):
        self.xp = xp
        self.dtype = dtype
        self.state = state
        self.patterns = patterns
        self.pattern_mask = xp.array(pattern_mask)

        self.noise_model = noise_model
        self.group_constraints = group_constraints
        self.iter_constraints = iter_constraints

        self.noise_model_state = noise_model_state or noise_model.init_state(self.state)
        self.group_constraint_states = group_constraint_states if group_constraint_states is not None else tuple(
            reg.init_state(self.state) for reg in group_constraints
        )
        self.iter_constraint_states = iter_constraint_states if iter_constraint_states is not None else tuple(
            reg.init_state(self.state) for reg in iter_constraints
        )

        self.start_iter = start_iter if start_iter is not None else self.state.iter.total_iter
        (self.ky, self.kx) = state.probe.sampling.recip_grid(dtype=dtype, xp=xp)

    def apply_group_constraints(self, group: NDArray[numpy.integer]) -> t.Self:
        def apply_reg(reg: GroupConstraint[t.Any], state: t.Any):
            (self.state, state) = reg.apply_group(group, self.state, state)
            return state

        self.group_constraint_states = tuple(
            apply_reg(reg, state) for (reg, state) in zip(self.group_constraints, self.group_constraint_states)
        )

        return self

    def apply_iter_constraints(self) -> t.Self:
        def apply_reg(reg: IterConstraint[t.Any], state: t.Any):
            (self.state, state) = reg.apply_iter(self.state, state)
            return state

        self.iter_constraint_states = tuple(
            apply_reg(reg, state) for (reg, state) in zip(self.iter_constraints, self.iter_constraint_states)
        )

        return self


def make_propagators(state: ReconsState) -> t.Optional[NDArray[numpy.complexfloating]]:
    xp = get_array_module(state.probe.data)
    dtype = to_real_dtype(state.probe.data.dtype)
    complex_dtype = to_complex_dtype(dtype)

    (ky, kx) = state.probe.sampling.recip_grid(xp=xp, dtype=dtype)

    # ignore last slice; we don't need it
    delta_zs = to_numpy(state.object.thicknesses)[:-1]
    if len(delta_zs) == 0:
        return None

    unique_zs = set(map(FloatKey, delta_zs))

    bwlim = numpy.min(state.probe.sampling.k_max) * 0.9 #* 2./3.

    k2 = ky**2 + kx**2
    bwlim_mask = k2 <= bwlim**2
    logger.info(f"Bandwidth limit: {bwlim * state.wavelength * 1e3:6.2f} mrad")

    props = {
        z: fresnel_propagator(ky, kx, state.wavelength, z).astype(complex_dtype) * bwlim_mask
        for z in unique_zs
    }

    return xp.stack(
        [props[FloatKey(z)] for z in delta_zs],
        axis = 0
    )

@t.overload
def cutout_group(
    ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],
    state: ReconsState, group: NDArray[numpy.integer],
    return_filters: t.Literal[False] = False
) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.complexfloating], NDArray[numpy.floating]]:
    ...

@t.overload
def cutout_group(
    ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],
    state: ReconsState, group: NDArray[numpy.integer],
    return_filters: t.Literal[True]
) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.complexfloating], NDArray[numpy.floating], NDArray[numpy.complexfloating]]:
    ...

def cutout_group(
    ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],
    state: ReconsState, group: NDArray[numpy.integer],
    return_filters: bool = False
):
    """Returns (probe, obj) in the cutout region"""
    probes = state.probe.data

    group_scan = state.scan[*group]
    group_obj = state.object.sampling.get_view_at_pos(state.object.data, group_scan, probes.shape[-2:])
    # group probes in real space
    # shape (len(group), 1, Ny, Nx)
    group_subpx_filters = fourier_shift_filter(ky, kx, state.object.sampling.get_subpx_shifts(group_scan, probes.shape[-2:]))[:, None, ...]
    # shape (len(group), probe modes, Ny, Nx)
    shifted_probes = ifft2(fft2(probes) * group_subpx_filters)

    if return_filters:
        return (shifted_probes, group_obj, group_scan, group_subpx_filters)

    return (shifted_probes, group_obj, group_scan)


def slice_forwards(
    props: t.Optional[NDArray[numpy.complexfloating]],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT]
) -> StateT:
    if props is None:
        return f(0, None, state)

    n_slices = len(props) + 1

    if is_jax(props):
        import jax
        state = jax.lax.fori_loop(0, n_slices - 1, lambda slice_i, state: f(slice_i, props[slice_i], state), state, unroll=False)
        return f(n_slices - 1, None, state)

    for slice_i in range(n_slices - 1):
        state = f(slice_i, props[slice_i], state)

    return f(n_slices - 1, None, state)


def slice_backwards(
    props: t.Optional[NDArray[numpy.complexfloating]],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT]
) -> StateT:
    if props is None:
        return f(0, None, state)

    n_slices = len(props) + 1

    if is_jax(props):
        import jax
        state = jax.lax.fori_loop(1, n_slices, lambda i, state: f(n_slices - i, props[n_slices - i - 1], state), state, unroll=False)
        return f(0, None, state)

    for slice_i in range(n_slices - 1, 0, -1):
        state = f(slice_i, props[slice_i - 1], state)

    return f(0, None, state)