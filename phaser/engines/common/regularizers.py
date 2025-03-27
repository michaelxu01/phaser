from functools import partial
import logging
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import (
    get_array_module, get_scipy_module, cast_array_module,
    jit, fft2, ifft2, abs2, xp_is_jax, to_real_dtype
)
from phaser.hooks.solver import (
    GradientRegularizer, ConstraintRegularizer, ClampObjectAmplitudeProps,
    LimitProbeSupportProps, RegularizeLayersProps, ObjLowPassProps
)
from .simulation import SimulationState


class ClampObjectAmplitude(ConstraintRegularizer[None]):
    def __init__(self, args: None, props: ClampObjectAmplitudeProps):
        self.amplitude = props.amplitude

    def init_state(self, sim: SimulationState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: SimulationState, state: None) -> t.Tuple[SimulationState, None]:
        amp: float = numpy.dtype(sim.dtype).type(self.amplitude)  # type: ignore
        sim.state.object.data = clamp_amplitude(sim.state.object.data, amp)
        return (sim, None)


@partial(jit, donate_argnames=('obj',), cupy_fuse=True)
def clamp_amplitude(obj: NDArray[numpy.complexfloating], amplitude: float) -> NDArray[numpy.complexfloating]:
    xp = get_array_module(obj)

    obj_amp = xp.abs(obj)
    scale = xp.minimum(obj_amp, amplitude) / obj_amp
    return obj * scale


class LimitProbeSupport(ConstraintRegularizer[NDArray[numpy.bool_]]):
    def __init__(self, args: None, props: LimitProbeSupportProps):
        self.max_angle = props.max_angle

    def init_state(self, sim: SimulationState) -> NDArray[numpy.bool_]:
        mask = sim.kx**2 + sim.ky**2 <= (self.max_angle*1e-3 / sim.state.wavelength)**2
        return mask

    def apply_iter(self, sim: SimulationState, state: NDArray[numpy.bool_]) -> t.Tuple[SimulationState, NDArray[numpy.bool_]]:
        mask = state
        #xp = get_array_module(sim.state.probe.data)
        #print(f"intensity before: {xp.sum(abs2(sim.state.probe.data))}")
        sim.state.probe.data = ifft2(fft2(sim.state.probe.data) * mask)
        #print(f"intensity after: {xp.sum(abs2(sim.state.probe.data))}")
        return (sim, mask)


class RegularizeLayers(ConstraintRegularizer[None]):
    def __init__(self, args: None, props: RegularizeLayersProps):
        self.weight = props.weight
        self.sigma = props.sigma

    def init_state(self, sim: SimulationState) -> None:
        return None

    def apply_iter(self, sim: SimulationState, state: None) -> t.Tuple[SimulationState, None]:
        xp = get_array_module(sim.state.object.data)
        scipy = get_scipy_module(sim.state.object.data)
        dtype = to_real_dtype(sim.state.object.data)

        if len(sim.state.object.thicknesses) < 2:
            return (sim, None)

        # approximate layers as equally spaced
        layer_spacing = numpy.mean(sim.state.object.thicknesses)
        # calculate size of filter (go to ~sigma in each direction)
        r = int(numpy.ceil(2. * self.sigma / layer_spacing))
        n = 2*r + 1

        # make Gaussian filter
        zs = ((xp.arange(0, n) - (n-1)//2) * layer_spacing).astype(dtype)
        kernel = xp.exp(-(zs / self.sigma)**2 / 2.)
        kernel /= xp.sum(kernel)

        # we convolve the log of object, because the transmission
        # function is multiplicative, not additive

        if xp_is_jax(xp):
            new_obj = xp.exp(scipy.signal.convolve(
                xp.pad(xp.log(sim.state.object.data), ((r, r), (0, 0), (0, 0)), mode='edge'),
                kernel[:, None, None],
                mode="valid"
            ))
        else:
            new_obj = xp.exp(scipy.ndimage.convolve1d(xp.log(
                sim.state.object.data
            ), kernel, axis=0, mode='nearest'))

        assert new_obj.shape == sim.state.object.data.shape
        assert new_obj.dtype == sim.state.object.data.dtype
        sim.state.object.data = (
            self.weight * new_obj + (1 - self.weight) * sim.state.object.data
        )
        return (sim, None)


class ObjLowPass(ConstraintRegularizer[NDArray[numpy.bool_]]):
    def __init__(self, args: None, props: ObjLowPassProps):
        self.logger = logging.getLogger(__name__)
        self.max_freq = props.max_freq

    def init_state(self, sim: SimulationState) -> NDArray[numpy.bool_]:
        samp = sim.state.object.sampling
        xp = cast_array_module(sim.xp)

        ky = xp.fft.fftfreq(samp.shape[0], 1.0)
        kx = xp.fft.fftfreq(samp.shape[1], 1.0)
        (ky, kx) = xp.meshgrid(ky, kx, indexing='ij')
        k2 = ky**2 + kx**2

        return k2 <= self.max_freq**2

    def apply_group(
        self, group: NDArray[numpy.integer], sim: SimulationState, state: NDArray[numpy.bool_]
    ) -> t.Tuple[SimulationState, NDArray[numpy.bool_]]:
        # TODO: should this be done in-place?
        sim.state.object.data = ifft2(state * fft2(sim.state.object.data))
        return (sim, state)