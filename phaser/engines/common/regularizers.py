from functools import partial
import logging
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import (
    get_array_module, get_scipy_module,
    jit, fft2, ifft2, xp_is_jax, to_real_dtype
)
from phaser.state import ReconsState
from phaser.hooks.solver import (
    GroupConstraint, #IterConstraint, CostRegularizer,
    ClampObjectAmplitudeProps, LimitProbeSupportProps,
    RegularizeLayersProps, ObjLowPassProps, CostRegularizerProps
)


class ClampObjectAmplitude:
    def __init__(self, args: None, props: ClampObjectAmplitudeProps):
        self.amplitude = props.amplitude

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        amp = to_real_dtype(sim.object.data.dtype)(self.amplitude)
        sim.object.data = clamp_amplitude(sim.object.data, amp)
        return (sim, None)


@partial(jit, donate_argnames=('obj',), cupy_fuse=True)
def clamp_amplitude(obj: NDArray[numpy.complexfloating], amplitude: t.Union[float, numpy.floating]) -> NDArray[numpy.complexfloating]:
    xp = get_array_module(obj)

    obj_amp = xp.abs(obj)
    scale = xp.minimum(obj_amp, amplitude) / obj_amp
    return obj * scale


class LimitProbeSupport:
    def __init__(self, args: None, props: LimitProbeSupportProps):
        self.max_angle = props.max_angle

    def init_state(self, sim: ReconsState) -> NDArray[numpy.bool_]:
        xp = get_array_module(sim.probe.data)
        (ky, kx) = sim.probe.sampling.recip_grid(xp=xp)
        mask = kx**2 + ky**2 <= (self.max_angle*1e-3 / sim.wavelength)**2
        return mask

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: NDArray[numpy.bool_]) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: NDArray[numpy.bool_]) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        mask = state
        #xp = get_array_module(sim.state.probe.data)
        #print(f"intensity before: {xp.sum(abs2(sim.state.probe.data))}")
        sim.probe.data = ifft2(fft2(sim.probe.data) * mask)
        #print(f"intensity after: {xp.sum(abs2(sim.state.probe.data))}")
        return (sim, mask)


class RegularizeLayers:
    def __init__(self, args: None, props: RegularizeLayersProps):
        self.weight = props.weight
        self.sigma = props.sigma

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        xp = get_array_module(sim.object.data)
        scipy = get_scipy_module(sim.object.data)
        dtype = to_real_dtype(sim.object.data)

        if len(sim.object.thicknesses) < 2:
            return (sim, None)

        # approximate layers as equally spaced
        layer_spacing = numpy.mean(sim.object.thicknesses)
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
                xp.pad(xp.log(sim.object.data), ((r, r), (0, 0), (0, 0)), mode='edge'),
                kernel[:, None, None],
                mode="valid"
            ))
        else:
            new_obj = xp.exp(scipy.ndimage.convolve1d(xp.log(
                sim.object.data
            ), kernel, axis=0, mode='nearest'))

        assert new_obj.shape == sim.object.data.shape
        assert new_obj.dtype == sim.object.data.dtype
        sim.object.data = (
            self.weight * new_obj + (1 - self.weight) * sim.object.data
        )
        return (sim, None)


class ObjLowPass:
    def __init__(self, args: None, props: ObjLowPassProps):
        self.logger = logging.getLogger(__name__)
        self.max_freq = props.max_freq

    def init_state(self, sim: ReconsState) -> NDArray[numpy.bool_]:
        samp = sim.object.sampling
        xp = get_array_module(sim.object.data)

        ky = xp.fft.fftfreq(samp.shape[0], 1.0)
        kx = xp.fft.fftfreq(samp.shape[1], 1.0)
        (ky, kx) = xp.meshgrid(ky, kx, indexing='ij')
        k2 = ky**2 + kx**2

        return k2 <= self.max_freq**2

    def apply_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: NDArray[numpy.bool_]
    ) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        return self.apply_iter(sim, state)

    def apply_iter(
        self, sim: ReconsState, state: NDArray[numpy.bool_]
    ) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        # TODO: should this be done in-place?
        sim.object.data = ifft2(state * fft2(sim.object.data))
        return (sim, state)


class ObjRecipL1:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[float, None]:
        xp = get_array_module(sim.object.data)

        # l1 norm of diff. pattern amplitude
        # TODO log object before this?
        cost = xp.sum(
            xp.abs(fft2(xp.prod(sim.object.data, axis=0)))
        )
        # scale cost by fraction of the total reconstruction in the group
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)

        return (cost * cost_scale * self.cost, state)


class ObjTotalVariation:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[float, None]:
        xp = get_array_module(sim.object.data)

        cost = xp.add(*(
            xp.sum(xp.abs(xp.diff(sim.object.data, axis=ax)))
            for ax in (-1, -2)
        ))
        # scale cost by fraction of the total reconstruction in the group
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)

        return (cost * cost_scale * self.cost, state)
