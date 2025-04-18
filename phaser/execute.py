import dataclasses
import logging
import time
import typing as t

import numpy
from numpy.typing import NDArray
import pane

from phaser.utils.num import cast_array_module, get_backend_module, xp_is_jax, Sampling
from phaser.utils.object import ObjectSampling
from .hooks import Hook, ObjectHook
from .plan import GradientEnginePlan, ReconsPlan, EnginePlan, ScanHook, ProbeHook
from .state import Patterns, ObjectState, ReconsState, PartialReconsState, IterState, ProgressState


_MISSING = object()


def merge(left: t.Any, right: t.Any) -> t.Any:
    print(f"merging {left} and {right}")
    def _as_dict(val) -> t.Optional[dict]:
        if isinstance(val, dict):
            return val
        if dataclasses.is_dataclass(val):
            return dataclasses.asdict(val)  # type: ignore
        if isinstance(val, pane.PaneBase):
            return val.dict()
        return None

    if left is _MISSING or right is _MISSING:
        out = left if right is _MISSING else right
        print(f"out: {out} (as single)")
        return out

    if isinstance(left, Hook) and isinstance(right, Hook):
        if left.ref != right.ref:
            return right
        d = merge(left.props or {}, right.props or {})
        d['type'] = right.type if right.type is not None else right.ref
        out = pane.from_data(d, right.__class__)
        print(f"out: {out} (as hook)")
        return out

    if (left_d := _as_dict(left)) is not None and (right_d := _as_dict(right)) is not None:
        keys = set(left_d.keys()) | set(right_d.keys())
        d = {k: merge(left_d.get(k, _MISSING), right_d.get(k, _MISSING)) for k in keys}
        print(f"out: {d} (as dict)")
        return d

    out = left if right is _MISSING else right
    print(f"out: {out}")
    return out


class Observer:
    def __init__(self):
        self.solver_start_time: t.Optional[float] = None
        self.iter_start_time: t.Optional[float] = None
        self.engine_i: int = 0
        self.start_iter: int = 0

    def _format_hhmmss(self, seconds: float) -> str:
        hh, ss = divmod(seconds, (60 * 60))
        mm, ss = divmod(ss, 60)
        return f"{int(hh):02d}:{int(mm):02d}:{ss:06.3f}"

    def _format_mmss(self, seconds: float) -> str:
        mm, ss = divmod(seconds, 60)
        return f"{int(mm):02d}:{ss:06.3f}"

    def init_solver(self, init_state: ReconsState, engine_i: int):
        self.engine_i = engine_i
        self.start_iter = init_state.iter.total_iter

        init_state.iter = IterState(self.engine_i, 1, self.start_iter + 1)

    def start_solver(self):
        logging.info("Engine initialized")
        self.iter_start_time = self.solver_start_time = time.monotonic()

    def heartbeat(self):
        pass

    def update_group(self, state: t.Union[ReconsState, PartialReconsState], force: bool = False):
        pass

    def update_iteration(self, state: t.Union[ReconsState, PartialReconsState], i: int, n: int, error: t.Optional[float] = None):
        finish_time = time.monotonic()

        if self.iter_start_time is not None:
            delta = finish_time - self.iter_start_time
            time_s = f" [{self._format_mmss(delta)}]"
        else:
            time_s = ""

        w = len(str(n))

        error_s = f" Error: {error:.3e}" if error is not None else ""
        logging.info(f"Finished iter {i:{w}}/{n}{time_s}{error_s}")

        state.iter = IterState(self.engine_i, i + 1, self.start_iter + i + 1)
        self.iter_start_time = finish_time

    def finish_solver(self):
        logging.info("Solver finished!")
        if self.solver_start_time is not None:
            finish_time = time.monotonic()
            delta = finish_time - self.solver_start_time
            logging.info(f"Total time: {self._format_hhmmss(delta)}")


def execute_plan(plan: ReconsPlan, observer: t.Optional[Observer] = None):
    xp = get_backend_module(plan.backend)

    if observer is None:
        observer = Observer()

    patterns, state = initialize_reconstruction(plan, xp, observer)
    dtype = patterns.patterns.dtype

    for (engine_i, engine) in enumerate(plan.engines):
        logging.info(f"Preparing for engine #{engine_i + 1}...")
        patterns, state = prepare_for_engine(patterns, state, xp, t.cast(EnginePlan, engine.props))
        state = engine({
            'data': patterns,
            'state': state,
            'dtype': dtype,
            'xp': xp,
            'recons_name': plan.name,
            'engine_i': engine_i,
            'observer': observer,
            'seed': None,
        })

    logging.info("Reconstruction finished!")


def initialize_reconstruction(plan: ReconsPlan, xp: t.Any, observer: Observer) -> t.Tuple[Patterns, ReconsState]:
    xp = cast_array_module(xp)

    logging.basicConfig(level=logging.INFO)

    logging.info("Executing plan...")

    seed = None
    dtype: type = numpy.float32 if plan.dtype == 'float32' else numpy.float64
    logging.info(f"dtype: {dtype} array backend: {xp.__name__}")
    if xp_is_jax(xp):
        import jax
        logging.info(f"jax backend: {jax.default_backend()} devices: {jax.devices()}")

    raw_data = plan.raw_data(None)

    wavelength = plan.wavelength or raw_data['wavelength']
    if wavelength is None:
        raise ValueError("`wavelength` must be specified by raw_data or manually")

    # merge raw data hooks with plan hooks
    init_state = PartialReconsState.read_hdf5(plan.init.state) if plan.init.state is not None else PartialReconsState()

    if init_state.wavelength is not None:
        if not numpy.isclose(init_state.wavelength, wavelength):
            logging.warning(f"Wavelength of reconstruction ('{wavelength:.3e}') doesn't match wavelength " \
                             "of previous state ('{init_state.wavelength:.3e}')")

    raw_data['scan_hook'] = merge(
        pane.from_data(t.cast(dict, raw_data['scan_hook']), ScanHook) if raw_data['scan_hook'] is not None else None,
        _MISSING if plan.init.scan is None else plan.init.scan
    )
    raw_data['probe_hook'] = merge(
        pane.from_data(t.cast(dict, raw_data['probe_hook']), ProbeHook) if raw_data['probe_hook'] is not None else None,
        _MISSING if plan.init.probe is None else plan.init.probe
    )
    print(f"scan_hook: {raw_data['scan_hook']}")
    print(f"probe_hook: {raw_data['probe_hook']}")

    if raw_data['scan_hook'] is None and init_state.scan is None:
        raise ValueError("`scan` must be specified by raw data, previous state, or manually in `init.scan`")
    if raw_data['probe_hook'] is None and init_state.probe is None:
        raise ValueError("`probe` must be specified by raw data, previous state, or manually in `init.probe`")

    raw_data['wavelength'] = wavelength
    raw_data['seed'] = seed

    # normalize pattern intensity
    #raw_data['patterns'] /= numpy.mean(numpy.sum(raw_data['patterns'], axis=(-1, -2)))
    # ensure raw data is of the correct type
    if raw_data['patterns'].dtype != dtype:
        raw_data['patterns'] = raw_data['patterns'].astype(dtype)

    # process post_load hooks:
    for p in plan.post_load:
        raw_data = p(raw_data)

    # materialize memmap
    if isinstance(raw_data['patterns'], numpy.memmap):
        raw_data['patterns'] = raw_data['patterns'].copy()

    data = Patterns(raw_data['patterns'], raw_data['mask'])

    sampling = raw_data['sampling']

    if init_state.probe is not None and plan.init.probe is None:
        logging.info("Re-using probe from initial state...")
        probe = init_state.probe
    else:
        logging.info("Initializing probe...")
        probe = t.cast(ProbeHook, raw_data['probe_hook'])(
            {'sampling': sampling, 'wavelength': wavelength, 'dtype': dtype, 'seed': seed, 'xp': xp}
        )
    if probe.data.ndim == 2:
        probe.data = probe.data.reshape((1, *probe.data.shape))

    if init_state.scan is not None and plan.init.scan is None:
        logging.info("Re-using scan from initial state...")
        scan = init_state.scan
    else:
        logging.info("Initializing scan...")
        scan = t.cast(ScanHook, raw_data['scan_hook'])(
            {'dtype': dtype, 'seed': seed, 'xp': xp}
        )
    if scan.shape[:-1] != data.patterns.shape[:-2]:
        raise ValueError(f"Scan shape {scan.shape[:-1]} doesn't match patterns shape {data.patterns.shape[:-2]}!")

    # TODO: magic numbers
    obj_sampling = ObjectSampling.from_scan(scan, sampling.sampling, sampling.extent / 2. + 20. * sampling.sampling)

    if init_state.object is not None and plan.init.object is None:
        logging.info("Re-using object from initial state...")
        obj = init_state.object
    else:
        logging.info("Initializing object...")
        obj = (plan.init.object or pane.from_data('random', ObjectHook))({
            'sampling': obj_sampling, 'slices': plan.slices, 'wavelength': wavelength,
            'dtype': dtype, 'seed': seed, 'xp': xp
        })
    if obj.data.ndim == 2:
        obj.data = obj.data.reshape((1, *obj.data.shape))
        obj.thicknesses = numpy.array([], dtype=dtype)

    state = ReconsState(
        iter=IterState(0, 0, 0),
        probe=probe,
        object=obj,
        scan=scan,
        progress=ProgressState(iters=numpy.array([]), detector_errors=numpy.array([])),
        wavelength=wavelength
    )

    # process post_init hooks
    for p in plan.post_init:
        (data, state) = p({
            'data': data, 'state': state,
            'dtype': dtype, 'seed': seed, 'xp': xp
        })

    # perform some checks on preprocessed data
    avg_pattern_intensity = float(numpy.nanmean(numpy.nansum(data.patterns, axis=(-1, -2))))

    if avg_pattern_intensity < 5.0:
        logging.warning(
            f"Mean pattern intensity very low ({avg_pattern_intensity} particles). "
            "Ensure that it is being scaled correctly to units of electrons/photons. "
            "For simulated data, use the 'scale' or 'poisson' preprocessing"
        )

    return (data, state)


def prepare_for_engine(patterns: Patterns, state: ReconsState, xp: t.Any, engine: EnginePlan) -> t.Tuple[Patterns, ReconsState]:
    state = state.to_xp(xp)

    if engine.sim_shape is not None and engine.sim_shape != state.probe.data.shape[-2:]:
        if engine.resize_method == 'pad_crop':
            new_sampling = Sampling(engine.sim_shape, extent=tuple(state.probe.sampling.extent))
        else:
            new_sampling = Sampling(engine.sim_shape, sampling=tuple(state.probe.sampling.sampling))

        logging.info(f"Resampling probe and patterns to shape {new_sampling.shape}...")
        state.probe.data = state.probe.sampling.resample(state.probe.data, new_sampling)
        # also resample patterns
        patterns.patterns = state.probe.sampling.resample_recip(patterns.patterns, new_sampling)
        # and pattern mask
        patterns.pattern_mask = state.probe.sampling.resample_recip(patterns.pattern_mask, new_sampling)

        state.probe.sampling = new_sampling

    if not numpy.allclose(state.probe.sampling.sampling, state.object.sampling.sampling):
        # resample object -> probe
        logging.info(f"Resampling object to pixel size {list(map(float, state.probe.sampling.sampling))}...")
        new_sampling = state.object.sampling.with_sampling(state.probe.sampling.sampling)
        state.object.data = state.object.sampling.resample(state.object.data, new_sampling)
        state.object.sampling = new_sampling

    if isinstance(engine, GradientEnginePlan) and not xp_is_jax(xp):
        raise ValueError("The gradient descent engine requires the jax backend.")

    current_probe_modes = state.probe.data.shape[0]
    if engine.probe_modes != current_probe_modes:
        # fix probe modes
        if engine.probe_modes < current_probe_modes:
            # TODO: redistribute intensity here
            state.probe.data = state.probe.data[:engine.probe_modes]
        else:
            from phaser.utils.optics import make_hermetian_modes
            if current_probe_modes != 1:
                logging.info("Summing probe modes (in real-space) before recreating with different # of modes")

            base_mode = xp.sum(state.probe.data, axis=0)
            state.probe.data = make_hermetian_modes(base_mode, engine.probe_modes, base_mode_power=engine.base_mode_power)

    if engine.slices is not None and (len(engine.slices.thicknesses) != len(state.object.thicknesses)
                                      or not numpy.allclose(engine.slices.thicknesses, state.object.thicknesses)):
        from phaser.utils.object import resample_slices
        logging.info(f"Reslicing object from {max(1, len(state.object.thicknesses))} to {max(1, len(engine.slices.thicknesses))} slice(s)...")
        state.object.data = resample_slices(state.object.data, state.object.thicknesses, engine.slices.thicknesses)
        state.object.thicknesses = xp.array(engine.slices.thicknesses, dtype=state.object.thicknesses.dtype)

    return patterns, state
