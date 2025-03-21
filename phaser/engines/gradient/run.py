import logging
from functools import partial
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.solver import NoiseModel
from phaser.utils.misc import create_sparse_groupings
from phaser.utils.num import (
    cast_array_module, jit, fft2, ifft2, abs2, check_finite
)
from phaser.utils.io import OutputDir
from phaser.execute import Observer
from phaser.state import ReconsState
from phaser.hooks import EngineArgs
from phaser.plan import GradientEnginePlan
from phaser.types import process_flag, flag_any_true, ReconsVar
from ..common.output import output_images, output_state
from ..common.simulation import cutout_group, make_propagators, slice_forwards


def select_params(sim: ReconsState, params: t.AbstractSet[ReconsVar]) -> t.Dict[ReconsVar, t.Any]:
    # TODO more elegant way to do this?
    d = {}
    if 'probe' in params:
        d['probe'] = sim.probe.data
    if 'object' in params:
        d['object'] = sim.object.data
    if 'scan' in params:
        d['scan'] = sim.scan
    return d


def update_with_params(sim: ReconsState, params: t.Dict[ReconsVar, t.Any]) -> ReconsState:
    if 'probe' in params:
        sim.probe.data = params['probe']
    if 'object' in params:
        sim.object.data = params['object']
    if 'scan' in params:
        sim.scan = params['scan']
    return sim


def run_engine(args: EngineArgs, props: GradientEnginePlan) -> ReconsState:
    import jax
    import jax.numpy

    logger = logging.getLogger(__name__)

    xp = cast_array_module(jax.numpy)
    dtype = args['dtype']

    observer: Observer = args.get('observer', [])
    recons_name = args['recons_name']
    engine_i = args['engine_i']

    logger.info(f"Starting engine #{args['engine_i'] + 1}...")

    state = args['state']
    patterns = args['data'].patterns
    pattern_mask = args['data'].pattern_mask

    noise_model = props.noise_model(None)
    noise_model_state = noise_model.init_state(state)

    solvers = [s(props) for s in props.solvers]

    all_params = set()
    solver_states = []
    for solver in solvers:
        solver_states.append(
            solver.init_state(state, xp)
        )
        all_params.update(solver.params)

    #update_probe = process_flag(props.update_probe)
    #update_object = process_flag(props.update_object)
    #update_positions = process_flag(props.update_positions)
    save = process_flag(props.save)
    save_images = process_flag(props.save_images)

    grouping = props.grouping or 64

    any_output = flag_any_true(save, props.niter) or flag_any_true(save_images, props.niter)

    with OutputDir(
        props.save_options.out_dir, any_output,
        engine_i=engine_i, name=recons_name,
        group=grouping, niter=props.niter,
        noise_model=noise_model.name(),
    ) as out_dir:
        groups = create_sparse_groupings(state.scan, grouping)

        propagators = make_propagators(state)

        observer.init_solver(state, engine_i)
        observer.start_solver()

        for i in range(1, props.niter+1):
            losses = []

            for (group_i, group) in enumerate(groups):

                (loss, grad) = jax.value_and_grad(run_model)(
                    select_params(state, all_params), state,
                    group=group, props=propagators, patterns=patterns, pattern_mask=pattern_mask,
                    noise_model=noise_model, noise_model_state=noise_model_state,
                    xp=xp, dtype=dtype
                )
                # steepest descent direction
                grad = jax.tree.map(lambda v: -v.conj(), grad, is_leaf=lambda x: x is None)

                losses.append(float(loss))

                # TODO: is there a way to put this inside the jit?
                for (sol_i, solver) in enumerate(solvers):
                    (state, solver_states[sol_i]) = solver.run_group(
                        state, solver_states[sol_i], grad, i, loss
                    )

                check_finite(state.object.data, state.probe.data, context=f"object or probe, group {group_i}")

                observer.update_group(state, props.send_every_group)

            # TODO: cleanup
            for (sol_i, solver) in enumerate(solvers):
                (state, solver_states[sol_i]) = solver.run_iter(
                    state, solver_states[sol_i], grad, i, loss
                )

            loss = float(numpy.mean(losses))
            observer.update_iteration(state, i, props.niter, loss)

            state.progress.iters = numpy.concatenate([state.progress.iters, [i]])
            state.progress.detector_errors = numpy.concatenate([state.progress.detector_errors, [loss]])

            if save({'state': state, 'niter': props.niter}):
                output_state(state, out_dir, props.save_options)

            if save_images({'state': state, 'niter': props.niter}):
                output_images(state, out_dir, props.save_options)

    observer.finish_solver()
    return state


@partial(
    jit,
    static_argnames=('xp', 'dtype', 'noise_model'),
)
def run_model(
    params: t.Dict[ReconsVar, t.Any],
    sim: ReconsState,
    group: NDArray[numpy.integer],
    props: t.Optional[NDArray[numpy.complexfloating]],
    patterns: NDArray[numpy.floating],
    pattern_mask: NDArray[numpy.floating],
    noise_model: NoiseModel[t.Any],
    noise_model_state: t.Any,
    xp: t.Any,
    dtype: t.Type[numpy.floating],
) -> float:
    sim = update_with_params(sim, params)

    (ky, kx) = sim.probe.sampling.recip_grid(dtype=dtype, xp=xp)

    (probes, group_obj, group_scan) = cutout_group(ky, kx, sim, group)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            return ifft2(fft2(psi * group_obj[:, slice_i, None]) * prop)

        return psi * group_obj[:, slice_i, None]

    model_wave = fft2(slice_forwards(props, probes, sim_slice))
    model_intensity = xp.sum(abs2(model_wave), axis=1)
    group_patterns = xp.array(patterns[*group])
    # TODO: how to pass noise_model_state out?
    (loss, noise_model_state) = noise_model.calc_loss(
        model_wave, model_intensity, group_patterns, pattern_mask, noise_model_state
    )

    return loss
