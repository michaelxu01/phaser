import logging
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.misc import create_compact_groupings, create_sparse_groupings, mask_fraction_of_groups
from phaser.utils.num import cast_array_module, to_numpy, to_complex_dtype
from phaser.utils.io import OutputDir
from phaser.execute import Observer
from phaser.hooks import EngineArgs
from phaser.hooks.solver import GroupConstraint, IterConstraint
from phaser.plan import ConventionalEnginePlan
from phaser.state import ReconsState
from phaser.types import process_flag, flag_any_true
from ..common.output import output_images, output_state
from ..common.simulation import SimulationState, make_propagators


def run_engine(args: EngineArgs, props: ConventionalEnginePlan) -> ReconsState:
    logger = logging.getLogger(__name__)

    xp = cast_array_module(args['xp'])
    dtype = args['dtype']
    observer: Observer = args.get('observer', [])
    recons_name = args['recons_name']
    engine_i = args['engine_i']

    logger.info(f"Starting engine #{args['engine_i'] + 1}...")

    noise_model = props.noise_model(None)
    group_constraints = tuple(reg(None) for reg in props.group_constraints)
    iter_constraints = tuple(reg(None) for reg in props.iter_constraints)

    update_probe = process_flag(props.update_probe)
    update_object = process_flag(props.update_object)
    update_positions = process_flag(props.update_positions)
    calc_error = process_flag(props.calc_error)
    save = process_flag(props.save)
    save_images = process_flag(props.save_images)

    grouping = props.grouping or 64

    sim = SimulationState(
        state=args['state'], noise_model=noise_model,
        group_constraints=group_constraints, iter_constraints=iter_constraints,
        patterns=args['data'].patterns, pattern_mask=args['data'].pattern_mask,
        xp=xp, dtype=dtype
    )
    assert sim.patterns.dtype == sim.dtype
    assert sim.pattern_mask.dtype == sim.dtype
    assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
    assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

    solver = props.solver(props)

    sim = solver.init(sim)

    any_output = flag_any_true(save, props.niter) or flag_any_true(save_images, props.niter)

    with OutputDir(
        props.save_options.out_dir, any_output,
        engine_i=engine_i, name=recons_name,
        group=grouping, niter=props.niter,
        solver=solver.name(),
        noise_model=noise_model.name(),
    ) as out_dir:

        if props.compact:
            groups = create_compact_groupings(sim.state.scan, grouping)
        else:
            groups = create_sparse_groupings(sim.state.scan, grouping)

        calc_error_mask = mask_fraction_of_groups(len(groups), props.calc_error_fraction)

        position_solver = None if props.position_solver is None else props.position_solver(None)
        position_solver_state = None if position_solver is None else position_solver.init_state(sim.state)

        propagators = make_propagators(sim.state)

        observer.init_solver(sim.state, engine_i)

        # runs rescaling
        sim = solver.presolve(sim, propagators, groups)

        observer.start_solver()

        for i in range(1, props.niter+1):
            iter_update_positions = update_positions({'state': sim.state, 'niter': props.niter})

            sim, pos_update, group_errors = solver.run_iteration(
                sim, propagators, groups,
                update_object=update_object({'state': sim.state, 'niter': props.niter}),
                update_probe=update_probe({'state': sim.state, 'niter': props.niter}),
                update_positions=iter_update_positions,
                calc_error=calc_error({'state': sim.state, 'niter': props.niter}),
                calc_error_mask=calc_error_mask,
                observer=observer,
            )
            assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
            assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

            sim = sim.apply_iter_constraints()

            if iter_update_positions:
                if not position_solver:
                    raise ValueError("Updating positions with no PositionSolver specified")

                # subtract mean position update
                pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
                pos_update, position_solver_state = position_solver.perform_update(sim.state.scan, pos_update, position_solver_state)
                # subtract mean again (this can change with momentum)
                pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
                update_mag = xp.linalg.norm(pos_update, axis=-1, keepdims=True)
                logger.info(f"Position update: mean {xp.mean(update_mag)}")
                sim.state.scan += pos_update
                assert sim.state.scan.dtype == sim.dtype

            error = None
            if group_errors is not None:
                error = float(to_numpy(xp.nanmean(xp.concatenate(group_errors))))

                # TODO don't do this
                sim.state.progress.iters = numpy.concatenate([sim.state.progress.iters, [i]])
                sim.state.progress.detector_errors = numpy.concatenate([sim.state.progress.detector_errors, [error]])

            observer.update_iteration(sim.state, i, props.niter, error)

            if save({'state': sim.state, 'niter': props.niter}):
                output_state(sim.state, out_dir, props.save_options)

            if save_images({'state': sim.state, 'niter': props.niter}):
                output_images(sim.state, out_dir, props.save_options)

    observer.finish_solver()
    return sim.state