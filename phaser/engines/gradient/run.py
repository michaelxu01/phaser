import logging
from functools import partial
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.solver import NoiseModel
from phaser.utils.misc import create_sparse_groupings
from phaser.utils.num import (
    get_array_module, cast_array_module, jit,
    fft2, ifft2, abs2, check_finite, at
)
from phaser.utils.optics import fourier_shift_filter
from phaser.utils.io import OutputDir
from phaser.execute import Observer
from phaser.state import ReconsState
from phaser.hooks import EngineArgs
from phaser.hooks.solver import GradientSolver, GradientSolverHook
from phaser.plan import GradientEnginePlan
from phaser.types import process_flag, flag_any_true, ReconsVar, ReconsVars
from ..common.output import output_images, output_state
from ..common.simulation import cutout_group, make_propagators, slice_forwards


logger = logging.getLogger(__name__)
_PER_ITER_VARS: t.FrozenSet[ReconsVar] = frozenset({'scan'})


def process_solvers(
    plan: GradientEnginePlan
) -> t.Tuple[t.FrozenSet[ReconsVar], t.Sequence[GradientSolver], t.FrozenSet[ReconsVar], t.Sequence[GradientSolver]]:
    # process solvers, and split into per-group and per-iter solvers
    solvers = plan.solvers

    seen = set()
    duplicate = set()

    group_vars = set()
    group_solvers = []
    iter_vars = set()
    iter_solvers = []

    for (vars, solver) in solvers.items():
        if len(vars) == 0:
            continue

        duplicate.update(vars.intersection(seen))

        if vars <= _PER_ITER_VARS:
            iter_vars |= vars
            iter_solvers.append(solver({'plan': plan, 'params': vars}))
            continue

        if len(vars & _PER_ITER_VARS):
            # TODO: is it easier to just split the solver here?
            raise ValueError(f"The same solver can't handle both per-iteration "
                             f"({', '.join(map(repr, vars & _PER_ITER_VARS))}) and per-group "
                             f"({', '.join(map(repr, vars - _PER_ITER_VARS))}) variables")

        group_vars |= vars
        group_solvers.append(solver({'plan': plan, 'params': vars}))

    if len(duplicate):
        raise ValueError(f"Duplicate solvers for variable(s) {', '.join(map(repr, duplicate))}.")

    return (
        frozenset(group_vars), tuple(group_solvers),
        frozenset(iter_vars), tuple(iter_solvers)
    )


def select_vars(state: ReconsState, vars: t.AbstractSet[ReconsVar],
                group: t.Optional[NDArray[numpy.integer]] = None) -> t.Dict[ReconsVar, t.Any]:
    # TODO more elegant way to do this?
    getters = {
        'probe': lambda sim: sim.probe.data,
        'object': lambda sim: sim.object.data,
        'scan': lambda sim: sim.scan[*group] if group is not None else sim.scan,
    }
    return {var: getters[var](state) for var in vars}


def apply_update(state: ReconsState, update: t.Dict[ReconsVar, t.Any]) -> ReconsState:
    if 'probe' in update:
        state.probe.data += update['probe']
    if 'object' in update:
        state.object.data += update['object']
    if 'scan' in update:
        xp = get_array_module(update['scan'])
        # subtract mean position update
        update['scan'] -= xp.mean(update['scan'], tuple(range(update['scan'].ndim - 1)))
        logger.info(f"Position update: mean {xp.mean(xp.linalg.norm(update['scan'], axis=-1))}")
        state.scan += update['scan']

    return state


def filter_vars(d: t.Dict[ReconsVar, t.Any], vars: t.AbstractSet[ReconsVar]) -> t.Dict[ReconsVar, t.Any]:
    return {k: v for (k, v) in d.items() if k in vars}


def run_engine(args: EngineArgs, props: GradientEnginePlan) -> ReconsState:
    import jax
    import jax.numpy
    from optax.tree_utils import tree_zeros_like
    jax.config.update('jax_traceback_filtering', 'off')

    xp = cast_array_module(jax.numpy)
    dtype = t.cast(t.Type[numpy.floating], args['dtype'])

    observer: Observer = args.get('observer', [])
    recons_name = args['recons_name']
    engine_i = args['engine_i']

    logger.info(f"Starting engine #{args['engine_i'] + 1}...")

    state = args['state']
    patterns = args['data'].patterns
    pattern_mask = args['data'].pattern_mask

    noise_model = props.noise_model(None)
    noise_model_state = noise_model.init_state(state)

    (group_vars, group_solvers, iter_vars, iter_solvers) = process_solvers(props)
    all_vars = group_vars | iter_vars

    #update_probe = process_flag(props.update_probe)
    #update_object = process_flag(props.update_object)
    #update_positions = process_flag(props.update_positions)
    save = process_flag(props.save)
    save_images = process_flag(props.save_images)

    grouping = props.grouping or 64

    any_output = flag_any_true(save, props.niter) or flag_any_true(save_images, props.niter)

    # TODO: this really needs cleanup

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

        iter_solver_states = [solver.init_state(state, xp) for solver in iter_solvers]
        group_solver_states = [solver.init_state(state, xp) for solver in group_solvers]

        #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        for i in range(1, props.niter+1):
            losses = []

            iter_grads = tree_zeros_like(select_vars(state, iter_vars))
            d = {k: v.shape for (k, v) in iter_grads.items()}

            for (group_i, group) in enumerate(groups):
                (state, loss, iter_grads, group_solver_states) = run_group(
                    state, group=group, vars=all_vars, iter_grads=iter_grads, props=propagators,
                    group_solvers=group_solvers, group_solver_states=group_solver_states,
                    patterns=patterns, pattern_mask=pattern_mask,
                    noise_model=noise_model, noise_model_state=noise_model_state,
                    xp=xp, dtype=dtype
                )

                losses.append(loss)
                check_finite(state.object.data, state.probe.data, context=f"object or probe, group {group_i}")
                observer.update_group(state, props.send_every_group)

            for (sol_i, solver) in enumerate(iter_solvers):
                (update, iter_solver_states[sol_i]) = solver.update(
                    state, iter_solver_states[sol_i], filter_vars(iter_grads, solver.params), loss
                )
                state = apply_update(state, update)

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
    static_argnames=('vars', 'xp', 'dtype', 'noise_model', 'group_solvers'),
    donate_argnames=('state', 'iter_grads', 'group_solver_states')
)
def run_group(
    state: ReconsState,
    group: NDArray[numpy.integer],
    vars: t.AbstractSet[ReconsVar],
    iter_grads: t.Dict[ReconsVar, t.Any],
    group_solvers: t.Sequence[GradientSolver[t.Any]],
    group_solver_states: t.List[t.Any],
    props: t.Optional[NDArray[numpy.complexfloating]],
    patterns: NDArray[numpy.floating],
    pattern_mask: NDArray[numpy.floating],
    noise_model: NoiseModel[t.Any],
    noise_model_state: t.Any,
    xp: t.Any,
    dtype: t.Type[numpy.floating],
) -> t.Tuple[ReconsState, float, t.Dict[ReconsVar, t.Any], t.List[t.Any]]:
    import jax
    xp = cast_array_module(xp)

    (loss, grad) = jax.value_and_grad(run_model)(
        select_vars(state, vars, group), state,
        group=group, props=props, patterns=patterns, pattern_mask=pattern_mask,
        noise_model=noise_model, noise_model_state=noise_model_state,
        xp=xp, dtype=dtype
    )
    # steepest descent direction
    grad = jax.tree.map(lambda v: -v.conj(), grad, is_leaf=lambda x: x is None)

    # update iter grads at group
    iter_grads = jax.tree.map(lambda v1, v2: at(v1, tuple(group)).set(v2), iter_grads, filter_vars(grad, vars & _PER_ITER_VARS))

    for (sol_i, solver) in enumerate(group_solvers):
        (update, group_solver_states[sol_i]) = solver.update(
            state, group_solver_states[sol_i], filter_vars(grad, solver.params), loss
        )
        state = apply_update(state, update)

    return (state, loss, iter_grads, group_solver_states)


@partial(
    jit,
    static_argnames=('xp', 'dtype', 'noise_model'),
)
def run_model(
    vars: t.Dict[ReconsVar, t.Any],
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
    # apply vars to simulation
    if 'probe' in vars:
        sim.probe.data = vars['probe']
    if 'object' in vars:
        sim.object.data = vars['object']
    group_scan = vars.get('scan', sim.scan[*group])

    (ky, kx) = sim.probe.sampling.recip_grid(dtype=dtype, xp=xp)
    probes = sim.probe.data
    group_obj = sim.object.sampling.get_view_at_pos(sim.object.data, group_scan, probes.shape[-2:])
    group_subpx_filters = fourier_shift_filter(ky, kx, sim.object.sampling.get_subpx_shifts(group_scan, probes.shape[-2:]))[:, None, ...]
    probes = ifft2(fft2(probes) * group_subpx_filters)

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
