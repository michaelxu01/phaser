#!/usr/bin/env python3

from pathlib import Path
import functools
import sys
import os
import typing as t

import numpy
from numpy.typing import NDArray
from matplotlib import pyplot
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalRedisBackend
import pane

from phaser.utils.num import get_backend_module, get_array_module, to_numpy, Sampling
from phaser.utils.analysis import align_object_to_ground_truth
from phaser.utils.image import affine_transform
from phaser.utils.misc import unwrap
from phaser.plan import ReconsPlan, EnginePlan, EngineHook
from phaser.state import ReconsState, PartialReconsState, Patterns
from phaser.execute import Observer, initialize_reconstruction, prepare_for_engine
from phaser.utils.analysis import get_filtered
# prepare
# pip install "redis[hiredis]"

base_dir = Path(__file__).parent.absolute()

STUDY_NAME = "CaO_25_nv"
MEASURE_START = 20
MEASURE_EVERY = 10
PATH = '/home/gridsan/jwei/CaO_Bi/acquisition_25/acquisition_25_of20nm.json'


def calc_error_variance_band(state: t.Union[ReconsState, PartialReconsState], scalling = 1e2) -> t.Tuple[float, NDArray[numpy.floating]]:
    """
    calculating the variance, not including normalization by mean here, good for cases that mean close to zero.
    so background level is not included in optimization
    """
    object_state = unwrap(state.object).data
    xp = get_array_module(object_state)

    exclude = int(object_state.shape[0] * 0.15)
    object_state_cut_surface = object_state[exclude:-exclude]
    phase = xp.angle(object_state_cut_surface)
    nv_mid = []
    for slice in phase:
        midband = get_filtered(slice, state.object.sampling.sampling[0],
                               squared_butterworth=True, order=3.0, npad=0)
        nv_mid.append(float((midband.std())**2))

    nv_mid = xp.array(nv_mid).mean()
    error = 1 - nv_mid * scalling
    return error

def plot_diff(obj: numpy.ndarray, ground_truth: numpy.ndarray, error: float, fname: t.Union[str, Path, None] = None):
    fig, (ax1, ax2) = pyplot.subplots(ncols=2, sharex=True, sharey=True, constrained_layout=True)
    fig.set_size_inches(8, 4)

    vmin = max(numpy.nanmin(obj).astype(float), numpy.nanmin(ground_truth).astype(float))
    vmax = max(numpy.nanmax(obj).astype(float), numpy.nanmax(ground_truth).astype(float))
    ax1.imshow(obj, cmap='Reds', alpha=0.5, vmin=vmin, vmax=vmax)
    ax1.imshow(ground_truth, cmap='Blues', alpha=0.5, vmin=vmin, vmax=vmax)

    diff = obj - ground_truth
    r = max(-numpy.nanmin(diff).astype(float), numpy.nanmax(diff).astype(float))
    sm = ax2.imshow(diff, cmap='bwr', vmin=-r, vmax=r)
    fig.colorbar(sm, shrink=0.9)

    fig.suptitle(f"Error: {error:.3e}", y=0.96)

    if fname is not None:
        fig.savefig(str(fname), dpi=400)
    else:
        pyplot.show()
    pyplot.close(fig)


class OptunaObserver(Observer):
    def __init__(self, trial: t.Optional[optuna.Trial] = None):
        self.trial: t.Optional[optuna.Trial] = trial
        self.trial_path = base_dir / f"trial{trial.number:04}" if trial is not None else base_dir
        self.last_error: float = 0.0

        if self.trial is not None:
            print(f"Parameters: {self.trial.params}", flush=True)
            self.trial_path.mkdir(exist_ok=True)
            os.chdir(self.trial_path)

        super().__init__()

    def save_json(self, plan: ReconsPlan, engine: EngineHook):
        import json

        if self.trial is None:
            return
        plan_json = t.cast(dict, plan.into_data())
        plan_json['engines'] = [pane.into_data(engine, EngineHook)]  # type: ignore

        with open(self.trial_path / 'plan.json', 'w') as f:
            json.dump(plan_json, f, indent=4)

    def update_iteration(self, state: t.Union[ReconsState, PartialReconsState], i: int, n: int, error: t.Optional[float]):
        super().update_iteration(state, i, n, error)

        if i > MEASURE_START and i % MEASURE_EVERY == 0:
            error = calc_error_variance_band(state)
            self.last_error = error
            print(f"Realspace neg variance : {error:.3e}", flush=True)
            if self.trial:
                self.trial.report(error, i)

            # plot_diff(mean_obj, error, self.trial_path / f"iter{i:02}_error.png")

        if self.trial and self.trial.should_prune():
            raise optuna.TrialPruned()


# cache the initalization steps for efficiency
@functools.cache
def initialize(defocus_nm, thickness_nm) -> t.Tuple[ReconsPlan, Patterns, ReconsState]:
    plan = ReconsPlan.from_data({
        "name": "bto_grad",
        "backend": "jax",
        'dtype': 'float32',
        'raw_data': {
            'type': 'empad',
            'path': PATH,
        },
        'slices':{
            'n': round(thickness_nm),
            'total_thickness': 10 * thickness_nm,
        },
        'init':{
            'probe': {
                'type': 'focused',
                'defocus': 10 * defocus_nm
            },
        },
        'post_init': [{'type': 'drop_nans'},{'type': 'diffraction_align'}],
        'post_load':[{'type': 'crop_data', 'crop': [1,64,1,64]}],
        'engines': [],
    })
    xp = get_backend_module(plan.backend)

    (patterns, state) = initialize_reconstruction(plan, xp)

    # pad reconstruction
    new_sampling = Sampling((192, 192), extent=tuple(state.probe.sampling.extent))
    print(f"Resampling probe and patterns to shape {new_sampling.shape}...", flush=True)
    state.probe.data = state.probe.sampling.resample(state.probe.data, new_sampling)
    patterns.patterns = state.probe.sampling.resample_recip(patterns.patterns, new_sampling)
    patterns.pattern_mask = state.probe.sampling.resample_recip(patterns.pattern_mask, new_sampling)
    state.probe.sampling = new_sampling

    # store ReconsState on the cpu, we duplicate to GPU for each trial
    return (plan, patterns, state.to_numpy())


def objective(trial: optuna.Trial):
    (plan, patterns, init_state) = initialize(
        # defocus_nm = trial.suggest_int('probe_defocus_nm', 1e+1, 3e+1, log=False),
        defocus_nm = trial.suggest_categorical('probe_defocus_nm',
                                               choices=[it * 2 for it in list(range(5, 16))] ),
        thickness_nm = trial.suggest_categorical('probe_thickness_nm',
                                                 choices=[it * 5 for it in list(range(3, 8))])

    )

    xp = get_backend_module(plan.backend)

    # nesterov = trial.suggest_categorical('nesterov', ['false', 'true']) == 'true'
    nesterov = True
    all_engines = {
            'type': 'gradient',
            'probe_modes': 6,
            'sim_shape': [128,128],
            'niter': 150,
            'grouping': 64,
            'noise_model': {'type': 'poisson', 'eps': 1}, # trial.suggest_float('noise_model_eps', 1.0e-4, 1.0e+1, log=True)
            'solvers': {
                'object': {
                    'type': 'adam',
                    'learning_rate': 1.0e-3, # trial.suggest_float('obj_learning_rate', 1.0e-4, 1.0e1, log=True)
                    'nesterov': nesterov
                },
                'probe': {
                    'type': 'adam',
                    'learning_rate': 1.0e-1, # trial.suggest_float('probe_learning_rate', 1.0e-4, 1.0e1, log=True)
                    'nesterov': nesterov
                },
                'positions':{
                    'type': 'sgd',
                    'learning_rate': 1.0,
                    'momentum': 0.99,
                    'nesterov': nesterov
                }
            },
            'regularizers': [
                {'type': 'obj_tv', 'cost': 0.1}, #trial.suggest_float('obj_tv', 1.0e-5, 1.0e+5, log=True)
                {'type': 'obj_l2', 'cost': 0.2},
                {'type': 'obj_tikh', 'cost': 0.2},
                {'type': 'layers_tikh', 'cost': 1},

            ],
            'iter_constraints': [
                # 'remove_phase_ramp',
                {'type': 'clamp_object_amplitude', 'amplitude': 1.1},
                {'type': 'limit_probe_support', 'max_angle': 27},

            ],
            'group_constraints': [],
            'update_probe': {'after': 5},
            'update_object': True,
            'update_positions': {'after': 5},
            'save': {'every': 10},
            'save_images': {'every': 10},
            'save_options': {
                'images': ['probe', 'probe_recip', 'object_phase_stack', 'object_mag_sum'],
            },
             },
    # engine_1 = all_engines.copy()
    # engine_1['sim_shape'] = [256,256]
    # engine_1['grouping'] = 32
    # engine_1['regularizers'][3]['cost'] = 0.1

    engine = pane.convert(all_engines, EngineHook)

    observer = OptunaObserver(trial)
    observer.save_json(plan, engine)

    (patterns, state) = prepare_for_engine(patterns, init_state, xp, t.cast(EnginePlan, engine.props))

    state = engine({
        'data': patterns,
        'state': state,
        'dtype': patterns.patterns.dtype,
        'xp': xp,
        'recons_name': plan.name,
        'engine_i': 0,
        'observer': observer,
        'seed': None,
    })

    return observer.last_error


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} run|create|recreate|delete STORAGE_URL", file=sys.stderr)
        sys.exit(2)

    verb = sys.argv[1]
    storage = sys.argv[2]

    if storage.startswith("redis"):
        storage = JournalStorage(
            JournalRedisBackend(storage, use_cluster=False)
        )

    if verb in ('recreate', 'create'):
        from optuna.samplers import TPESampler
        from optuna.pruners import PercentilePruner

        if verb == 'recreate':
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        study = optuna.create_study(
            study_name=STUDY_NAME, storage=storage,
            # `constant_liar` ensures efficient batch optimization
            sampler=TPESampler(constant_liar=True),
            # don't prune before iteration 20, don't prune at a given step unless we have at least 8 datapoints
            pruner=PercentilePruner(50.0, n_min_trials=8, n_warmup_steps=20)
        )
    elif verb == 'delete':
        optuna.delete_study(study_name=STUDY_NAME, storage=storage)
    elif verb == 'run':
        study = optuna.load_study(
            study_name=STUDY_NAME, storage=storage
        )
        study.optimize(objective, n_trials=100)
    else:
        print(f"Unknown command '{verb}'. Expected 'run', 'create', 'recreate', or 'delete'", file=sys.stderr)
        sys.exit(2)
