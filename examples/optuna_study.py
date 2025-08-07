#!/usr/bin/env python3

from pathlib import Path
import functools
import sys
import os
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
import tifffile
from matplotlib import pyplot
import optuna

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalRedisBackend
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner
import pane

from phaser.utils.num import get_backend_module, to_numpy, get_array_module
from phaser.utils.analysis import align_object_to_ground_truth
from phaser.utils.image import affine_transform
from phaser.utils.physics import Electron
from phaser.utils.misc import unwrap
from phaser.plan import ReconsPlan, EnginePlan, EngineHook
from phaser.state import ReconsState, PartialReconsState, Patterns, PreparedRecons, IterState, ProgressState
from phaser.execute import Observer, execute_engine, initialize_reconstruction, prepare_for_engine

base_dir = Path(__file__).parent.absolute()

STUDY_NAME = "prsco3-grad-exp4"
MEASURE_EVERY = 10
GROUND_TRUTH_PATH = base_dir / "../ground_truth_PrScO3_300kV.tif"
assert GROUND_TRUTH_PATH.exists()
ROT_ANGLE = -43.0


@functools.cache
def load_ground_truth() -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]: #units of rad/ang
    with tifffile.TiffFile(GROUND_TRUTH_PATH) as f:
        ground_truth = f.asarray()
        ground_truth_sampling: NDArray[numpy.float64] = numpy.array(f.shaped_metadata[0]['spacing'])  # type: ignore

    # rotate ground truth (assumes periodic boundary conditions)
    theta = 0.0
    if theta:
        c, s = numpy.cos(theta * numpy.pi/180.), numpy.sin(theta * numpy.pi/180.)
        rot = numpy.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        ground_truth = affine_transform(ground_truth, rot, order=1, mode='grid-wrap')

    return (ground_truth, ground_truth_sampling)


def calc_error(state: t.Union[ReconsState, PartialReconsState]) -> t.Tuple[float, NDArray[numpy.floating], NDArray[numpy.floating]]:
    object_state = unwrap(state.object)
    (ground_truth, ground_truth_sampling) = load_ground_truth()

    (upsamp_obj, ground_truth) = align_object_to_ground_truth(
        object_state, ground_truth, ground_truth_sampling,
        rotation_angle=ROT_ANGLE, refinement_niter=80
    )
    xp = get_array_module(upsamp_obj, ground_truth)

    upsamp_mean = xp.mean(upsamp_obj, axis=0)
    error = float(xp.sqrt(
        xp.nanmean((upsamp_mean - ground_truth - xp.nanmean(upsamp_mean - ground_truth))**2)
    ))
    """
    error = numpy.sqrt(sum(
        float(xp.nanmean((slice - ground_truth - xp.nanmean(slice - ground_truth))**2))
        for slice in upsamp_obj
    ) / upsamp_obj.shape[0])
    """

    return error, to_numpy(upsamp_mean - xp.nanmean(upsamp_mean - ground_truth)), to_numpy(ground_truth)


def calc_error_ssim(state: t.Union[ReconsState, PartialReconsState]) -> t.Tuple[float, NDArray[numpy.floating], NDArray[numpy.floating]]:
    from skimage.metrics import structural_similarity
    object_state = unwrap(state.object)
    (ground_truth, ground_truth_sampling) = load_ground_truth()

    (upsamp_obj, ground_truth) = map(
        to_numpy, align_object_to_ground_truth(object_state, ground_truth, ground_truth_sampling,
                                               rotation_angle=ROT_ANGLE, refinement_niter=80)
    )

    data_range = numpy.nanmax(ground_truth) - numpy.nanmin(ground_truth)
    # ssim mean

    upsamp_mean = numpy.mean(upsamp_obj, axis=0)

    ssim = float(
        structural_similarity(  # type: ignore
            upsamp_mean, ground_truth, data_range=data_range,
            gaussian_weights=True, sigma=3.0,
        )
    )
    """ssim = float(numpy.mean(tuple(
        structural_similarity(
            slice, ground_truth, data_range=data_range,
            gaussian_weights=True, sigma=3.0,
        )
        for slice in upsamp_obj
    )))"""

    error = 1.0 - ssim
    return error, upsamp_mean, ground_truth


def plot_diff(obj: numpy.ndarray, ground_truth: numpy.ndarray, error: float, fname: t.Union[str, Path, None] = None):
    fig, (ax1, ax2) = pyplot.subplots(ncols=2, sharex=True, sharey=True, constrained_layout=True)
    fig.set_size_inches(8, 4)

    #vmin = max(numpy.nanmin(obj).astype(float), numpy.nanmin(ground_truth).astype(float))
    #vmax = max(numpy.nanmax(obj).astype(float), numpy.nanmax(ground_truth).astype(float))
    ax1.imshow(obj, cmap='Reds', alpha=0.5)
    ax1.imshow(ground_truth, cmap='Blues', alpha=0.5)

    diff = obj - ground_truth
    r = max(-numpy.nanmin(diff).astype(float), numpy.nanmax(diff).astype(float))
    sm = ax2.imshow(diff, cmap='bwr', vmin=-r, vmax=r)
    fig.colorbar(sm, shrink=0.9)

    fig.suptitle(f"Error: {error:.3e}\n", y=0.96)

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

    def save_json(self, plan: ReconsPlan, engines: t.Sequence[EngineHook]):
        import json

        if self.trial is None:
            return
        plan_json = t.cast(dict, plan.into_data())
        plan_json['engines'] = pane.into_data(engines, t.Sequence[EngineHook])  # type: ignore

        with open(self.trial_path / 'plan.json', 'w') as f:
            json.dump(plan_json, f, indent=4)

    def update_iteration(self, state: ReconsState, i: int, n: int, error: t.Optional[float] = None):
        super().update_iteration(state, i, n, error)

        #i += self.start_iter

        if i % MEASURE_EVERY == 0:
            error, mean_obj, ground_truth = calc_error(state)
            self.last_error = error
            print(f"Realspace error: {error:.3e}", flush=True)
            if self.trial:
                self.trial.report(error, i)

            plot_diff(mean_obj, ground_truth, error, self.trial_path / f"iter{i:02}_error.png")

        if self.trial and self.trial.should_prune():
            raise optuna.TrialPruned()


def initialize(observer: Observer) -> t.Tuple[ReconsPlan, PreparedRecons]: # t.Tuple[ReconsPlan, Patterns, ReconsState]:
    plan = ReconsPlan.from_data({
        "name": "prsco3-grad",
        "backend": "jax",
        'dtype': 'float32',
        'raw_data': {
            'type': 'empad',
            'path': '~/lebeau_shared/ptycho_example/exp-prsco3/chen2021-PSO/PSO.json',
        },
        # "init": {
            # 'state': str(base_dir / "init.h5"),
        # },
        'post_load': [
        ],
        'post_init': [
            "drop_nans",
            "diffraction_align",
        ],
        'slices': {'n': 21, 'total_thickness': 210.0},
        'engines': [],
    })
    xp = get_backend_module(plan.backend)

    # (patterns, state) = 
    recons = initialize_reconstruction(plan=plan, xp=xp, observers=observer)

    # pad reconstruction
    # new_sampling = Sampling((192, 192), extent=tuple(state.probe.sampling.extent))
    # print(f"Resampling probe and patterns to shape {new_sampling.shape}...", flush=True)
    # state.probe.data = state.probe.sampling.resample(state.probe.data, new_sampling)
    # patterns.patterns = state.probe.sampling.resample_recip(patterns.patterns, new_sampling)
    # patterns.pattern_mask = state.probe.sampling.resample_recip(patterns.pattern_mask, new_sampling)
    # state.probe.sampling = new_sampling

    return (plan, recons) #(plan, patterns, state.to_numpy())


def objective(trial: optuna.Trial):
    # (plan, patterns, init_state) = 
    observer = OptunaObserver(trial)
    (plan, recons) = initialize(observer)
    # xp = get_backend_module(plan.backend)
    xp = get_array_module(recons.state.object.data, recons.state.probe.data)
    
    noise_model_eps = trial.suggest_float('noise_model_eps', 1.0e-1, 10.0, log=True)

    obj_lr = trial.suggest_float('obj_learning_rate', 1.0e-4, 1.0, log=True)
    probe_lr = trial.suggest_float('probe_learning_rate', 1.0e-3, 1.0, log=True)

    obj_l1 = trial.suggest_float('obj_l1', 1.0e-1, 1.0e+2, log=True)
    obj_l2 = trial.suggest_float('obj_l2', 1.0e-1, 1.0e+2, log=True)
    obj_tikh = trial.suggest_float('obj_tikh_1', 1.0e-1, 1.0e+2, log=True)
    layers_tikh = trial.suggest_float('layers_tikh', 1.0e+2, 5.0e+3, log=True)

    engine_1 = pane.convert({  # type: ignore
        'type': 'gradient',
        'niter': 150,
        'grouping': 128,
        'bwlim_frac': 0.8,
        'probe_modes': 8,
        'sim_shape': [128, 128],
        'noise_model': {
            'type': 'poisson',
            'eps': noise_model_eps,
        },
        'solvers': {
            'object': {
                'type': 'adam',
                'learning_rate': obj_lr,
                'nesterov': True,
            },
            'probe': {
                'type': 'adam',
                'learning_rate': probe_lr,
                'nesterov': True,
            },
            'positions': {
                'type': 'sgd',
                'learning_rate': 1.0,
                'momentum': 0.99,
                'nesterov': True,
            },
        },
        'regularizers': [
            {'type': 'obj_l1', 'cost': obj_l1},
            {'type': 'obj_l2', 'cost': obj_l2},
            {'type': 'obj_tikh', 'cost': obj_tikh},
            {'type': 'layers_tikh', 'cost': layers_tikh},
            #{'type': 'probe_recip_tv', 'cost': probe_recip_tv},
        ],
        'group_constraints': [],
        'iter_constraints': [
            {'type': 'limit_probe_support', 'max_angle': 23.0},
            {'type': 'clamp_object_amplitude', 'amplitude': 1.0},
        ],
        'update_object': True,
        'update_probe': {'after': 5},
        'update_positions': {'after': 5},
        'save': {'every': 10},
        'save_images': {'every': 10},
        'save_options': {
            'images': ['probe', 'probe_recip', 'object_phase_sum', 'object_mag_sum', 'object_phase_stack', 'object_mag_stack'],
        },
    }, EngineHook)

    observer.save_json(plan, [engine_1])
    execute_engine(recons, engine_1)

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

    # `constant_liar` ensures efficient batch optimization
    sampler = TPESampler(constant_liar=True)
    # don't prune before iteration 21, don't prune at a given step unless we have at least 10 datapoints
    pruner = PercentilePruner(50.0, n_min_trials=10, n_warmup_steps=21)
    # noppruner will not prune

    if verb in ('recreate', 'create'):
        if verb == 'recreate':
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        study = optuna.create_study(
            study_name=STUDY_NAME, storage=storage,
            sampler=sampler, pruner=pruner,
        )
    elif verb == 'delete':
        optuna.delete_study(study_name=STUDY_NAME, storage=storage)
    elif verb == 'run':
        study = optuna.load_study(
            study_name=STUDY_NAME, storage=storage,
            sampler=sampler, pruner=pruner,
        )
        study.optimize(objective, n_trials=200, catch=ValueError)
    else:
        print(f"Unknown command '{verb}'. Expected 'run', 'create', 'recreate', or 'delete'", file=sys.stderr)
        sys.exit(2)
