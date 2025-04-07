from __future__ import annotations

import typing as t

from .types import Dataclass, Slices, BackendName, Flag, ReconsVars
from .hooks import RawDataHook, ProbeHook, ObjectHook, ScanHook, EngineHook, FlagHook, PostInitHook, PostLoadHook
from .hooks.solver import NoiseModelHook, ConventionalSolverHook, PositionSolverHook, GradientSolverHook
from .hooks.solver import IterConstraintHook, GroupConstraintHook, CostRegularizerHook


FlagLike: t.TypeAlias = t.Union[bool, Flag, FlagHook]


SaveType: t.TypeAlias = t.Literal[
    'probe', 'probe_mag', 'probe_recip', 'probe_recip_mag',
    'object_phase_stack', 'object_phase_sum',
    'object_mag_stack', 'object_mag_sum',
]


class SaveOptions(Dataclass, kw_only=True):
    images: t.Tuple[SaveType, ...] = ('probe', 'object_phase_stack')
    crop_roi: bool = True
    unwrap_phase: bool = True
    img_dtype: t.Literal['float', '8bit', '16bit', '32bit'] = '16bit'

    out_dir: str = "{name}"
    img_fmt: str = "{type}_iter{iter.total_iter}.tiff"
    hdf5_fmt: str = "iter{iter.total_iter}.h5"


class EnginePlan(Dataclass, kw_only=True):
    sim_shape: t.Optional[t.Tuple[int, int]] = None
    resize_method: t.Literal['pad_crop', 'resample'] = 'pad_crop'

    probe_modes: int = 1

    slices: t.Optional[Slices] = None

    niter: int = 10
    grouping: t.Optional[int] = None
    compact: bool = False
    shuffle_groups: t.Optional[FlagLike] = None

    update_probe: FlagLike = True
    update_object: FlagLike = True
    update_positions: FlagLike = False

    calc_error: FlagLike = Flag(every=1)
    calc_error_fraction: float = 0.1

    save: FlagLike = False
    save_images: FlagLike = False
    save_options: SaveOptions = SaveOptions()

    send_every_group: bool = False


class AmplitudeNoisePlan(Dataclass, kw_only=True):
    gaussian_variance: float = 0.1
    eps: float = 1.0e-3
    offset: float = 0.0


class AnscombeNoisePlan(AmplitudeNoisePlan, kw_only=True):
    offset: float = 0.375


class PoissonNoisePlan(AmplitudeNoisePlan, kw_only=True):
    eps: float = 1.0e-3


NoiseModelHook.known['amplitude'] = ('phaser.engines.common.noise_models:AmplitudeNoiseModel', AmplitudeNoisePlan)
NoiseModelHook.known['anscombe'] = ('phaser.engines.common.noise_models:AnscombeNoiseModel', AnscombeNoisePlan)
NoiseModelHook.known['poisson'] = ('phaser.engines.common.noise_models:PoissonNoiseModel', PoissonNoisePlan)


class LSQMLSolverPlan(Dataclass, kw_only=True):
    stochastic: bool = True

    beta_object: float = 1.0
    beta_probe: float = 1.0

    illum_reg_object: float = 1e-2
    illum_reg_probe: float = 1e-2

    gamma: float = 1e-4


class EPIESolverPlan(Dataclass, kw_only=True):
    beta_object: float = 1.0
    beta_probe: float = 1.0


ConventionalSolverHook.known['lsqml'] = ('phaser.engines.conventional.solvers:LSQMLSolver', LSQMLSolverPlan)
ConventionalSolverHook.known['epie'] = ('phaser.engines.conventional.solvers:EPIESolver', EPIESolverPlan)


class ConventionalEnginePlan(EnginePlan, kw_only=True):
    noise_model: NoiseModelHook
    solver: ConventionalSolverHook
    position_solver: t.Optional[PositionSolverHook] = None

    group_constraints: t.List[GroupConstraintHook]
    iter_constraints: t.List[IterConstraintHook]


class GradientEnginePlan(EnginePlan):
    noise_model: NoiseModelHook
    solvers: t.Dict[ReconsVars, GradientSolverHook]

    regularizers: t.List[CostRegularizerHook]
    group_constraints: t.List[GroupConstraintHook]
    iter_constraints: t.List[IterConstraintHook]


class SGDSolverPlan(Dataclass, kw_only=True):
    learning_rate: float
    momentum: t.Optional[float] = None
    nesterov: bool = True


class AdamSolverPlan(Dataclass, kw_only=True):
    learning_rate: float

    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1.0e-8
    eps_root: float = 0.0

    nesterov: bool = False


class PolyakSGDSolverPlan(Dataclass, kw_only=True):
    max_learning_rate: float
    f_min: float
    scaling: float = 1.0
    eps: float = 0.0


GradientSolverHook.known['sgd'] = ('phaser.engines.gradient.solvers:SGDSolver', SGDSolverPlan)
GradientSolverHook.known['adam'] = ('phaser.engines.gradient.solvers:AdamSolver', AdamSolverPlan)
GradientSolverHook.known['polyak_sgd'] = ('phaser.engines.gradient.solvers:PolyakSGDSolver', PolyakSGDSolverPlan)

EngineHook.known['conventional'] = ('phaser.engines.conventional.run:run_engine', ConventionalEnginePlan)
EngineHook.known['gradient'] = ('phaser.engines.gradient.run:run_engine', GradientEnginePlan)


class ReconsPlan(Dataclass, kw_only=True):
    name: str

    backend: t.Optional[BackendName] = None
    dtype: t.Literal['float32', 'float64'] = 'float32'

    wavelength: t.Optional[float] = None

    raw_data: RawDataHook

    post_load: t.Sequence[PostLoadHook] = ()

    init_probe: ProbeHook
    init_object: ObjectHook
    init_scan: ScanHook

    post_init: t.Sequence[PostInitHook] = ()

    slices: t.Optional[Slices] = None

    engines: t.List[EngineHook]
    #engines: t.List[t.Annotated[t.Union[ConventionalEngine, GradientEngine], Tagged('type')]]