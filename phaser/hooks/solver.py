import abc
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass
from . import Hook, FlagArgs

StateT = t.TypeVar('StateT')

if t.TYPE_CHECKING:
    from phaser.engines.common.simulation import SimulationState
    from phaser.execute import Observer
    from phaser.plan import ConventionalEnginePlan


class HasState(abc.ABC, t.Generic[StateT]):
    @abc.abstractmethod
    def init_state(self, sim: 'SimulationState') -> StateT:
        ...


class NoiseModel(HasState[StateT], abc.ABC):
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        ...

    @abc.abstractmethod
    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: StateT,
    ) -> t.Tuple[float, StateT]:
        """
        Return the calculated loss, summed across the detector and averaged across the scan.

        May be called in a JAX jit context, so must have no side effects.
        """
        ...

    @abc.abstractmethod
    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: StateT,
    ) -> t.Tuple[NDArray[numpy.complexfloating], StateT]:
        """
        Return the calculated wave update `chi` in reciprocal space.

        May be called in a JAX jit context, so must have no side effects.
        """
        ...


class NoiseModelHook(Hook[None, NoiseModel]):
    known = {}


class PositionSolver(HasState[StateT], abc.ABC):
    @abc.abstractmethod
    def perform_update(
        self,
        positions: NDArray[numpy.floating],
        gradients: NDArray[numpy.floating],
        state: StateT
    ) -> t.Tuple[NDArray[numpy.floating], StateT]:
        """
        Return the calculated position updates
        """
        ...


class SteepestDescentPositionSolverProps(Dataclass):
    # fraction of optimal step to take
    step_size: float = 1e-2
    # maximum step size (in angstroms)
    max_step_size: t.Optional[float] = None


class MomentumPositionSolverProps(Dataclass):
    # fraction of optimal step to take
    step_size: float = 1e-2
    # maximum step size (in angstroms)
    max_step_size: t.Optional[float] = None
    # momentum decay rate
    momentum: float = 0.9


class PositionSolverHook(Hook[None, PositionSolver]):
    known = {
        'steepest_descent': ('phaser.engines.common.position_correction:SteepestDescentPositionSolver', SteepestDescentPositionSolverProps),
        'momentum': ('phaser.engines.common.position_correction:MomentumPositionSolver', MomentumPositionSolverProps),
    }


# refactoring:
# HasState should be an interface, as well as the different ways of regularization + noise model
# then there can be a list of each of these types in the engine


class ConstraintRegularizer(HasState[StateT], abc.ABC):
    def apply_group(self, group: NDArray[numpy.integer], sim: 'SimulationState', state: StateT) -> t.Tuple['SimulationState', StateT]:
        return (sim, state)

    def apply_iter(self, sim: 'SimulationState', state: StateT) -> t.Tuple['SimulationState', StateT]:
        return (sim, state)


class GradientRegularizer(HasState[StateT], abc.ABC):
    @abc.abstractmethod
    def calc_loss_group(self, group: NDArray[numpy.integer], sim: 'SimulationState', state: StateT) -> t.Tuple[float, StateT]:
        ...


class ClampObjectAmplitudeProps(Dataclass):
    amplitude: float = 1.1


class LimitProbeSupportProps(Dataclass):
    max_angle: float


class RegularizeLayersProps(Dataclass):
    weight: float = 0.9  # weight of regularization to apply
    sigma: float = 50.0  # standard deviation of gaussian filter


class ObjLowPassProps(Dataclass):
    max_freq: float = 0.4  # 1/px (nyquist = 0.5)


class RegularizerHook(Hook[None, t.Union[ConstraintRegularizer, GradientRegularizer]]):
    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
        'layers': ('phaser.engines.common.regularizers:RegularizeLayers', RegularizeLayersProps),
        'obj_low_pass': ('phaser.engines.common.regularizers:ObjLowPass', ObjLowPassProps),
    }


class ConventionalSolver(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        ...

    @abc.abstractmethod
    def init(self, sim: 'SimulationState') -> 'SimulationState':
        ...

    @abc.abstractmethod
    def presolve(
        self,
        sim: 'SimulationState',
        propagators: t.Optional[NDArray[numpy.complexfloating]],
        groups: t.Sequence[NDArray[numpy.int_]],
    ) -> 'SimulationState':
        ...

    @abc.abstractmethod
    def run_iteration(
        self,
        sim: 'SimulationState',
        propagators: t.Optional[NDArray[numpy.complexfloating]],
        groups: t.Sequence[NDArray[numpy.int_]], *,
        update_object: bool,
        update_probe: bool,
        update_positions: bool,
        calc_error: bool,
        calc_error_mask: NDArray[numpy.bool_],
        observer: 'Observer',
    ) -> t.Tuple['SimulationState', NDArray[numpy.floating], t.List[NDArray[numpy.floating]]]:
        """
        Run an iteration of the reconstruction. Return a tuple `(sim, pos_update, iter_errors)`.
        """
        ...


class ConventionalSolverHook(Hook['ConventionalEnginePlan', ConventionalSolver]):
    known = {}