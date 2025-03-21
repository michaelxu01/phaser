
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.solver import NoiseModel
from phaser.plan import AmplitudeNoisePlan, AnscombeNoisePlan
from phaser.utils.num import get_array_module
from phaser.state import ReconsState


class AmplitudeNoiseModel(NoiseModel[None]):
    @classmethod
    def name(cls) -> str:
        return "amplitude"

    def __init__(self, args: None, props: AmplitudeNoisePlan):
        self.offset: float = props.offset
        self.gaussian_variance: float = props.gaussian_variance

        self.var: float = 1 + self.gaussian_variance

        self.eps: float = props.eps

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[float, None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)
        patterns = xp.maximum(exp_patterns, 0.0)

        return (t.cast(float,
            2. * xp.sum(mask * (
                xp.sqrt(patterns + self.offset) - xp.sqrt(model_intensity + self.offset)
            )**2) / self.var),
            state
        )

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[NDArray[numpy.complexfloating], None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)
        patterns = xp.maximum(exp_patterns, 0.0)

        update = xp.sqrt(patterns + self.offset) / (xp.sqrt(model_intensity + self.offset) + self.eps) - 1.0
        update *= mask # / self.var
        #print(f"min update: {xp.min(update).get()}")
        #print(f"max update: {xp.max(update).get()}")

        return (update * model_wave, state)


class AnscombeNoiseModel(AmplitudeNoiseModel):
    @classmethod
    def name(cls) -> str:
        return "anscombe"

    def __init__(self, args: None, props: AnscombeNoisePlan):
        super().__init__(args, props)