from typing import Any
import torch
import configs


class HardenedIdentity(torch.nn.Identity):
    __MAX_LAYER_VALUE = 1024.0
    __MIN_LAYER_VALUE = -__MAX_LAYER_VALUE
    __TO_NAN_VALUE = 0.0

    # values from profiling, see profiler.py
    __PROFILES = {
        configs.VIT_BASE_PATCH16_224: (-35.314239501953125, 18.586524963378906),
        configs.VIT_BASE_PATCH16_384: (-56.74940872192383, 21.78042984008789),
        configs.VIT_BASE_PATCH32_224_SAM: (-33.16692352294922, 4.52992582321167),
        configs.VIT_HUGE_PATCH14_CLIP_224: (-90.05049896240234, 29.989328384399414),
        configs.VIT_HUGE_PATCH14_CLIP_336: (-74.11883544921875, 41.5010871887207),
        configs.VIT_LARGE_PATCH14_CLIP_224: (-223.64398193359375, 52.204036712646484),
        configs.EVA_BASE_PATCH14_448_MIM: (-702.4556274414062, 38.16182327270508),
    }

    # each min/max value is multiply by this constant to avoid too tight value restriction
    __BOUND_RATIO = 1.3

    def __init__(self, model: str | None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if model in self.__PROFILES:
            self.__MIN_LAYER_VALUE, self.__MAX_LAYER_VALUE = self.__get_min_max(model)

    def __get_min_max(self, model):
        if model in self.__PROFILES:
            min, max = self.__PROFILES[model]
            min = round(min * self.__BOUND_RATIO, 0)
            max = round(max * self.__BOUND_RATIO, 0)

        return min, max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Keep the layer original behavior and implement smart hardening
        return torch.clamp(  # Remove too large values, positive and negative
            input=torch.nan_to_num(  # Remove all nan values
                super(HardenedIdentity, self).forward(input=input),
                nan=self.__TO_NAN_VALUE,
                posinf=self.__MAX_LAYER_VALUE,
                neginf=self.__MIN_LAYER_VALUE,
            ),
            min=self.__MIN_LAYER_VALUE,
            max=self.__MAX_LAYER_VALUE,  # Other values that are too large and are not inf
        )
