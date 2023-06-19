import torch


class Profiling:
    def __init__(self):
        self._hook_handles = []
        self._min_max = {}

    def __call__(self, module, _mod_in, mod_out):
        if module in self._min_max:
            mm = self._min_max[module]
            mm.set_min(min(mm.get_min(), torch.min(mod_out)))
            mm.set_max(min(mm.get_max(), torch.max(mod_out)))
        else:
            self._min_max[module] = MinMax(torch.min(mod_out), torch.max(mod_out))

    def get_min_max(self):
        return self._min_max

    def register_hook(self, model, layer_class):
        for layer in model.modules():
            if isinstance(layer, layer_class):
                handle = layer.register_forward_hook(self)
                self._hook_handles.append(handle)

        return self._hook_handles


class MinMax:
    def __init__(self, min, max) -> None:
        self._min = min
        self._max = max

    def __str__(self) -> str:
        return f"({self._min}, {self._max})"

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def set_min(self, min):
        self._min = min

    def set_max(self, max):
        self._max = max


def get_deltas(dict):
    min_vals, max_vals = [], []

    for k in dict:
        mm = dict[k]
        min_vals.append(float(mm.get_min()))
        max_vals.append(float(mm.get_max()))

    # return abs((max(min_vals) - min(min_vals))), abs((max(max_vals) - min(min_vals)))
    return min(min_vals), max(min_vals), min(max_vals), max(max_vals)
