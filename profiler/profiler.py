#! /usr/bin/python3

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import configs, hardened_identity, profiling
import timm
import time

MIN_VALS, MAX_VALS = [], []

class ProfileIdentity(torch.nn.Identity):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        global MIN_VALS, MAX_VALS
        MIN_VALS.append(float(torch.min(input)))
        MAX_VALS.append(float(torch.max(input)))
        return input

def replace_identity(module, name):
    """Recursively put desired module in nn.module module."""
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Identity:
            # print("replaced: ", name, attr_str)
            new_identity = ProfileIdentity()
            setattr(module, attr_str, new_identity)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_identity(immediate_child_module, name)


def main():
    models = [
        configs.VIT_BASE_PATCH16_224,
        configs.VIT_BASE_PATCH16_384,
        configs.VIT_LARGE_PATCH14_CLIP_224,
        configs.VIT_BASE_PATCH32_224_SAM,
    ]

    for model_name in models:
        start = time.time()
        model = timm.create_model(model_name, pretrained=True)

        # putting model on GPU
        model.to("cuda")

        # setting mode for inference
        model.eval()

        # REPLACING IDENTITY LAYER
        replace_identity(model, "model")

        cfg = timm.data.resolve_data_config({}, model=model)
        transforms = timm.data.transforms_factory.create_transform(**cfg)

        # initializing the dataset
        test_set = ImageNet(
            root=configs.DATA_PATH,
            transform=transforms,
            split="val",
        )

        # initializing the dataloader
        data_loader = DataLoader(test_set, batch_size=64, shuffle=True)

        # image labels
        # imagenet_labels = dict(
        #     enumerate(open(f"{configs.BASE_DIR}/data/ilsvrc2012_wordnet_lemmas.txt"))
        # )

        # profiling part
        # profiler = profiling.Profiling()
        # _hooks = profiler.register_hook(model, hardened_identity.HardenedIdentity)

        data_iter = iter(data_loader)

        print(f" [+] Profiling model {model_name}...\n")

        with torch.no_grad():
            # inference w/ dataloader
            for image, _ in data_iter:
                # puting image on GPU
                image = image.to("cuda")

                # getting the prediction
                output = model(image)
                # min, max = profiler.get_min_max()
                # min_min, min_max, max_min, max_max = profiling.get_deltas(min, max)

                del image, output

        print(
            f"Model: {model_name} ({time.time()-start}s)\n"
            + f"min: {min(MIN_VALS)}, max: {max(MAX_VALS)}\n"
            + "-" * 80
        )


if __name__ == "__main__":
    main()
