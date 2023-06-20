#! /usr/bin/python3

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import torchvision.transforms as T
import configs, hardened_identity, profiling
import timm


def replace_identity(module, name):
    """Recursively put desired module in nn.module module."""
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Identity:
            # print("replaced: ", name, attr_str)
            new_identity = hardened_identity.HardenedIdentity()
            setattr(module, attr_str, new_identity)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_identity(immediate_child_module, name)


def main():
    ### --
    # model initialization, Vision Transformer
    model = timm.create_model(configs.VIT_BASE_PATCH32_224_SAM, pretrained=True)

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
    data_loader = DataLoader(test_set, batch_size=1)

    # image labels
    imagenet_labels = dict(
        enumerate(open(f"{configs.BASE_DIR}/data/ilsvrc2012_wordnet_lemmas.txt"))
    )

    ### --
    # model initialization, Vision Transformer
    model = timm.create_model(configs.EVA_BASE_PATCH14_448_MIM, pretrained=True)

    # putting model on GPU
    model.to("cuda")

    # setting mode for inference
    model.eval()

    # REPLACING IDENTITY LAYER
    replace_identity(model, "model")

    # profiling part
    profiler = profiling.Profiling()
    _hooks = profiler.register_hook(model, hardened_identity.HardenedIdentity)

    data_iter = iter(data_loader)

    with torch.no_grad():
        # inference w/ dataloader
        for image, label in data_iter:
            # puting image on GPU
            image = image.to("cuda")

            # getting the prediction
            output = model(image)

            min_min, min_max, max_min, max_max = profiling.get_deltas(
                profiler.get_min_max()
            )

            # moving output to CPU
            image = image.to("cpu")
            output = output.to("cpu")
            del image, output
    print(
        f"min_min: {min_min},  min_max: {min_max}\nmax_min: {max_min}, max_max: {max_max}"
    )
    # print(min_min, min_max, max_min, max_max)


if __name__ == "__main__":
    main()
