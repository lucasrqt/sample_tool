#! /usr/bin/python3

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import PIL
import matplotlib.pyplot as plt
import configs
import argparse

transforms = [
              T.Resize(configs.IMG_SIZE),
              T.ToTensor(),
              T.Normalize(configs.NORMALIZE_MEAN, configs.NORMALIZE_STD),
              ]

transforms = T.Compose(transforms)

# initializing the dataset
test_set = ImageNet(
    root=configs.DATA_PATH,
    transform=transforms,
    split="val",
)

# initializing the dataloader
data_loader = DataLoader(test_set, batch_size=1, shuffle=True)

#image labels
imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))

# model initialization, Vision Transformer
model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)

# putting model on GPU
model.to("cuda")

# setting mode for inference
model.eval()


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = 0) -> bool:
    """ Compare based or not in a threshold, if threshold is none then it is equal comparison    """
    if threshold > 0:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))


def main():
    # parser part
    arg_parser = argparse.ArgumentParser(prog="sample-tool", add_help=True)
    arg_parser.add_argument('-l', "--loadsave", help="path to the save to load", type=str)
    args = arg_parser.parse_args()

    # inference w/ dataloader
    image, label = next(iter(data_loader))

    # puuting image on GPU
    image = image.to("cuda")

    # getting the prediction
    output = model(image)

    # moving output to CPU
    output_cpu = output.to("cpu")
    #pred = int(torch.argmax(output))

    if not args.loadsave:
        torch.save(output_cpu, configs.OUTPUT_PATH)
    else:
        prev_output = torch.load(configs.OUTPUT_PATH, map_location=torch.device("cuda"))
        if not equal(output, prev_output):
            pass

    #print(f"label: {imagenet_labels[label.item()]}prediction: {imagenet_labels[pred]}")


if __name__ == '__main__':
    main()
