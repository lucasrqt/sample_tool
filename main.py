#! /usr/bin/python3

from torchvision.datasets import ImageNet
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision.transforms import ToTensor
from datasets import load_dataset

PATH="../ILSVRC2012/"

test_set = ImageNet(
    root=PATH,
    transform=ToTensor(),
    split="val"
)

# model initialization, Vision Transformer
model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)

# putting model on GPU
model.to("cuda")

# setting mode for inference
model.eval()