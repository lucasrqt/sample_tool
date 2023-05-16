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


if __name__ == '__main__':
    # inference w/ dataloader
    image, label = next(iter(data_loader))
    #for image, label in data_loader:
    image = image.to("cuda")
    output = model(image)
    pred = int(torch.argmax(output))

    torch.save(output, configs.MODEL_PATH)

    print(f"label: {imagenet_labels[label.item()]}prediction: {imagenet_labels[pred]}")

    cpu = torch.device("cpu")
    gold_output = torch.load(configs.MODEL_PATH, map_location=cpu)
    gold_op_val = int(torch.argmax(gold_output))
    print(imagenet_labels[gold_op_val])
