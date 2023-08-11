#!/bin/python3

import configs
import timm
import torch
from torch.profiler import profile
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


def main():
    for model_name in configs.MODELS:
        model = timm.create_model(model_name, pretrained=True)
        model = model.to("cuda")
        model.eval()
        
        cfg = timm.data.resolve_data_config({}, model=model)
        transforms = timm.data.transforms_factory.create_transform(**cfg)

        # initializing the dataset
        test_set = ImageNet(
            root=configs.DATA_PATH,
            transform=transforms,
            split="val",
        )
        # initializing the dataloader
        
        data_loader = DataLoader(test_set, batch_size=5)
        data_iter = iter(data_loader)
        inputs, label = next(data_iter)
        inputs = inputs.to("cuda")
        torch.cuda.synchronize(device=torch.device("cuda"))

        size = torch.cuda.memory_allocated(device=torch.cuda.current_device())/1024**2

        print(f"{model_name}: {round(size,0)}MB")

        del inputs, model


if __name__ == '__main__':
    main()
