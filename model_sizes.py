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

        with profile(activities=[torch.profiler.ProfilerActivity.CUDA],      
            profile_memory=True,
            record_shapes=True,) as prof:
            model(inputs)
        print(model_name)
        print(prof.key_averages().table(row_limit=7))

        del inputs, model

        # param_size = 0
        # for param in model.parameters():
        #     param_size += param.nelement() * param.element_size()

        # buffer_size = 0
        # for buffer in model.buffers():
        #     buffer_size += buffer.nelement() * buffer.element_size()

        # size_all = (param_size + buffer_size) / 1024**2
        # print('{}: {:.3f}MB'.format(model_name, size_all))


if __name__ == '__main__':
    main()
