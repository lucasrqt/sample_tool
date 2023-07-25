#!/bin/python3

import configs
import timm
import torch


def main():
    for model_name in configs.MODELS:
        model = timm.create_model(model_name, pretrained=True)
        model.to("cuda")
        model.eval()
        
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement * buffer.element_size()

        size_all = (param_size + buffer_size) / 1024**2
        print('{}: {:.3f}MB'.format(model_name, size_all))


if __name__ == '__main__':
    main()
