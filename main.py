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
import os
import logging

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
data_loader = DataLoader(test_set, batch_size=1)

#image labels
imagenet_labels = dict(enumerate(open('/home/lucasroquet/sample-tool/ilsvrc2012_wordnet_lemmas.txt')))

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

def get_top_k_labels(tensor: torch.tensor, top_k: int) -> torch.tensor:
    proba = torch.nn.functional.softmax(tensor, dim=1)
    return torch.topk(proba, k=top_k).indices.squeeze(0)

def compare_classification(output_tsr: torch.tensor, golden_tsr: torch.tensor, top_k: int, logger=None) -> int:
    output_errors = 0
    output_tsr, golden_tsr = output_tsr.to("cpu"), golden_tsr.to("cpu")
    output_topk = get_top_k_labels(output_tsr, top_k)
    golden_topk = get_top_k_labels(golden_tsr, top_k)
    if equal(output_topk, golden_topk) is False:
        for i, (tpk_found, tpk_gold) in enumerate(zip(output_topk, golden_topk)):
            if tpk_found != tpk_gold:
                err_str = f"error i:{i} -- g:{tpk_gold}  o:{tpk_found}"
                output_errors += 1
                if logger:
                    logger.error(err_str)

    return output_errors

def main():
    # parser part
    arg_parser = argparse.ArgumentParser(prog="sample-tool", add_help=True)
    arg_parser.add_argument('-l', "--loadsave", help="path to the save to load", type=str)
    args = arg_parser.parse_args()

    # logger
    log_filename = "./" + configs.LOG_FILENAME
    logging.basicConfig(filename=log_filename,
                                 filemode="a",
                                 format="%(asctime)s %(message)s",
                                 level=logging.ERROR,)

    logger = logging.getLogger()

    # inference w/ dataloader
    image, label = next(iter(data_loader))

    # puuting image on GPU
    image = image.to("cuda")

    # getting the prediction
    output = model(image)

    # moving output to CPU
    output_cpu = output.to("cpu")

    if not args.loadsave:
        torch.save(output_cpu, configs.OUTPUT_PATH)
    else:
        pred = int(torch.argmax(output))
        prev_output = torch.load(args.loadsave, map_location=torch.device("cuda"))
        prev_pred = int(torch.argmax(prev_output))
        print(f"loaded: {imagenet_labels[prev_pred]}calculated: {imagenet_labels[pred]}")
        nb_errs = compare_classification(output, prev_output, configs.TOP_K_MAX, logger=logger)
        print(f" [+] nb errors: {nb_errs}")


if __name__ == '__main__':
    main()
