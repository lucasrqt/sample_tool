#! /usr/bin/python3

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import torchvision.transforms as T
import configs
import argparse
import logging
import timm


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = 0) -> bool:
    """Compare based or not in a threshold, if threshold is none then it is equal comparison"""
    if threshold > 0:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))


def get_top_k_labels(tensor: torch.tensor, top_k: int) -> torch.tensor:
    proba = torch.nn.functional.softmax(tensor, dim=1)
    return torch.topk(proba, k=top_k).indices.squeeze(0)


def compare_classification(
    output_tsr: torch.tensor, golden_tsr: torch.tensor, top_k: int, logger=None
) -> int:
    output_errors, classification_errors = 0, 0
    output_tsr, golden_tsr = output_tsr.to("cpu"), golden_tsr.to("cpu")

    # tensor comparison
    if not equal(output_tsr, golden_tsr, threshold=1e-4):
        for _i, (output, golden) in enumerate(zip(output_tsr, golden_tsr)):
            if not equal(output, golden):
                err_str = (
                    f"error, output modified -- expected:{golden}  output:{output}"
                )
                output_errors += 1
                if logger:
                    logger.error(err_str)

    # top k comparison to check if classification has changed
    output_topk = get_top_k_labels(output_tsr, top_k)
    golden_topk = get_top_k_labels(golden_tsr, top_k)
    if equal(output_topk, golden_topk) is False:
        for i, (tpk_found, tpk_gold) in enumerate(zip(output_topk, golden_topk)):
            if tpk_found != tpk_gold:
                err_str = (
                    f"wrong classification -- expected:{tpk_gold}  output:{tpk_found}"
                )
                classification_errors += 1
                if logger:
                    logger.error(err_str)

    return output_errors, classification_errors


def main():
    # parser part
    arg_parser = argparse.ArgumentParser(prog="sample-tool", add_help=True)
    arg_parser.add_argument(
        "-l", "--loadsave", help="path to the save to load", type=str
    )
    args = arg_parser.parse_args()

    # logger
    log_filename = f"{configs.BASE_DIR}/{configs.LOG_FILENAME}"
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(asctime)s %(message)s",
        level=logging.ERROR,
    )

    logger = logging.getLogger()

    # model init
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

    # image labels
    imagenet_labels = dict(
        enumerate(open(f"{configs.BASE_DIR}/data/ilsvrc2012_wordnet_lemmas.txt"))
    )

    ### --
    # model initialization, Vision Transformer
    model = timm.create_model(configs.VIT_BASE_PATCH16_224, pretrained=True)

    # putting model on GPU
    model.to("cuda")

    # setting mode for inference
    model.eval()

    data_iter = iter(data_loader)

    # inference w/ dataloader
    for _i in range(configs.DEFAULT_INDEX + 1):
        image, label = next(data_iter)

    # puting image on GPU
    image = image.to("cuda")

    # getting the prediction
    output = model(image)

    # moving output to CPU
    output_cpu = output.to("cpu")

    if not args.loadsave:
        pred = get_top_k_labels(output, top_k=configs.TOP_K_MAX).item()
        if pred != label.item():
            print(f" [-] wrong classification value {pred}, expected {label.item()}")

        torch.save(output_cpu, f"{configs.BASE_DIR}/{configs.OUTPUT_PATH}")
    else:
        prev_output = torch.load(args.loadsave, map_location=torch.device("cuda"))
        output_errs, class_errs = compare_classification(
            output, prev_output, configs.TOP_K_MAX, logger=logger
        )
        print(
            f" [+] ouput errors: {output_errs} -- classification errors: {class_errs}"
        )


if __name__ == "__main__":
    main()
