#! /usr/bin/python3

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import torchvision.transforms as T
import configs, hardened_identity
import argparse
import logging
import timm


def replace_identity(module, name, model_name):
    """Recursively put desired module in nn.module module."""
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.Identity:
            # print("replaced: ", name, attr_str)
            new_identity = hardened_identity.HardenedIdentity(model_name)
            setattr(module, attr_str, new_identity)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_identity(immediate_child_module, name, model_name)


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
    arg_parser = argparse.ArgumentParser(
        prog="sample-tool", add_help=True, formatter_class=argparse.RawTextHelpFormatter
    )
    arg_parser.add_argument(
        "-l",
        "--loadsave",
        help="load saved for the chosen model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    arg_parser.add_argument(
        "-r",
        "--replace-id",
        help="replace identity layers by hardened ones",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    arg_parser.add_argument(
        "-m",
        "--model",
        help="specify the wanted TIMM model: (default: {}) \n{}".format(
            configs.VIT_BASE_PATCH16_224, "\n".join(configs.MODELS)
        ),
        default=configs.VIT_BASE_PATCH16_224,
        type=str,
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

    ### --
    # model initialization, Vision Transformer
    model_name = args.model
    if not model_name in configs.MODELS:
        print(
            f" [-] Model '{model_name}' not available, selecting {configs.VIT_BASE_PATCH16_224}.\n"
            "     Please see available models by running this tool with option -h (./main.py -h)."
        )
        model_name = configs.VIT_BASE_PATCH16_224

    model = timm.create_model(model_name, pretrained=True)

    # putting model on GPU
    model.to("cuda")

    # setting mode for inference
    model.eval()

    # check if hardened mode
    if args.replace_id:
        replace_identity(model, "model", model_name)

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
    data_iter = iter(data_loader)

    # inference w/ dataloader
    for _i in range(configs.DEFAULT_INDEX):
        image, label = next(data_iter)

    with torch.no_grad():
        # puting image on GPU
        image = image.to("cuda")

        # getting the prediction
        output = model(image)
        torch.cuda.synchronize(device=torch.device("cuda"))

        # moving output to CPU
        output_cpu = output.to("cpu")

    if args.replace_id:
        save_name = (
            f"{configs.BASE_DIR}/{configs.GOLD_BASE}/goldsave_{model_name}-HD.pt"
        )
    else:
        save_name = f"{configs.BASE_DIR}/{configs.GOLD_BASE}/goldsave_{model_name}.pt"

    if not args.loadsave:
        pred = get_top_k_labels(output, top_k=configs.TOP_K_MAX).item()
        if pred != label.item():
            print(f" [-] wrong classification value {pred}, expected {label.item()}")

        torch.save(output_cpu, save_name)
    else:
        prev_output = torch.load(save_name, map_location=torch.device("cuda"))
        output_errs, class_errs = compare_classification(
            output, prev_output, configs.TOP_K_MAX, logger=logger
        )
        print(
            f" [+] ouput errors: {output_errs} -- classification errors: {class_errs}"
        )


if __name__ == "__main__":
    main()
