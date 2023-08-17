#!/usr/bin/python3
import argparse
import logging
import os
import time

import timm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

import configs
from main import replace_identity

WARM_UP_ITERATIONS = 10


def main():
    torch.cuda.cudart().cudaProfilerStop()
    # parser part
    arg_parser = argparse.ArgumentParser(
        prog="perf_measure", add_help=True, formatter_class=argparse.RawTextHelpFormatter
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
        "-s",
        "--seed",
        help="set the seed for batch inference",
        type=int,
        default=493,
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
    arg_parser.add_argument(
        "-i",
        "--iterations",
        help="How many iterations will run of the same inference",
        default=1,
        type=int,
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

    # logger = logging.getLogger()

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

    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(args.seed)
    # initializing the dataset
    test_set = ImageNet(
        root=configs.DATA_PATH.replace("lucasroquet", os.getlogin()),
        transform=transforms,
        split="val",
    )
    subset = torch.utils.data.RandomSampler(
        data_source=test_set, replacement=False, generator=sampler_generator
    )

    # initializing the dataloader
    data_loader = DataLoader(test_set, batch_size=configs.BATCH_SIZE, sampler=subset)
    data_iter = iter(data_loader)

    # imagenet_labels = dict(
    #     enumerate(
    #         open(f"/home/{os.getlogin()}/sample_tool/data/ilsvrc2012_wordnet_lemmas.txt")
    #     )
    # )

    # inference w/ dataloader
    # for _i in range(configs.DEFAULT_INDEX):
    #     image, label = next(data_iter)
    images, labels = next(data_iter)

    with torch.no_grad():
        # putting image on GPU
        images = images.to("cuda")
        # Warm up
        print(f"Warming up with {WARM_UP_ITERATIONS} iterations")
        tic = time.time()
        for _ in range(WARM_UP_ITERATIONS):
            _ = model(images)
            torch.cuda.synchronize(device=torch.device("cuda"))
        print(f"Warm up finished in {time.time() - tic}s")

        torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()
        for iteration in range(args.iterations):
            # getting the prediction
            output = model(images)
            torch.cuda.synchronize(device=torch.device("cuda"))
        end_time = time.time()
        torch.cuda.cudart().cudaProfilerStop()

        # moving output to CPU
        output_cpu = output.to("cpu")
    print(f"PERF_MEASURE::model:{args.model} hardening:{args.replace_id} "
          f"iterations:{args.iterations} it_time:{(end_time - start_time) / args.iterations:.4f} "
          f"shape cpu:{output_cpu.shape}")
    # if args.replace_id:
    #     save_name = (
    #         f"{configs.BASE_DIR}/{configs.GOLD_BASE}/goldsave_{model_name}-HD.pt"
    #     )
    # else:
    #     save_name = f"{configs.BASE_DIR}/{configs.GOLD_BASE}/goldsave_{model_name}.pt"

    # if not args.loadsave:
    #     pred = get_top_k_labels(output_cpu, top_k=configs.TOP_K_MAX)
    #     print(labels == pred.squeeze())
    #     for i in range(configs.BATCH_SIZE):
    #         img_lbl = imagenet_labels[labels[i].item()].rstrip("\n")
    #         pred_lbl = imagenet_labels[pred[i].item()].rstrip("\n")
    #         print(f"expected: {img_lbl} -- pred: {pred_lbl}")
    #
    #     # if pred != labels.item():
    #     #   print(f" [-] wrong classification value {pred}, expected {labels.item()}")
    #
    #     torch.save(output_cpu, save_name)
    # else:
    #     prev_output = torch.load(save_name, map_location=torch.device("cpu"))
    #     errors = compare_classification(output_cpu, prev_output, configs.TOP_K_MAX)
    #     if errors != {}:
    #         res = f""
    #         for idx in errors:
    #             otpt, clss = errors[idx]
    #             res += f"{idx}: ouput errors: {otpt} -- classification: {clss}\n"
    #         print(res)


if __name__ == "__main__":
    main()
