import os

DATA_PATH = "/home/lucasroquet/ILSVRC2012/"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
GOLD_BASE = "data"
LOG_FILENAME = "log/sample-tool.log"

# top_k max
TOP_K_MAX = 1

# DATASET RELATIVE PARAMS
NB_DATA_TO_LOAD = 1000
DEFAULT_INDEX = 493
BATCH_SIZE = 5

# MODEL NAMES
VIT_LARGE_PATCH14_CLIP_224 = "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k"

VIT_BASE_PATCH16_224 = "vit_base_patch16_224"
VIT_BASE_PATCH16_384 = "vit_base_patch16_384"

VIT_BASE_PATCH32_224_SAM = "vit_base_patch32_224.sam"

# SwinV2
# https://huggingface.co/timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k
# https://huggingface.co/timm/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
# https://huggingface.co/timm/swinv2_large_window12to16_192to256.ms_in22k_ft_in1k
# https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k
SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K = (
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"
)
SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K = (
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"
)
SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K = (
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"
)
SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K = (
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"
)

# EVA
# https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in1k
EVA_LARGE_PATCH14_448_MIM = "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k"
EVA_BASE_PATCH14_448_MIM = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
EVA_SMALL_PATCH14_448_MIN = "eva02_small_patch14_336.mim_in22k_ft_in1k"

# Max vit
# https://huggingface.co/timm/maxvit_large_tf_384.in21k_ft_in1k
# https://huggingface.co/timm/maxvit_large_tf_512.in21k_ft_in1k
MAXVIT_LARGE_TF_384 = "maxvit_large_tf_384.in21k_ft_in1k"
MAXVIT_LARGE_TF_512 = "maxvit_large_tf_512.in21k_ft_in1k"

MODELS = [
    VIT_BASE_PATCH16_224,
    VIT_BASE_PATCH16_384,
    VIT_BASE_PATCH32_224_SAM,
    # VIT_HUGE_PATCH14_CLIP_224,
    # VIT_HUGE_PATCH14_CLIP_336,
    VIT_LARGE_PATCH14_CLIP_224,
    SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K,
    SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K,
    SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K,
    SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K,
    EVA_BASE_PATCH14_448_MIM,
    EVA_LARGE_PATCH14_448_MIM,
    EVA_SMALL_PATCH14_448_MIN,
    MAXVIT_LARGE_TF_384,
    MAXVIT_LARGE_TF_512,
]