import os

DATA_PATH = "/home/lucasroquet/ILSVRC2012/"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH = "data/gold_save.pt"
LOG_FILENAME = "log/sample-tool.log"

# image transformation parameters
# IMG_SIZE = (518, 518)  # vit h 14 expects 518*518 resolution
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# top_k max
TOP_K_MAX = 1

# DATASET RELATIVE PARAMS
NB_DATA_TO_LOAD = 1000
DEFAULT_INDEX = 493

# MODEL NAMES
VIT_LARGE_PATCH14_CLIP_224 = "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k"

VIT_BASE_PATCH16_224 = "vit_base_patch16_224"
VIT_BASE_PATCH16_384 = "vit_base_patch16_384"

EVA_BASE_PATCH14_448_MIM = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"

VIT_HUGE_PATCH14_CLIP_336 = "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
VIT_HUGE_PATCH14_CLIP_224 = "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k"

VIT_BASE_PATCH32_224_SAM = "vit_base_patch32_224.sam_in1k"
