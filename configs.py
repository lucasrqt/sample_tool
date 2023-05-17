
DATA_PATH="../ILSVRC2012/"
OUTPUT_PATH="./gold_save.pt"

# image transformation parameters
IMG_SIZE = (518, 518) # vit h 14 expects 518*518 resolution
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)