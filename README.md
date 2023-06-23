# SAMPLE TOOL

## Summary

Tool used to measure the impact of injections on ViTs.
This tool iterates over ImageNet validation set to perform an inference always on the same image.
(To change the index, modify the variable `DEFAULT_INDEX` from `configs.py`)

## Model choice

When running the tool, you can choose wich TIMM model you want (the avaiable models are displayed in the help of the program).
To add models, simply add the name in a string in `configs.py` and add the new variable to the `MODELS` list.

To select a model, just use the option `-m <model-name>`.

By default, the model is `"vit_base_patch16_224"`.

## How to use it

There are 2 modes, simple inference and comparison.

### Simple inference

The tool will perform an inference on the image at the given index and the result will be saved in the file `gold_save.pt` in `data/`.
To run in this mode, simply do `./main.py` 

### Comparison

In comparison mode, the tool will perform the inference on the given image but it will compare the results with the `goldsave_<model-name>(-HD).pt` file (`-HD` if hardened mode is enabled).
This will give us the information if something has been modified during the inference.

To run the comparison, do `./main.py -l` (or `--loadsave`).

## Hardened mode

The tool comes with a hardened mode. It means that we replace Identity layers of the different models with hardened ones that applies value restriction. You can profile the wanted models with the `profiler.py` utiliy and then add the model profile in `hardened_identity.py` file.

By default, the models are not used with hardened identity layers.

To apply replace the identity layers, use option `-r` (`--replace-id`).