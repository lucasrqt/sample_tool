# SAMPLE TOOL

## Summary

Tool used to measure the impact of injections on ViTs.
This tool iterates over ImageNet validation set to perform an inference always on the same image.
(To change the index, modify the variable `DEFAULT_INDEX` from `configs.py`)

## How to use it

There are 2 modes, simple inference and comparison.

### Simple inference

The tool will perform an inference on the image at the given index and the result will be saved in the file `gold_save.pt` in `data/`.
To run in this mode, simply do `./main.py` 

### Comparison

In comparison mode, the tool will perform the inference on the given image but it will compare the results with the `gold_save.pt` file.
This will give us the information if something has been modified during the inference.

To run the comparison, do `./main.py -l data/gold_save.pt` (or `--load`).