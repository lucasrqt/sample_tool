#!/usr/bin/python3

import pandas as pd
from configs_parser import *
from typing import List, Tuple
from app_parser import App, Parser
import os
import re
import linecache


def main():
    # parsser init
    parser = Parser()

    # sample tool app init
    results_folder_path = f"{BASE_DIR}/sample_tool_warp_40"
    injections = {
        G_GP: [FLIP_SINGLE_BIT, ZERO_VALUE, WARP_ZERO_VALUE],
        G_FP32: [FLIP_SINGLE_BIT, ZERO_VALUE, WARP_ZERO_VALUE],
    }
    sample_tool = App("sample-tool", results_folder_path, 40, injections)

    print(f" [+] parsing results...")
    res_stdout, res_stderr = parser.parse_per_bfm(sample_tool)

    print(f" [+] parsing results for kernel")
    res_kernels = parser.parse_per_kernel(sample_tool)

    # print(f" [+] deep parsing results for kernel")
    # res_ker_bfm = parser.parse_per_kernel_bfm(sample_tool)

    df_stdout = parser.dict_to_dataframe(res_stdout)
    df_stderr = parser.dict_to_dataframe(res_stderr)
    df_stdout.to_csv(f"{results_folder_path}/results_stdout.csv")
    df_stderr.to_csv(f"{results_folder_path}/results_stderr.csv")

    df_kernels = pd.DataFrame.from_dict(res_kernels, orient="index")
    df_kernels.to_csv(f"{results_folder_path}/results_kernel.csv")

    # df_kbfm = dict_to_dataframe(res_ker_bfm)
    # df_kbfm.to_csv(f"./{results_folder_path}/results_kernel_bfm.csv")


if __name__ == "__main__":
    main()
