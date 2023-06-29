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
    inj_per_fm = 250
    injections = {
        G_GP: [FLIP_SINGLE_BIT, RANDOM_VALUE],
        G_LD: [FLIP_SINGLE_BIT, RANDOM_VALUE],
        G_FP32: [FLIP_SINGLE_BIT, RANDOM_VALUE, WARP_RANDOM_VALUE],
    }

    st_16b_224_name = "vit16_224-mulin"
    st_16b_224_path = f"{BASE_DIR}/vit16_224-mulin"
    sample_tool_16b_384 = App(st_16b_224_name, st_16b_224_path, inj_per_fm, injections)

    apps = [sample_tool_16b_384]

    errs_by_model = {}
    for app in apps:
        print(f" [+] parsing results per category")
        res = parser.parse_per_cat(app)
        print(res)
        errs_by_model[app.app_name] = {"SDC": 0.0, "Critical SDC": 0.0, "DUE": 0.0}
        bfms_cnt = 0
        for group in res:
            for bfm in res[group]:
                bfms_cnt += 1
                errs_by_model[app.app_name]["SDC"] += res[group][bfm]["SDC"]
                errs_by_model[app.app_name]["Critical SDC"] += res[group][bfm][
                    "Critical SDC"
                ]
                errs_by_model[app.app_name]["DUE"] += res[group][bfm]["DUE"]

        errs_by_model[app.app_name]["SDC"] /= bfms_cnt
        errs_by_model[app.app_name]["Critical SDC"] /= bfms_cnt
        errs_by_model[app.app_name]["DUE"] /= bfms_cnt

    print(f" [+] parsing results per bfm")
    res_stdout, res_stderr = parser.parse_per_bfm(app)

    print(f" [+] parsing results for kernel")
    res_kernels = parser.parse_per_kernel(app)

    # print(f" [+] deep parsing results for kernel")
    # res_ker_bfm = parser.parse_per_kernel_bfm(sample_tool)

    df_res = parser.dict_to_dataframe(res)
    df_stdout = parser.dict_to_dataframe(res_stdout)
    df_stderr = parser.dict_to_dataframe(res_stderr)

    df_res.to_csv(f"{app.app_folder}/results_cat.csv")
    df_stdout.to_csv(f"{app.app_folder}/results_stdout.csv")
    df_stderr.to_csv(f"{app.app_folder}/results_stderr.csv")

    # df_kernels = pd.DataFrame.from_dict(res_kernels, orient="index")
    # df_kernels.to_csv(f"{app.app_folder}/results_kernel.csv")

    # df_kbfm = dict_to_dataframe(res_ker_bfm)
    # df_kbfm.to_csv(f"./{app.app_folder}/results_kernel_bfm.csv")

    print(errs_by_model)


if __name__ == "__main__":
    main()
