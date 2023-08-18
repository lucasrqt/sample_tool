#!/usr/bin/python3

import pandas as pd
from configs_parser import *
from typing import List, Tuple
from app_parser import App, Parser
import os
import re
import linecache
import os.path

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

    # swin_b1216_n = "swin_b12to16-hd"
    # swin_b1216_path = f"{BASE_DIR}/pascal/{swin_b1216_n}"
    # swin_b1216 = App(swin_b1216_n, swin_b1216_path, inj_per_fm, injections)

    # swin_b1224_n = "swin_b12to14-hd"
    # swin_b1224_path = f"{BASE_DIR}/pascal/{swin_b1224_n}"
    # swin_b1224 = App(swin_b1224_n, swin_b1224_path, inj_per_fm, injections)

    # swin_l1216_n = "swin_l12to16-hd"
    # swin_l1216_path = f"{BASE_DIR}/pascal/{swin_l1216_n}"
    # swin_l1216 = App(swin_l1216_n, swin_l1216_path, inj_per_fm, injections)

    # eva_b_n = "eva_base-hd"
    # eva_b_path = f"{BASE_DIR}/pascal/{eva_b_n}"
    # eva_b = App(eva_b_n, eva_b_path, inj_per_fm, injections)

    # eva_l_n = "eva_large-hd"
    # eva_l_path = f"{BASE_DIR}/pascal/{eva_l_n}"
    # eva_l = App(eva_l_n, eva_l_path, inj_per_fm, injections)

    maxvit_384_n = "maxvit_l384-hd"
    maxvit_384_path = f"{BASE_DIR}/pascal/{maxvit_384_n}"
    maxvit_384 = App(maxvit_384_n, maxvit_384_path, inj_per_fm, injections)

    maxvit_512_n = "maxvit_l512-hd"
    maxvit_512_path = f"{BASE_DIR}/pascal/{maxvit_512_n}"
    maxvit_512 = App(maxvit_512_n, maxvit_512_path, inj_per_fm, injections)

    vit14h_n = "vit14h-hd"
    vit14h_path = f"{BASE_DIR}/pascal/{vit14h_n}"
    vit14h = App(vit14h_n, vit14h_path, inj_per_fm, injections)

    apps = [
        # swin_b1216,
        # swin_b1224,
        # swin_l1216,
        # eva_b,
        # eva_l,
        maxvit_384,
        maxvit_512,
        vit14h,
    ]

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
        # df_stdout = parser.dict_to_dataframe(res_stdout)
        # df_stderr = parser.dict_to_dataframe(res_stderr)

        df_res.to_csv(f"{app.app_folder}/results_cat.csv")
        # df_stdout.to_csv(f"{app.app_folder}/results_stdout.csv")
        # df_stderr.to_csv(f"{app.app_folder}/results_stderr.csv")

        # df_kernels = pd.DataFrame.from_dict(res_kernels, orient="index")
        # df_kernels.to_csv(f"{app.app_folder}/results_kernel.csv")

        # df_kbfm = dict_to_dataframe(res_ker_bfm)
        # df_kbfm.to_csv(f"./{app.app_folder}/results_kernel_bfm.csv"
        output_pvf_path = f"{BASE_DIR}/pascal/pvf_by_models.csv"
        df_pvf_models = pd.DataFrame.from_dict(errs_by_model, orient="index")
          
        # issue on appending data to csv file
        # when check if file exists or not, it appends 
        # the same row multiple times
        #if not os.path.isfile(output_pvf_path):
        df_pvf_models.to_csv(output_pvf_path)
        #else:
        #    df_pvf_models.to_csv(output_pvf_path, mode='a', index=True, header=False)

        #for k in errs_by_model:
        #    sdc = errs_by_model[k]["SDC"]
        #    crit = errs_by_model[k]["Critical SDC"]
        #    due = errs_by_model[k]["DUE"]
        #    print(f"{k},{sdc},{crit},{due}")


if __name__ == "__main__":
    main()
