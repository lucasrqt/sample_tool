#!/usr/bin/python3

import pandas as pd
from configs_parser import *
from typing import List, Tuple
import os
import re
import linecache
import json


def init_parse_kernel_dict():
    # fmt: off
    otpt, cls, msk, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[MASKED_ERR], OT_STR[OTHER_ERR]
    return {otpt: 0, cls: 0, msk: 0, oth: 0}


def parse_per_bfm(
    groups: List[int],
    models: List[int],
    faults_per_fm: int,
    base_folder: str,
    app_name: str,
) -> Tuple[dict, dict]:
    """
    Traverse all results folders to parse the different results depending on the fault model
    """
    folder_base = f"{base_folder}/{app_name}"
    res_stdout, res_stderr = {}, {}

    # fmt: off
    otpt, cls, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[OTHER_ERR]

    for i, group in enumerate(groups):
        gp = IGID_STR[group]
        res_stdout[gp], res_stderr[gp] = {}, {}
        print(f"    [+] parsing group {gp}")
        for model in models[i]:
            md = EM_STR[model]
            res_stdout[gp][md] = {}
            crashes = 0
            print(" "*8 + f"• fault model: {md}")

            res_stdout[gp][md][otpt] = 0
            res_stdout[gp][md][cls] = 0
            res_stdout[gp][md][oth] = 0

            for injection in range(1, faults_per_fm + 1):
                dir = f"{folder_base}-group{group}-model{model}-icount{injection}"
                stdout_diff = f"{dir}/stdout_diff.log"
                stderr_diff = f"{dir}/stderr_diff.log"

                # STDOUT part
                if os.stat(stdout_diff).st_size > 0:                
                    output_errs, class_errs, other_errs = parse_stdout(
                        f"{dir}/stdout.txt"
                    )
                    
                    res_stdout[gp][md][otpt] += output_errs
                    res_stdout[gp][md][cls] += class_errs
                    res_stdout[gp][md][oth] += other_errs

                # STDERR part
                if os.stat(stderr_diff).st_size > 0:
                    crashes += 1

            res_stderr[gp][md] = {"crashes": crashes}

            res_stdout[gp][md][otpt] = res_stdout[gp][md][otpt] * 100 / faults_per_fm
            res_stdout[gp][md][cls] = res_stdout[gp][md][cls] * 100 / faults_per_fm
            res_stdout[gp][md][oth] = res_stdout[gp][md][oth] * 100 / faults_per_fm

    return (res_stdout, res_stderr)


def parse_per_kernel(
    groups: List[int],
    models: List[int],
    faults_per_fm: int,
    base_folder: str,
    app_name: str,
) -> Tuple[dict, dict]:
    """
    Traverse all results folders to parse the different results depending kernel call
    """
    folder_base = f"{base_folder}/{app_name}"
    kernels = {}

    # fmt: off
    otpt, cls, msk, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[MASKED_ERR], OT_STR[OTHER_ERR]

    for i, group in enumerate(groups):
        for model in models[i]:
            for injection in range(1, faults_per_fm + 1):
                dir = f"{folder_base}-group{group}-model{model}-icount{injection}"
                stdout_diff = f"{dir}/stdout_diff.log"
                stderr_diff = f"{dir}/stderr_diff.log"
                inj_info = f"{dir}/nvbitfi-injection-info.txt"

                kernel_name = linecache.getline(inj_info, 3).strip()
                if not kernel_name in kernels:
                    kernels[kernel_name] = init_parse_kernel_dict()

                # STDOUT part
                if os.stat(stdout_diff).st_size > 0:
                    output_errs, class_errs, other_errs = parse_stdout(
                        f"{dir}/stdout.txt"
                    )
                    kernels[kernel_name][otpt] += output_errs
                    kernels[kernel_name][cls] += class_errs
                    kernels[kernel_name][oth] += other_errs
                else:
                    kernels[kernel_name][msk] += 1

                # STDERR part
                if os.stat(stderr_diff).st_size > 0:
                    kernels[kernel_name][oth] += other_errs

    return kernels


# TODO
def parse_per_kernel_bfm(
    groups: List[int],
    models: List[int],
    faults_per_fm: int,
    base_folder: str,
    app_name: str,
) -> Tuple[dict, dict]:
    """
    Traverse all results folders to parse the different results depending kernel call
    """
    folder_base = f"{base_folder}/{app_name}"
    kernels = {}

    # fmt: off
    otpt, cls, msk, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[MASKED_ERR], OT_STR[OTHER_ERR]

    for i, group in enumerate(groups):
        gp = IGID_STR[group]
        for model in models[i]:
            md = EM_STR[model]
            for injection in range(1, faults_per_fm + 1):
                dir = f"{folder_base}-group{group}-model{model}-icount{injection}"
                stdout_diff = f"{dir}/stdout_diff.log"
                stderr_diff = f"{dir}/stderr_diff.log"
                inj_info = f"{dir}/nvbitfi-injection-info.txt"

                kernel_name = linecache.getline(inj_info, 3).strip()
                if not kernel_name in kernels:
                    kernels[kernel_name] = init_parse_kernel_dict()

                # STDOUT part
                if os.stat(stdout_diff).st_size > 0:
                    output_errs, class_errs, other_errs = parse_stdout(
                        f"{dir}/stdout.txt"
                    )
                    kernels[kernel_name][otpt] += output_errs
                    kernels[kernel_name][cls] += class_errs
                    kernels[kernel_name][oth] += other_errs
                else:
                    kernels[kernel_name][msk] += 1

                # STDERR part
                if os.stat(stderr_diff).st_size > 0:
                    kernels[kernel_name][oth] += other_errs

    return kernels


def parse_stdout(path: str) -> Tuple[int, int, int]:
    """
    Analyze the stdout.txt file to get the number of output errors and classification errors
    """
    with open(path) as file:
        output_errs, class_errs, other_errs = 0, 0, 0

        # only extract digits surrounded by boundaries
        output = re.findall(r"\b\d+\b", file.read())

        # getting output results
        if output:
            output = [int(x) for x in output]
            output_errs += output[OUTPUT_ERR]
            class_errs += output[CLASS_ERR]
        # if there's no value it means that there was an other error
        else:
            other_errs += 1

    return output_errs, class_errs, other_errs


def dict_to_dataframe(results: dict) -> pd.DataFrame:
    """
    getting the right format for the pandas dataframe
    """
    return pd.DataFrame.from_dict(
        {(i, j): results[i][j] for i in results.keys() for j in results[i].keys()},
        orient="index",
    )


def main():
    results_folder_path = "sample_tool"
    faults_per_fm = 100
    injections = {
        G_FP32: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
        G_GP: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE],
    }
    app = "sample-tool"

    groups, models = list(injections.keys()), list(injections.values())

    print(f" [+] parsing results...")
    res_stdout, res_stderr = parse_per_bfm(
        groups, models, faults_per_fm, results_folder_path, app
    )

    print(f" [+] parsing results for kernel")
    res_kernels = parse_per_kernel(
        groups, models, faults_per_fm, results_folder_path, app
    )

    df_stdout = dict_to_dataframe(res_stdout)
    df_stderr = dict_to_dataframe(res_stderr)
    df_stdout.to_csv(f"./{results_folder_path}/results_stdout.csv")
    df_stderr.to_csv(f"./{results_folder_path}/results_stderr.csv")

    df_kernels = pd.DataFrame.from_dict(res_kernels, orient="index")
    df_kernels.to_csv(f"./{results_folder_path}/results_kernel.csv")


if __name__ == "__main__":
    main()
