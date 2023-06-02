#!/usr/bin/python3

import pandas as pd
from configs_parser import *
from typing import List, Tuple
import os
import re


def parse_results(
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

    return (res_stdout, res_stderr)


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
    res_stdout, res_stderr = parse_results(
        groups, models, faults_per_fm, results_folder_path, app
    )
    df_stdout = dict_to_dataframe(res_stdout)
    df_stderr = dict_to_dataframe(res_stderr)
    df_stdout.to_csv("./results_stdout.csv")
    df_stderr.to_csv("./results_stderr.csv")


if __name__ == "__main__":
    main()
