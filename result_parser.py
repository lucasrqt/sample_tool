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

    for i, group in enumerate(groups):
        res_stdout[group] = {}
        print(f" [+] parsing group {IGID_STR[group]}")
        for model in models[i]:
            res_stdout[group][model] = [0] * 3
            print(f"\t• fault model: {EM_STR[model]}")
            for injection in range(1, faults_per_fm + 1):
                dir = f"{folder_base}-group{group}-model{model}-icount{injection}"
                stdout_diff = f"{dir}/stdout_diff.log"
                stderr_diff = f"{dir}/stderr_diff.log"

                # STDOUT part
                if os.stat(stdout_diff).st_size > 0:
                    output_errs, class_errs, other_errs = parse_stdout(
                        f"{dir}/stdout.txt"
                    )
                    res_stdout[group][model][OUTPUT_ERR] += output_errs
                    res_stdout[group][model][CLASS_ERR] += class_errs
                    res_stdout[group][model][OTHER_ERR] += other_errs

                # STDERR part
                # TODO: same for stderr

    return (res_stdout, res_stderr)


def parse_stdout(path: str) -> Tuple[int, int, int]:
    """
    Analyze the stdout.txt file to get the number of output errors and classification errors
    """
    with open(path) as file:
        output_errs, class_errs, other_errs = 0, 0, 0
        output = re.findall(r"\b\d+\b", file.read())

        if output:
            output = [int(x) for x in output]
            output_errs += output[OUTPUT_ERR]
            class_errs += output[CLASS_ERR]
        else:
            other_errs += 1

    return output_errs, class_errs, other_errs


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
    print(res_stdout)
    df = pd.DataFrame(res_stdout)
    df.to_csv("./results_stdout.csv", index=False)


if __name__ == "__main__":
    main()
