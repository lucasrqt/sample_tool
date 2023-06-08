from configs_parser import *
import pandas as pd
from typing import Tuple, List
import re
import linecache


class App:
    def __init__(
        self,
        app_name: str,
        app_foler: str,
        flt_p_fm: int,
        injections: dict,
    ) -> None:
        self.app_name = app_name
        self.app_folder = app_foler
        self.flt_p_fm = flt_p_fm
        self.injections = injections
        self.groups = list(injections.keys())
        self.bfms = list(injections.values())

    def __str__(self) -> str:
        return f" App: {self.app_name}\n Faults per fault model: {self.flt_p_fm}\n Faults:\n{self.__print_inj()}"

    def __print_inj(self) -> str:
        res = " {"
        for i, group in enumerate(self.groups):
            bfms_str = f"[{EM_STR[self.bfms[i][0]]}"
            for bfm in self.bfms[i][1:]:
                bfms_str += f", {EM_STR[bfm]}"
            bfms_str += "]"
            res += f"\n    {IGID_STR[group]}: {bfms_str}"
        res += "\n }"

        return res


class Parser:
    @staticmethod
    def dict_to_dataframe(results: dict) -> pd.DataFrame:
        """
        getting the right format for the pandas dataframe
        """
        return pd.DataFrame.from_dict(
            {(i, j): results[i][j] for i in results.keys() for j in results[i].keys()},
            orient="index",
        )

    def __parse_stdout(self, path: str) -> Tuple[int, int, int]:
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

    def __init_parse_kernel_dict(self):
        # fmt: off
        otpt, cls, msk, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[MASKED_ERR], OT_STR[OTHER_ERR]
        return {otpt: 0, cls: 0, msk: 0, oth: 0}

    def parse_per_bfm(self, app: App) -> Tuple[dict, dict]:
        """
        Traverse all results folders to parse the different results depending on the fault model
        """
        folder_base = f"{app.app_folder}/{app.app_name}"
        res_stdout, res_stderr = {}, {}

        # fmt: off
        otpt, cls, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[OTHER_ERR]

        for i, group in enumerate(app.groups):
            gp = IGID_STR[group]
            res_stdout[gp], res_stderr[gp] = {}, {}
            print(f"    [+] parsing group {gp}")
            for model in app.bfms[i]:
                md = EM_STR[model]
                res_stdout[gp][md] = {}
                crashes = 0
                print(" "*8 + f"• fault model: {md}")

                res_stdout[gp][md][otpt] = 0
                res_stdout[gp][md][cls] = 0
                res_stdout[gp][md][oth] = 0

                for injection in range(1, app.flt_p_fm + 1):
                    dir = f"{folder_base}-group{group}-model{model}-icount{injection}"
                    stdout_diff = f"{dir}/stdout_diff.log"
                    stderr_diff = f"{dir}/stderr_diff.log"

                    # STDOUT part
                    try:
                        if os.stat(stdout_diff).st_size > 0:                
                            output_errs, class_errs, other_errs = self.__parse_stdout(
                                f"{dir}/stdout.txt"
                            )
                            
                            res_stdout[gp][md][otpt] += output_errs
                            res_stdout[gp][md][cls] += class_errs
                            res_stdout[gp][md][oth] += other_errs
                    except:
                        crashes+=1

                    # STDERR part
                    try:
                        if os.stat(stderr_diff).st_size > 0:
                            crashes += 1
                    except:
                        crashes += 1

                res_stderr[gp][md] = {"crashes": crashes}

                res_stdout[gp][md][otpt] = res_stdout[gp][md][otpt] * 100 / app.flt_p_fm
                res_stdout[gp][md][cls] = res_stdout[gp][md][cls] * 100 / app.flt_p_fm
                res_stdout[gp][md][oth] = res_stdout[gp][md][oth] * 100 / app.flt_p_fm

        return (res_stdout, res_stderr)

    def parse_per_kernel(self, app: App) -> Tuple[dict, dict]:
        """
        Traverse all results folders to parse the different results depending kernel call
        """
        folder_base = f"{app.app_folder}/{app.app_name}"
        kernels = {}

        # fmt: off
        otpt, cls, msk, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[MASKED_ERR], OT_STR[OTHER_ERR]

        for i, group in enumerate(app.groups):
            for model in app.bfms[i]:
                for injection in range(1, app.flt_p_fm + 1):
                    dir = f"{folder_base}-group{group}-model{model}-icount{injection}"
                    stdout_diff = f"{dir}/stdout_diff.log"
                    stderr_diff = f"{dir}/stderr_diff.log"
                    inj_info = f"{dir}/nvbitfi-injection-info.txt"

                    kernel_name = linecache.getline(inj_info, 3).strip()
                    if not kernel_name in kernels:
                        kernels[kernel_name] = self.__init_parse_kernel_dict()

                    # STDOUT part
                    try:
                        if os.stat(stdout_diff).st_size > 0:
                            output_errs, class_errs, other_errs = self.__parse_stdout(
                                f"{dir}/stdout.txt"
                            )
                            kernels[kernel_name][otpt] += output_errs
                            kernels[kernel_name][cls] += class_errs
                            kernels[kernel_name][oth] += other_errs
                        else:
                            kernels[kernel_name][msk] += 1
                    except:
                        kernels[kernel_name][oth] += 1

                    # STDERR part
                    try:
                        if os.stat(stderr_diff).st_size > 0:
                            kernels[kernel_name][oth] += 1
                    except:
                        kernels[kernel_name][oth] += 1

        return kernels

    def parse_per_kernel_bfm(self, app: App) -> Tuple[dict, dict]:
        """
        Traverse all results folders to parse the different results depending kernel call
        """
        folder_base = f"{app.app_folder}/{app.app_name}"
        kernels_per_bfm = {}

        # fmt: off
        otpt, cls, msk, oth, = OT_STR[OUTPUT_ERR], OT_STR[CLASS_ERR], OT_STR[MASKED_ERR], OT_STR[OTHER_ERR]

        for i, group in enumerate(app.groups):
            gp = IGID_STR[group]
            group_kernels = {}
            for model in app.bfms[i]:
                md = EM_STR[model]
                kernel_bfm = {}
                for injection in range(1, app.flt_p_fm + 1):
                    dir = f"{folder_base}-group{group}-model{model}-icount{injection}"
                    stdout_diff = f"{dir}/stdout_diff.log"
                    stderr_diff = f"{dir}/stderr_diff.log"
                    inj_info = f"{dir}/nvbitfi-injection-info.txt"

                    kernel_name = linecache.getline(inj_info, 3).strip()
                    if not kernel_name in kernel_bfm:
                        kernel_bfm[kernel_name] = {}

                    if not md in kernel_bfm[kernel_name]:
                        kernel_bfm[kernel_name][md] = self.__init_parse_kernel_dict()

                    # STDOUT part
                    try:
                        if os.stat(stdout_diff).st_size > 0:
                            output_errs, class_errs, other_errs = self.__parse_stdout(
                                f"{dir}/stdout.txt"
                            )
                            kernel_bfm[kernel_name][md][otpt] += output_errs
                            kernel_bfm[kernel_name][md][cls] += class_errs
                            kernel_bfm[kernel_name][md][oth] += other_errs
                        else:
                            kernel_bfm[kernel_name][md][msk] += 1
                    except:
                        kernel_bfm[kernel_name][md][oth] += 1

                    # STDERR part
                    try:
                        if os.stat(stderr_diff).st_size > 0:
                            kernel_bfm[kernel_name][md][oth] += other_errs
                    except:
                        kernel_bfm[kernel_name][md][oth] += 1

                if gp in kernels_per_bfm:
                    kernels_per_bfm[gp] |= kernel_bfm
                else:
                    kernels_per_bfm[gp] = kernel_bfm

        return kernels_per_bfm
