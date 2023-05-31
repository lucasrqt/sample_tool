#!/usr/bin/python3

import pandas as pd


if __name__ == '__main__':
    results = pd.read_csv("./tsv-files/results_inst_value_10000_NVBitFI_details.tsv", sep="\t",)
    print(f" [+] parsing results...\n{results}")
    