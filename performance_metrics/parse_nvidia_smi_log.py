#!/usr/bin/python3
import pandas as pd

CSV_FILE = "../data/performance_metrics/nvidia_smi_data.csv"


def main():
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
    # 7.20 W, 139 MHz, 405 MHz, 139 MHz
    df["power.draw [W]"] = df["power.draw [W]"].str.replace(" W", "").astype(float)
    df["clocks.current.sm [MHz]"] = df["clocks.current.sm [MHz]"].str.replace(" MHz", "").astype(float)
    df["clocks.current.memory [MHz]"] = df["clocks.current.memory [MHz]"].str.replace(" MHz", "").astype(float)
    df["clocks.current.graphics [MHz]"] = df["clocks.current.graphics [MHz]"].str.replace(" MHz", "").astype(float)
    describe = df.describe()
    print(describe)


if __name__ == '__main__':
    main()
