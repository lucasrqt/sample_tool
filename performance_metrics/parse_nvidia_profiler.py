#!/usr/bin/python3
import os.path
import re

import pandas as pd

import common
import profiler_class
import profiler_flags

VOLTA_THREAD_PER_WARP = 32
BENCHMARKS = {

}


def read_csv(csv_path):
    # Find the first line of the valid data
    ignore_warnings = ["Warning: One or more events or metrics overflowed.",
                       "Warning: One or more events or metrics can't be profiled.",
                       "No API activities were profiled", "noinputensurance"]
    lines_to_skip = list()
    try:
        with open(csv_path) as fp:
            lines = fp.readlines()
            first_line = 0
            for i, line in enumerate(lines):
                if re.match(r"==(\d+)== (\S+) result:", line) or "==PROF==" in line:
                    first_line = i + 1
                if any([to_ignore in line for to_ignore in ignore_warnings]):
                    lines_to_skip.append(i)

        # Skip the useless rows
        lines_to_skip = list(range(first_line)) + lines_to_skip
        df = pd.read_csv(csv_path, skiprows=lines_to_skip)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(csv_path)
        print(lines_to_skip)
        print(first_line)
        raise
    return df


def convert_metric_unit_nsight(row):
    metric_value = row["Metric Value"]
    if type(metric_value) != float:
        try:
            metric_value = float(metric_value)
        except ValueError:
            metric_value = float(metric_value.replace(".", "").replace(",", "."))

    unit = row["Metric Unit"]
    if unit == "nsecond":
        row["Metric Value"] = metric_value / 1e9
    elif unit == "%":
        row["Metric Value"] = metric_value / 100.0
    elif unit == "KB":
        row["Metric Value"] = metric_value * 1024
    elif any([unit == i for i in ["byte", "byte/block", "register/thread", "byte/second", "cycle", "cycle/second"]]):
        row["Metric Value"] = metric_value
    else:
        raise ValueError(f"Invalid unit {unit}")
    return row


def parse_nsight_time(csv_path):
    df = read_csv(csv_path=csv_path)
    df_time = df[df["Metric Name"] == "gpu__time_active.avg"]
    df_time = df_time.apply(convert_metric_unit_nsight, axis="columns")
    kernel_time_weight = df_time[["Metric Value", "Kernel Name"]].groupby(["Kernel Name"]).sum()
    kernel_time_weight /= kernel_time_weight.sum()
    assert kernel_time_weight.shape[0] == df_time["Kernel Name"].unique().shape[0], str(df_time)
    execution_time = df_time["Metric Value"].sum()
    return execution_time, kernel_time_weight


def parse_nvprof_time(csv_path):
    df = read_csv(csv_path=csv_path)
    trace_config = df.iloc[0]
    filtered_df = df.iloc[1:]
    # Remove the API calls
    filtered_df = filtered_df[~filtered_df.Name.str.match(r"\[CUDA.*\]")]
    avg = filtered_df["Time"].astype(float)
    unit = trace_config["Time"]
    if unit == "ms":
        avg /= 1e3
    elif unit == "us":
        avg /= 1e6
    elif unit != "s":
        raise ValueError(f"Invalid unit {unit}")
    # Calc the percentage of the time by kernel
    kernel_time_weight = filtered_df[["Max", "Name"]].groupby(["Name"]).sum()
    kernel_time_weight["Max"] = kernel_time_weight["Max"].astype(float)
    kernel_time_weight /= kernel_time_weight.sum()
    assert filtered_df.shape[0] == kernel_time_weight.shape[0], f"\n{kernel_time_weight}\n{filtered_df}"
    return avg.sum(), kernel_time_weight


def parse_nsight_metrics(csv_path, kernel_time_weights):
    df = read_csv(csv_path=csv_path)
    line_dict = dict()
    filter_metrics = [y for x in profiler_flags.METRICS_NSIGHT_CLI_SM70.values() if x for y in x]
    # df["weights"] = df["Kernel Name"].apply(lambda x: kernel_time_weights.loc[x]).astype(float)
    df = df[df["Metric Name"].isin(filter_metrics)]

    for nvprof_metric, nsight_set in profiler_flags.QUANTITATIVE_METRICS_NSIGHT_CLI.items():
        # sum_of_metrics = 0
        # if nsight_set:
        # # UNCOMMENT TO CHECK ANY PROBLEM
        # tmp = df_metrics[df_metrics["Metric Name"].isin(nsight_set)]
        # if (tmp["Metric Unit"] != "inst").any() and (tmp["Metric Value"] != 0).any():
        #     print(tmp)
        # sum_of_metrics = df[df["Metric Name"].isin(nsight_set)].sum()["Metric Value"]
        line_dict[nvprof_metric] = df[df["Metric Name"].isin(nsight_set)].sum()["Metric Value"] if nsight_set else 0

    # Occupancy and ipc
    occupancy_df = df[
        df["Metric Name"].isin(profiler_flags.PERFORMANCE_METRICS_NSIGHT_CLI["achieved_occupancy"])].copy()
    ipc_df = df[df["Metric Name"].isin(profiler_flags.PERFORMANCE_METRICS_NSIGHT_CLI["ipc"])]

    # Divide by 100 if it is a percentage. Same as nvprof
    occupancy_df["Metric Value"] = occupancy_df.apply(
        lambda r: (float(r["Metric Value"]) / 100 if r["Metric Unit"] == "%" else float(r["Metric Value"])),
        axis="columns")

    # TODO: Ask Paolo if IP and OCC are wt_mean vs mean vs max
    # occupancy_df["wt_mean"] = occupancy_df["weights"] * occupancy_df["Metric Value"]
    # Mean
    # line_dict["achieved_occupancy"] = occupancy_df["wt_mean"].mean()
    # line_dict["ipc"] = ipc_df["wt_mean"].mean()
    line_dict["achieved_occupancy"] = occupancy_df["Metric Value"].max()
    line_dict["ipc"] = ipc_df["Metric Value"].max()

    return line_dict


def parse_nvprof_metrics(csv_path, kernel_time_weights):
    df = read_csv(csv_path=csv_path)
    metric_dict = df[df["Metric Name"].isin(profiler_flags.QUANTITATIVE_METRICS_NSIGHT_CLI.keys())]
    metric_dict = metric_dict[["Metric Name", "Avg"]].groupby("Metric Name").sum().sum(axis=1)
    line_dict = metric_dict.to_dict()
    # df["weights"] = df["Kernel"].apply(lambda x: kernel_time_weights.loc[x]).astype(float)
    df = df[df["Metric Name"].isin(["ipc", "achieved_occupancy"])].copy()
    # df["wt_mean"] = df["weights"] * df["Max"].astype(float)

    for key in ["ipc", "achieved_occupancy"]:
        occ_and_ipc = df[df["Metric Name"] == key]
        # line_dict[key] = occ_and_ipc["wt_mean"].mean()
        line_dict[key] = occ_and_ipc["Max"].max()

    return line_dict


def parse_nsight_memory(csv_path, kernel_time_weights):
    df = read_csv(csv_path=csv_path)
    df["weights"] = df["Kernel Name"].apply(lambda x: kernel_time_weights.loc[x]).astype(float)
    memory_dict = dict()
    # "Shared Memory Configuration Size," "Registers Per Thread"
    memory_metric = {"Dynamic Shared Memory Per Block": "dynamic_shared_per_block",
                     "Static Shared Memory Per Block": "static_shared_per_block",
                     "Shared Memory Configuration Size": "shared", "Registers Per Thread": "rf"}
    for key, name in memory_metric.items():
        df_mem = df[df["Metric Name"] == key]
        df_mem = df_mem.apply(convert_metric_unit_nsight, axis="columns")
        # RF must be evaluated by the lifecycle of the variables
        # memory_dict[name] = df_mem.loc[df["Metric Name"] == key, "Metric Value"].max()
        memory_dict[name] = (df_mem["Metric Value"].astype(float) * df_mem["weights"]).mean()
    return memory_dict


def parse_nvprof_memory(csv_path, kernel_time_weights):
    df = read_csv(csv_path=csv_path)
    metric_units = df.iloc[0]
    dfi = df.iloc[1:]
    memory_dict = dict()
    dfi = dfi[~dfi.Name.str.match(r"\[CUDA.*\]")]
    memory_metric = {"Static SMem": "static_shared_per_block", "Dynamic SMem": "dynamic_shared_per_block",
                     "Registers Per Thread": "rf"}
    dfi["weights"] = dfi["Name"].apply(lambda x: kernel_time_weights.loc[x]).astype(float)
    for key, name in memory_metric.items():
        converter = 1.0
        if metric_units[key] == "KB":
            converter = 1024
        elif metric_units[key] == "B":
            converter = 1.0
        elif key != "Registers Per Thread":
            raise ValueError(f"{metric_units}")
        # RF must be evaluated by the lifecycle of the variables
        memory_dict[name] = ((dfi[key].astype(float) * converter) * dfi["weights"]).mean()
        # memory_dict[name] = dfi[key].astype(float).max() * converter

    return memory_dict


def main():
    list_final_metrics = list()
    boards = {"kepler": "NVIDIATeslaK40c", "volta": "NVIDIATITANV", "ampere": "NVIDIAGeForceRTX3060Ti"}
    boards_obj = {"kepler": profiler_class.ProfilerNvprof, "volta": profiler_class.ProfilerNsight}
    boards_obj["ampere"] = boards_obj["volta"]
    parser_functions = {
        "kepler": {"metric": parse_nvprof_metrics, "time": parse_nvprof_time, "memory": parse_nvprof_memory},
        "volta": {"metric": parse_nsight_metrics, "time": parse_nsight_time, "memory": parse_nsight_memory},
    }
    parser_functions["ampere"] = parser_functions["volta"]
    for board, board_name in boards.items():
        board_path = f"{common.PROFILE_DATA_PATH}/{board}_profile"
        for app in BENCHMARKS:
            parse_metrics = parser_functions[board]["metric"]
            parse_time = parser_functions[board]["time"]
            parse_memory = parser_functions[board]["memory"]
            object_profiler = boards_obj[board](make_parameters="", app_dir="", app=app, metrics="",
                                                log_base_path=board_path, board=board_name)
            csv_time_path = object_profiler.get_log_name(target="time")
            check_time = os.path.isfile(csv_time_path)
            # IF time exists, the others also exist
            if check_time:
                # Parse the time ---------------------------------------------------------------------------
                execution_time, kernel_time_weights = parse_time(csv_path=csv_time_path)
                # Parse the Metrics ------------------------------------------------------------------------
                csv_metrics_path = object_profiler.get_log_name(target="metrics")
                metrics_dict = parse_metrics(csv_path=csv_metrics_path,
                                             kernel_time_weights=kernel_time_weights)
                # Parse the memory -------------------------------------------------------------------------
                csv_memory_path = object_profiler.get_log_name(target="memory")
                memory_dict = parse_memory(csv_path=csv_memory_path,
                                           kernel_time_weights=kernel_time_weights)
                line_dict = {"board": board, "app": app, "nvcc_version": "11.7",
                             "execution_time": execution_time, **metrics_dict, **memory_dict}
                list_final_metrics.append(line_dict)
            else:
                print("NOT NEEDED", csv_time_path)

    final_df = pd.DataFrame(list_final_metrics)
    print(final_df)
    final_df.to_csv(common.FINAL_PROFILE_DATABASE, index=False)


if __name__ == '__main__':
    main()