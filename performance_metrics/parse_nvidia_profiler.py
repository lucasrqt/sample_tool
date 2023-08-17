#!/usr/bin/python3
import os
import re

import pandas as pd

import common
import configs
import profiler_class

DEFAULT_LOG = str(os.path.basename(__file__)).upper().replace(".PY", "")

ORIGINAL, SWINV2, EVA, MAXVIT = "Original", "SwinV2", "EVA", "MaxViT"
FAMILY = dict(
    **{k: ORIGINAL for k in
       [configs.VIT_BASE_PATCH16_224, configs.VIT_BASE_PATCH16_384, configs.VIT_BASE_PATCH32_224_SAM,
        configs.VIT_LARGE_PATCH14_CLIP_224, configs.VIT_HUGE_PATCH14_CLIP_224]},
    **{k: SWINV2 for k in
       [configs.SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K, configs.SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K,
        configs.SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K, configs.SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K]},
    **{k: EVA for k in
       [configs.EVA_BASE_PATCH14_448_MIM, configs.EVA_LARGE_PATCH14_448_MIM, configs.EVA_SMALL_PATCH14_448_MIN]},
    **{k: MAXVIT for k in [configs.MAXVIT_LARGE_TF_384, configs.MAXVIT_LARGE_TF_512]},
)


def get_pretty_name(model_name, family):
    model_name = model_name.upper()
    if family == ORIGINAL:
        m = re.match(r"VIT_(\S+)_PATCH(\d+)[_\S+]*_(\d+)", model_name)
        if "SAM" in model_name:
            return f"{m.group(1)[0]}S{m.group(2)}-{m.group(3)}"
        return f"{m.group(1)[0]}{m.group(2)}-{m.group(3)}"
    elif family == SWINV2:
        m = re.match(r"SWINV2_(\S+)_WINDOW\d+TO\d+_192TO(\d+)", model_name)
        return f"{m.group(1)[0]}-{m.group(2)}"
    elif family == EVA:
        m = re.match(r"EVA02_(\S+)_PATCH(\d+)_(\d+)", model_name)
        if m is None: print(model_name)
        return f"{m.group(1)[0]}{m.group(2)}-{m.group(3)}"
    elif family == MAXVIT:
        m = re.match(r"MAXVIT_(\S+)_TF_(\d+).IN21K_FT_IN1K", model_name)
        return f"{m.group(1)[0]}-{m.group(2)}"
    else:
        raise ValueError(f"Not a valid model:{model_name}")


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


def parse_nsight_metrics(csv_path):
    df = read_csv(csv_path=csv_path)
    line_dict = dict()
    # filter_metrics = [y for x in common.METRICS_NSIGHT_CLI.values() if x for y in x]
    # df["weights"] = df["Kernel Name"].apply(lambda x: kernel_time_weights.loc[x]).astype(float)
    # df = df[df["Metric Name"].isin(filter_metrics)]
    df["Metric Value"] = df["Metric Value"].astype(float)
    # It's possible to sum quantitative metrics
    for nvprof_metric, nsight_set in common.QUANTITATIVE_METRICS_NSIGHT_CLI.items():
        line_dict[nvprof_metric] = df[df["Metric Name"].isin(nsight_set)].sum()["Metric Value"]

    # For performance get the highest
    for nvprof_metric, nsight_set in common.PERFORMANCE_METRICS_NSIGHT_CLI.items():
        metric_df = df[df["Metric Name"].isin(nsight_set)]
        # Divide by 100 if it is a percentage. Same as nvprof
        metric_value = metric_df.apply(
            lambda r: (float(r["Metric Value"]) / 100.0 if r["Metric Unit"] == "%" else float(r["Metric Value"])),
            axis="columns")
        line_dict[nvprof_metric] = metric_value.max()
    # Occupancy and ipc, it's necessary to calculate it separately
    # occupancy_df = df[df["Metric Name"].isin(common.PERFORMANCE_METRICS_NSIGHT_CLI["achieved_occupancy"])].copy()
    # ipc_df = df[df["Metric Name"].isin(common.PERFORMANCE_METRICS_NSIGHT_CLI["ipc"])]
    #
    # # Divide by 100 if it is a percentage. Same as nvprof
    # occupancy_df["Metric Value"] = occupancy_df.apply(
    #     lambda r: (float(r["Metric Value"]) / 100 if r["Metric Unit"] == "%" else float(r["Metric Value"])),
    #     axis="columns")

    # occupancy_df["wt_mean"] = occupancy_df["weights"] * occupancy_df["Metric Value"]
    # Mean
    # line_dict["achieved_occupancy"] = occupancy_df["wt_mean"].mean()
    # line_dict["ipc"] = ipc_df["wt_mean"].mean()
    # line_dict["achieved_occupancy"] = occupancy_df["Metric Value"].max()
    # line_dict["ipc"] = ipc_df["Metric Value"].max()

    return line_dict


def parse_nvprof_metrics(csv_path):
    df = read_csv(csv_path=csv_path)
    metric_dict = df[df["Metric Name"].isin(common.QUANTITATIVE_METRICS_NSIGHT_CLI.keys())].copy()
    metric_dict["Avg"] = metric_dict["Avg"].astype(float) * metric_dict["Invocations"]
    metric_dict = metric_dict[["Metric Name", "Avg"]].groupby("Metric Name").sum().sum(axis=1)
    line_dict = metric_dict.to_dict()
    for key in common.PERFORMANCE_METRICS_NSIGHT_CLI:
        # metric_df = df[df["Metric Name"] == key]
        # line_dict[key] = occ_and_ipc["wt_mean"].mean()
        line_dict[key] = df[df["Metric Name"] == key]["Max"].max()

    return line_dict


def parse_nsight_memory(csv_path):
    df = read_csv(csv_path=csv_path)
    # df["weights"] = df["Kernel Name"].apply(lambda x: kernel_time_weights.loc[x]).astype(float)
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
        # memory_dict[name] = (df_mem["Metric Value"].astype(float) * df_mem["weights"]).mean()
        # memory_dict[f"{name}_mean"] = df_mem["Metric Value"].astype(float).mean()
        memory_dict[f"{name}_max"] = df_mem["Metric Value"].astype(float).max()
        # memory_dict[f"{name}_min"] = df_mem["Metric Value"].astype(float).min()

    return memory_dict


def parse_nvprof_memory(csv_path):
    df = read_csv(csv_path=csv_path)
    metric_units = df.iloc[0]
    dfi = df.iloc[1:]
    memory_dict = dict()
    dfi = dfi[~dfi.Name.str.match(r"\[CUDA.*\]")]
    memory_metric = {"Static SMem": "static_shared_per_block", "Dynamic SMem": "dynamic_shared_per_block",
                     "Registers Per Thread": "rf"}
    # dfi["weights"] = dfi["Name"].apply(lambda x: kernel_time_weights.loc[x]).astype(float)
    for key, name in memory_metric.items():
        converter = 1.0
        if metric_units[key] == "KB":
            converter = 1024
        elif metric_units[key] == "B":
            converter = 1.0
        elif key != "Registers Per Thread":
            raise ValueError(f"{metric_units}")
        # RF must be evaluated by the lifecycle of the variables
        # memory_dict[name] = ((dfi[key].astype(float) * converter) * dfi["weights"]).mean()
        # memory_dict[f"{name}_mean"] = (dfi[key].astype(float) * converter).mean()
        memory_dict[f"{name}_max"] = (dfi[key].astype(float) * converter).max()
        # memory_dict[f"{name}_min"] = (dfi[key].astype(float) * converter).min()

        # memory_dict[name] = dfi[key].astype(float).max() * converter

    return memory_dict


def parse_nvprof_events(csv_path):
    df = read_csv(csv_path=csv_path)
    df["Total"] = df["Total"].astype(float)
    metric_dict = df[df["Event Name"].isin(common.EVENTS_NSIGHT_CLI.keys())]
    metric_dict = metric_dict[["Event Name", "Total"]].groupby("Event Name").sum().sum(axis=1)
    return metric_dict.to_dict()


def parse_nsight_events(csv_path):
    df = read_csv(csv_path=csv_path)
    df["Metric Value"] = df["Metric Value"].astype(float)
    line_dict = {
        nvprof_metric: df[df["Metric Name"].isin(nsight_set)].sum()["Metric Value"]
        for nvprof_metric, nsight_set in common.EVENTS_NSIGHT_CLI.items()
    }
    return line_dict


def parse_naive_execution_time_profiling(csv_path):
    with open(csv_path) as fp:
        for line in fp:
            m = re.match(r"PERF_MEASURE::model:\S+ hardening:\S+ iterations:\S+ it_time:(\S+) shape cpu:.*", line)
            if m:
                return float(m.group(1))
    raise ValueError


def main():
    logger = common.create_logger(default_log=DEFAULT_LOG)

    list_final_metrics = list()
    # Select which boards to parse
    boards = {
        "pascal": "QuadroP2200",
        # "volta": "NVIDIATITANV",
        "ampere": "NVIDIAGeForceRTX3060Ti"
    }
    boards_obj = {
        "pascal": profiler_class.ProfilerNvprof,
        **{b: profiler_class.ProfilerNsight for b in ["volta", "ampere"]}
    }

    parser_functions: callable = {
        "pascal": {
            "metric": parse_nvprof_metrics,
            # "time": parse_nvprof_time,
            "memory": parse_nvprof_memory,
            "events": parse_nvprof_events
        },
        **{b: {
            "metric": parse_nsight_metrics,
            # "time": parse_nsight_time,
            "memory": parse_nsight_memory,
            "events": parse_nsight_events
        } for b in ["volta", "ampere"]},
    }

    for board, board_name in boards.items():
        for model_name in common.BENCHMARKS:
            for hardening in ["replace-id", "no-replace-id"]:
                new_model_name = f"{model_name}_{hardening}"
                parse_metrics = parser_functions[board]["metric"]
                # parse_time = parser_functions[board]["time"]
                parse_events = parser_functions[board]["events"]
                parse_memory = parser_functions[board]["memory"]
                profiler_obj = boards_obj[board](
                    execute_parameters="", app_dir="", app=common.APP_NAME, metrics="", events="",
                    cuda_version=common.CUDA_VERSION, log_base_path=common.PROFILE_DATA_PATH, model=new_model_name,
                    board=board_name, logger=logger
                )
                csv_time_path = profiler_obj.get_log_name(target="time")
                # # Parse the time ---------------------------------------------------------------------------
                # execution_time, kernel_time_weights = parse_time(csv_path=csv_time_path)
                execution_time = parse_naive_execution_time_profiling(csv_path=csv_time_path)
                # Parse the Metrics ------------------------------------------------------------------------
                csv_metrics_path = profiler_obj.get_log_name(target="metrics")
                metrics_dict = parse_metrics(csv_path=csv_metrics_path)
                # Parse the memory -------------------------------------------------------------------------
                csv_memory_path = profiler_obj.get_log_name(target="memory")
                memory_dict = parse_memory(csv_path=csv_memory_path)
                # Parse the events -------------------------------------------------------------------------
                csv_events_path = profiler_obj.get_log_name(target="events")
                events_dict = parse_events(csv_path=csv_events_path)
                family = FAMILY[model_name]
                line_dict = {
                    "nvcc_version": common.CUDA_VERSION, "board": board, "app": model_name, "hardening": hardening,
                    "family": family, "config": get_pretty_name(model_name=model_name, family=family),
                    "execution_time": execution_time, **metrics_dict, **memory_dict, **events_dict
                }
                list_final_metrics.append(line_dict)

    final_df = pd.DataFrame(list_final_metrics)
    print(final_df)
    final_df.to_csv(common.FINAL_PROFILE_DATABASE, index=False)


if __name__ == '__main__':
    main()
