#!/usr/bin/python3
import datetime
import logging
import os
import sys
import time

import common
import profiler_class

DEFAULT_LOG = str(os.path.basename(__file__)).upper().replace(".PY", "")


def clean_last_profile(app_binary, logger):
    # Clean the last profiling
    for to_clean in [app_binary, "nvprof", "nv-nsight-cu-cli", "nvcc"]:
        common.execute_cmd(f"if pgrep  {to_clean}; then pkill  {to_clean}; fi", logger=logger)
    common.execute_cmd("sync", logger=logger)


def profile_all():
    # create logger
    logger = logging.getLogger(DEFAULT_LOG)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Check which GPU I'm executing
    gpu_name = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().strip()
    # The files will be saved on this repository
    if not os.path.isdir(common.PROFILE_DATA_PATH):
        os.mkdir(common.PROFILE_DATA_PATH)
    logger.debug(f"GPU:{gpu_name}")

    tic = time.time()
    # Loop through each benchmark
    try:
        for model_name, app_parameters in common.BENCHMARKS.items():
            # Check if it is empty and the GPU is supported
            assert gpu_name in app_parameters["SUPPORTED_GPUS"], f"{gpu_name} not supported"
            # Extract the binary name and the default SASS code
            app_binary = app_parameters["APP_BIN"]
            for hardening in ["replace-id", "no-replace-id"]:
                exec_parameters = app_parameters["EXEC_PARAMETERS"] + f" --{hardening}"
                new_model_name = f"{model_name}_{hardening}"
                logger.debug(f"Profiling for the {new_model_name}")
                profiler = profiler_class.ProfilerNsight if gpu_name in common.NSIGHT_GPUS else profiler_class.ProfilerNvprof
                profiler_obj = profiler(
                    execute_parameters=exec_parameters, app_dir=common.REPOSITORY_HOME, app=common.APP_NAME,
                    metrics=common.METRICS_NSIGHT_CLI, events=common.EVENTS_NSIGHT_CLI,
                    cuda_version=common.CUDA_VERSION, log_base_path=common.PROFILE_DATA_PATH, model=new_model_name,
                    board=gpu_name, logger=logger
                )
                profiler_obj.profile()
                clean_last_profile(app_binary=app_binary, logger=logger)
    except KeyboardInterrupt:
        clean_last_profile(app_binary=common.APP_NAME, logger=logger)

    toc = time.time()
    logger.debug(f"Time necessary to process the profile {datetime.timedelta(seconds=(toc - tic))}")


if __name__ == '__main__':
    profile_all()
