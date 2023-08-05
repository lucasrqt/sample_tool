#!/usr/bin/python3
import datetime
import logging
import os
import sys
import time
import typing

import common
import profiler_class

DEFAULT_LOG = str(os.path.basename(__file__)).upper().replace(".PY", "")
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

def get_correct_profiler_object(gpu_name) -> typing.Type[
    typing.Union[profiler_class.ProfilerNsight, profiler_class.ProfilerNvprof]
]:
    if any([gpu_name in gpu for gpu in common.AMPERE]):
        return profiler_class.ProfilerNsight
    elif any([gpu_name in gpu for gpu in common.PASCAL]):
        return profiler_class.ProfilerNvprof


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
    log_folder = f"{SCRIPT_PATH}/../data/performance_metrics"
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logger.debug(f"GPU:{gpu_name}")

    app_dir = f"{SCRIPT_PATH}/.."
    tic = time.time()
    # Loop through each benchmark
    for model_name, app_parameters in common.BENCHMARKS.items():
        # Check if it is empty and the GPU is supported
        if gpu_name in app_parameters["SUPPORTED_GPUS"]:
            exec_parameters = app_parameters["EXEC_PARAMETERS"]
            # Extract the binary name and the default SASS code
            app_binary = app_parameters["APP_BIN"]
            logger.debug("Profiling for the default config")
            object_caller = get_correct_profiler_object(gpu_name=gpu_name)
            profiler_obj = object_caller(execute_parameters=exec_parameters, app_dir=app_dir, app=common.APP_NAME,
                                         metrics=common.METRICS_NSIGHT_CLI, events=common.EVENTS_NSIGHT_CLI,
                                         cuda_version=common.CUDA_VERSION,
                                         log_base_path=log_folder, model=model_name, board=gpu_name, logger=logger)
            profiler_obj.profile()
            clean_last_profile(app_binary=app_binary, logger=logger)

    toc = time.time()
    logger.debug(f"Time necessary to process the profile {datetime.timedelta(seconds=(toc - tic))}")
    # execute_cmd("sudo shutdown -h now", logger=logger)

if __name__ == '__main__':
    profile_all()
