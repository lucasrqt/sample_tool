#!/usr/bin/python3
import asyncio
import datetime
import logging
import os
import sys
import time
import typing

import common
import profiler_class

DEFAULT_LOG = str(os.path.basename(__file__)).upper().replace(".PY", "")


async def task_coroutine(log_path, log_interval):
    nvidia_smi_cmd = ("nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.mem,clocks.gr "
                      f"--format=csv -l {log_interval} > {log_path} 2>&1")
    os.system(f"echo '' > {log_path}")
    try:
        print('executing the nvidia-smi logging')
        os.system(nvidia_smi_cmd)
    except asyncio.CancelledError as e:
        assert os.system(f"pkill -9 -f nvidia-smi") == 0, "Command to kill nvidia-smi failed"
        print(f'received request to cancel with: {e}')


def get_correct_profiler_object(gpu_name) -> typing.Type[
    typing.Union[profiler_class.ProfilerNsight, profiler_class.ProfilerNvprof]
]:
    if any([gpu_name in gpu for gpu in common.AMPERE]):
        return profiler_class.ProfilerNsight
    elif any([gpu_name in gpu for gpu in common.PASCAL]):
        return profiler_class.ProfilerNvprof


async def profile_all():
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
    script_path = os.path.dirname(os.path.abspath(__file__))
    # The files will be saved on this repository
    log_folder = f"{script_path}/../data/performance_metrics"
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logger.debug(f"GPU:{gpu_name}")
    # Set the asynchronous task to log frequency
    # create and schedule the task
    task = asyncio.create_task(task_coroutine(log_path=f"{log_folder}/nvidia_smi.log", log_interval=5))

    app_dir = f"{script_path}/.."
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

    was_cancelled = task.cancel('Stop Right Now')
    # report whether the cancel request was successful
    logger.debug(f'was canceled: {was_cancelled}')
    # wait a moment
    await asyncio.sleep(0.1)
    # check the status of the task
    logger.debug(f'canceled: {task.cancelled()}')
    toc = time.time()
    logger.debug(f"Time necessary to process the profile {datetime.timedelta(seconds=(toc - tic))}")
    # execute_cmd("sudo shutdown -h now", logger=logger)


def clean_last_profile(app_binary, logger):
    # Clean the last profiling
    for to_clean in [app_binary, "nvprof", "nv-nsight-cu-cli", "nvcc"]:
        common.execute_cmd(f"if pgrep  {to_clean}; then pkill  {to_clean}; fi", logger=logger)
    common.execute_cmd("sync", logger=logger)


if __name__ == '__main__':
    asyncio.run(profile_all())
