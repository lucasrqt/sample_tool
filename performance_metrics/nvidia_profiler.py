#!/usr/bin/python3

import datetime
import filecmp
import logging
import os
import re
import pathlib
import time

import profiler_class
import profiler_flags

# GPU name : SMS
KEPLER_GPUS = {"NVIDIA Tesla K40c": 35}
JETSON_NANO = {}
VOLTA = {"NVIDIA TITAN V": 70}
AMPERE = {"NVIDIA GeForce RTX 3060 Ti": 86}
SUPPORT_HALF_GPUS = {**JETSON_NANO, **VOLTA, **AMPERE}
ALL_GPUS = {**KEPLER_GPUS, **SUPPORT_HALF_GPUS}

BENCHMARKS = {

}



DEFAULT_LOG = str(os.path.basename(__file__)).upper().replace(".PY", "")
# For time profiling this will be replaced for the suitable iterations
ITERATIONS_PER_APP = 1


def get_correct_profiler_object(gpu_name):
    if any([gpu_name in gpu for gpu in SUPPORT_HALF_GPUS]):
        return ProfilerNsight
    elif any([gpu_name in gpu for gpu in KEPLER_GPUS]):
        return ProfilerNvprof


def profile_nsight():
    # create logger
    logging.setLoggerClass(ColoredLogger)
    logger = logging.getLogger(DEFAULT_LOG)
    # Update the radiation-benchmarks and nvbitfi
    pull_git_directory(path="/home/carol/radiation-benchmarks", logger=logger)
    pull_git_directory(path="/home/carol/NVBITFI/nvbit_release/tools/nvbitfi", logger=logger)
    # Check which GPU i'm executing
    gpu_name = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read()
    gpu_name = gpu_name.strip()
    script_path = os.path.dirname(os.path.abspath(__file__))
    # The files will be saved on this repository
    log_folder = f"{script_path}/data/multi_apps_profile"
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logger.debug(f"GPU: {gpu_name}")
    # Select the class based on the GPU name
    metrics = STALL_METRICS if GAMBIARRA_STALL else METRICS_NSIGHT_CLI_SM70
    # Update the benchmarks iterations
    # Get home dir
    home = str(pathlib.Path.home())
    radiation_benchmarks_dir = f"{home}/radiation-benchmarks"
    tic = time.time()
    # Loop through each benchmark
    for app, app_parameters in BENCHMARKS.items():
        # Check if it is empty and the GPU is supported
        if app_parameters and gpu_name in app_parameters["SUPPORTED_GPUS"]:
            make_parameters = [f"{k}={v}" for k, v in app_parameters["MAKE_PARAMETERS"].items()]
            make_parameters += [f"ITERATIONS={ITERATIONS_PER_APP}"]
            make_parameters = " ".join(make_parameters)
            target_sm = app_parameters["SUPPORTED_GPUS"][gpu_name]
            # Extract the binary name and the default SASS code
            app_binary = app_parameters["APP_BIN"]
            app_dir = f"{radiation_benchmarks_dir}/src/cuda/{app_parameters['APP_DIR']}"
            default_build_sass_file = f"/tmp/default_build_{app_binary}"
            flag_parsed_def = re.sub("-*=*[ ]*\"*", "", DEFAULT_OPTC)
            default_cuda_path = f"/usr/local/cuda-{BIGGEST_COMPILER['NVCC']}"
            make_for_a_given_flag(benchmark_path=app_dir, cuda_path=default_cuda_path, gxx=BIGGEST_COMPILER["CXX"],
                                  gcc=BIGGEST_COMPILER["CC"], flag_to_add=DEFAULT_OPTC,
                                  target_sm=target_sm, logger=logger)
            generate_sass_code(compiler_path=default_cuda_path, binary_path=f"{app_dir}/{app_binary}",
                               output_sass_path=default_build_sass_file, logger=logger)
            logger.debug("Profiling for the default config")
            make_generate(benchmark_path=app_dir, make_cmd=make_parameters, cuda_path=default_cuda_path, logger=logger)
            object_caller = get_correct_profiler_object(gpu_name=gpu_name)
            profiler_obj = object_caller(make_parameters=make_parameters, app_dir=app_dir, app=app,
                                         metrics=metrics, cuda_version=BIGGEST_COMPILER["NVCC"],
                                         log_base_path=log_folder, flag=flag_parsed_def, board=gpu_name,
                                         logger=logger)
            profiler_obj.profile()
            clean_last_profile(app_binary=app_binary, logger=logger)

            # Loop through each compilers
            for compiler_config in COMPILERS:
                for flag in ALL_FLAGS:
                    flag_processed = flag
                    if "FLAG_PARSER" in app_parameters:
                        flag_processed = app_parameters["FLAG_PARSER"](flag_processed)
                    cuda_version = compiler_config["NVCC"]
                    gxx_version = compiler_config["CXX"]
                    gcc_version = compiler_config["CC"]
                    cuda_path = f"/usr/local/cuda-{cuda_version}"
                    flag_parsed = re.sub("-*=*[ ]*\"*", "", flag_processed)
                    flag_build_sass_file = f"/tmp/flag_build_{app_binary}_{cuda_version}_{flag_parsed}"
                    # Building the binary
                    logger.debug(f"Making the binary for the {flag_processed} NVCC {cuda_version}")
                    make_for_a_given_flag(benchmark_path=app_dir, cuda_path=cuda_path, gxx=gxx_version,
                                          gcc=gcc_version, flag_to_add=flag_processed, target_sm=target_sm,
                                          logger=logger)
                    # Gen SASS
                    generate_sass_code(compiler_path=cuda_path, binary_path=f"{app_dir}/{app_binary}",
                                       output_sass_path=flag_build_sass_file, logger=logger)
                    # Comparing
                    logger.debug(f"Comparing {default_build_sass_file} and {flag_build_sass_file}")
                    is_necessary_to_profile = filecmp.cmp(default_build_sass_file, flag_build_sass_file, shallow=False)
                    # Profile the benchmark if necessary
                    if is_necessary_to_profile is False:
                        logger.debug("It is necessary to profile as it is different, first generate a new gold")
                        make_generate(benchmark_path=app_dir, make_cmd=make_parameters, cuda_path=cuda_path,
                                      logger=logger)
                        profiler_obj = object_caller(make_parameters=make_parameters, app_dir=app_dir, app=app,
                                                     metrics=metrics, cuda_version=cuda_version,
                                                     log_base_path=log_folder, flag=flag_parsed, board=gpu_name,
                                                     logger=logger)
                        profiler_obj.profile()
                        clean_last_profile(app_binary=app_binary, logger=logger)

    toc = time.time()
    logger.debug(f"Time necessary to process the profile {datetime.timedelta(seconds=(toc - tic))}")
    # execute_cmd("sudo shutdown -h now", logger=logger)


def clean_last_profile(app_binary, logger):
    # Clean the last profiling
    for to_clean in [app_binary, "nvprof", "nv-nsight-cu-cli", "nvcc"]:
        execute_cmd(f"if pgrep  {to_clean}; then pkill  {to_clean}; fi", logger=logger)
    execute_cmd("sync", logger=logger)


if __name__ == '__main__':
    profile_nsight()
