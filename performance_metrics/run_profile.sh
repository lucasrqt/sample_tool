#!/bin/bash
set -e
set -x

log_interval=5
NVIDIA_SMI_LOG_DIR=../data/performance_metrics

mkdir -p ../log
mkdir -p ${NVIDIA_SMI_LOG_DIR}
log_path=${NVIDIA_SMI_LOG_DIR}/nvidia_smi_data.csv

QUERIES=power.draw,clocks.sm,clocks.mem,clocks.gr,temperature.gpu
# Release the running of nvidia-smi
nvidia-smi --query-gpu=${QUERIES} --format=csv -l ${log_interval} >${log_path} 2>&1 &

# run the profiler
./nvidia_profiler.py

pkill -9 -f nvidia-smi

wait

exit 0
