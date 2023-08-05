#!/bin/bash
set -e
set -x

log_interval=5
log_path=../data/performance_metrics/nvidia_smi_data.log

# Release the running of nvidia-smi
nvidia-smi --query-gpu=power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l ${log_interval} > ${log_path} 2>&1 &

# run the profiler
./nvidia_profiler.py

pkill -9 -f nvidia-smi

wait

exit 0
