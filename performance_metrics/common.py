import os
import sys
from inspect import getframeinfo, stack

sys.path.insert(0, '../')
import configs

APP_NAME = "perf_measure.py"

PROFILE_DATA_PATH = "data/performance_metrics"
FINAL_PROFILE_DATABASE = f"{PROFILE_DATA_PATH}/final_profile_processed.csv"


def execute_cmd(cmd, logger):
    logger.debug(f"Executing {cmd}")
    ret = os.system(cmd)
    caller = getframeinfo(stack()[1][0])
    if ret != 0:
        logger.error(f"ERROR AT: {caller.filename}:{caller.lineno} CMD: {cmd}")
        logger.error(f"Command was not correctly executed error code {ret}")
        raise ValueError()


# GPU name : SMS
PASCAL = {"Quadro P2000": 60}
AMPERE = {"NVIDIA GeForce RTX 3060 Ti": 86}
ALL_GPUS = {**PASCAL, **AMPERE}
CUDA_VERSION = "11.7"

PROFILE_METRICS_ITERATIONS = 5
PROFILE_TIME_ITERATIONS = 50

BENCHMARKS = {
    k: dict(
        SUPPORTED_GPUS=ALL_GPUS,
        EXEC_PARAMETERS=f"--{hardening} --model {k} --iterations {PROFILE_METRICS_ITERATIONS}",
        APP_BIN="perf_measure.py"
    )
    for k in configs.MODELS for hardening in ["replace-id", "no-replace-id"]
}

################################################################
# Metrics
################################################################
QUANTITATIVE_METRICS_NSIGHT_CLI = {
    # # Double arithmetic
    # "flop_count_dp_fma": ["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"],
    # "flop_count_dp_mul": ["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"],
    # "flop_count_dp_add": ["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"],
    #
    # # Single arithmetic
    # "flop_count_sp_add": ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"],
    # "flop_count_sp_mul": ["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"],
    # "flop_count_sp_fma": ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"],
    # "flop_count_sp_special": None,
    #
    # # # Half arithmetic
    # "flop_count_hp_add": ["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"],
    # "flop_count_hp_mul": ["smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"],
    # "flop_count_hp_fma": ["smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"],
    #
    # # LD/ST, IF, and INT
    # "inst_compute_ld_st": ["smsp__sass_thread_inst_executed_op_memory_pred_on.sum"],
    # "inst_control": ["smsp__sass_thread_inst_executed_op_control_pred_on.sum"],
    # "inst_integer": ["smsp__sass_thread_inst_executed_op_integer_pred_on.sum"],
    #
    # # Communication, MISC, and conversion
    # "inst_inter_thread_communication": ["smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum"],
    # "inst_misc": ["smsp__sass_thread_inst_executed_op_misc_pred_on.sum"],
    # "inst_bit_convert": ["smsp__sass_thread_inst_executed_op_conversion_pred_on.sum"],
    #
    # # Atomic functions
    # "atomic_transactions": ["l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum",
    #                         "l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum",
    #                         "l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum",
    #                         "l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum"],
    #
    # # Tensor instructions
    # # It can be one of these two, following the other metrics pattern *executed_pipe_tensor_op*
    # # smsp__inst_executed_pipe_tensor.sum smsp__inst_executed_pipe_tensor_op_hmma.sum
    # "tensor_count": ["smsp__inst_executed_pipe_tensor_op_hmma.sum"]
}

PERFORMANCE_METRICS_NSIGHT_CLI = {
    # Performance metrics
    # "alu_fu_utilization": ["smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    #                        "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    #                        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    #                        "smsp__inst_executed_pipe_fp16.sum",
    #                        "smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active"],
    "achieved_occupancy": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "ipc": ["sm__inst_executed.avg.per_cycle_active"],
    "ldst_fu_utilization": ["smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"],
    # "cf_fu_utilization": None,
    # "tex_fu_utilization": ["smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active"],
    # "l1_shared_utilization": ["l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed"],
    # "l2_utilization": ["lts__t_sectors.avg.pct_of_peak_sustained_elapsed"],
    # "tex_utilization": ["l1tex__texin_sm2tex_req_cycles_active.avg.pct_of_peak_sustained_elapsed",
    #                     "l1tex__f_tex2sm_cycles_active.avg.pct_of_peak_sustained_elapsed"],
    # "dram_utilization": ["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    # "sysmem_utilization": None,
    # "sysmem_read_utilization": None,
    # "sysmem_write_utilization": None,
    # "issued_ipc": ["sm__inst_issued.avg.per_cycle_active"],
    # Efficiency metrics
    "sm_efficiency": ["smsp__cycles_active.avg.pct_of_peak_sustained_elapsed"],
    # "sm_efficiency_instance": None,
    # "shared_efficiency": None,
    # "gld_efficiency": None, Does not work with FASTER-RCNN
    # "gst_efficiency": None,
    # "warp_execution_efficiency": ["smsp__thread_inst_executed_per_inst_executed.ratio"],
    # "warp_nonpred_execution_efficiency": ["smsp__thread_inst_executed_per_inst_executed.pct"],
    # "eligible_warps_per_cycle": ["smsp__warps_eligible.sum.per_cycle_active"],
    # Penalty metrics
    # -- Replay
    # "inst_replay_overhead": None,
    # "shared_replay_overhead": None,
    # "global_replay_overhead": None,
    # "global_cache_replay_overhead": None,
    # "local_replay_overhead": None,
    # "atomic_replay_overhead": None,
    # -- HIT
    # "l1_cache_global_hit_rate": None, "l1_cache_local_hit_rate": None,
    # "tex_cache_hit_rate": ["l1tex__t_sector_hit_rate.pct"],
    # keep in mind that it is not the directly flag related for these
    # "l2_l1_read_hit_rate": None,
    # l2 texture does not work
    # "l2_texture_read_hit_rate": ["l2_tex_read_hit_rate"],
    # "nc_cache_global_hit_rate": None,
    # # Quantitative metrics (less significant)
    # "inst_issued": ["smsp__inst_issued.sum"],
    "inst_executed": ["smsp__inst_executed.sum"],
    # "cf_issued": None,
    "cf_executed": ["smsp__inst_executed_pipe_cbu.sum", "smsp__inst_executed_pipe_adu.sum"],
    # "ldst_issued": None, "ldst_executed": None,
}

# -- Stall
STALL_METRICS = {
    # stall_inst_fetch: Percentage of stalls occurring because the next assembly instruction has not yet been fetched
    "stall_inst_fetch": ["smsp__warp_issue_stalled_no_instruction_per_warp_active.pct"],
    # stall_exec_dependency: Percentage of stalls occurring because an input required
    # by the instruction is not yet available
    "stall_exec_dependency": ["smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
                              "smsp__warp_issue_stalled_wait_per_warp_active.pct"],
    # stall_memory_dependency: Percentage of stalls occurring because a memory operation
    # cannot be performed due to the required resources not being available or fully utilized,
    # or because too many requests of a given type are outstanding
    "stall_memory_dependency": ["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"],
    # stall_texture: Percentage of stalls occurring because the texture sub-system is
    # fully utilized or has too many outstanding requests
    "stall_texture": ["smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct"],
    # stall_sync: Percentage of stalls occurring because the warp is blocked at a __syncthreads() call
    "stall_sync": ["smsp__warp_issue_stalled_barrier_per_warp_active.pct",
                   "smsp__warp_issue_stalled_membar_per_warp_active.pct"],
    # stall_other: Percentage of stalls occurring due to miscellaneous reasons
    "stall_other": ["smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",
                    "smsp__warp_issue_stalled_misc_per_warp_active.pct"],
    # "stall_pipe_busy": ["smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    #                     "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"],
    # "stall_constant_memory_dependency": ["smsp__warp_issue_stalled_imc_miss_per_warp_active.pct"],
    # "stall_memory_throttle": ["smsp__warp_issue_stalled_drain_per_warp_active.pct",
    #                           "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"],
    # "stall_not_selected": ["smsp__warp_issue_stalled_not_selected_per_warp_active.pct"],
}

EVENTS_NSIGHT_CLI = {
    # Number of cycles a multiprocessor has at least one active warp.
    "active_cycles_pm": ["sm__cycles_active.sum"],
    # Elapsed clocks
    "elapsed_cycles_sm": ["sm__cycles_elapsed.sum"],
    # Elapsed clocks
    "elapsed_cycles_pm": ["sm__cycles_elapsed.sum"]
}

METRICS_NSIGHT_CLI = {**QUANTITATIVE_METRICS_NSIGHT_CLI, **PERFORMANCE_METRICS_NSIGHT_CLI}
