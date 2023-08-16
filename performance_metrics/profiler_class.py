import abc
import os
import re

import common

# "MemoryWorkloadAnalysis",
NSIGHT_PROFILE_SECTIONS = ["LaunchStats", "SpeedOfLight", "Occupancy", "SchedulerStats", "WarpStateStats"]


class Profiler(abc.ABC):
    def __init__(self, execute_parameters, app, app_dir, metrics, events, cuda_version, log_base_path, model, board,
                 logger):
        self._app = app
        self._app_dir = app_dir
        self._metrics = metrics
        self._events = events
        self._cuda_path = f"/usr/local/cuda-{cuda_version}"
        self._cuda_bin = f"{self._cuda_path}/bin"
        self._log_base_path = log_base_path
        self._cuda_version = cuda_version
        self._logger = logger
        self._board = board.replace(" ", "")
        self._model = model
        self._base_execute_cmd = f"eval LD_LIBRARY_PATH={self._cuda_path}"
        self._base_execute_cmd += "/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} "
        self._app_make_execute = f"{self._app_dir}/{self._app} {execute_parameters} "

    def _log_name(self, target, profiler_tool):
        log_name = f"{self._log_base_path}/{profiler_tool}_log_{target}"
        log_name += f"_{self._model}_{self._app}_{self._cuda_version}_{self._board}.csv"
        return log_name

    @abc.abstractmethod
    def _metrics_profiler_cmd(self):
        raise NotImplemented("Abstract method")

    @abc.abstractmethod
    def _time_profiler_cmd(self):
        raise NotImplemented("Abstract method")

    @abc.abstractmethod
    def _memory_profiler_cmd(self):
        raise NotImplemented("Abstract method")

    @abc.abstractmethod
    def _events_profiler_cmd(self):
        raise NotImplemented("Abstract method")

    def profile(self):
        common.execute_cmd(f"mkdir -p {self._log_base_path}", self._logger)
        benchmark_path = self._app_dir
        self._logger.debug("BENCHMARK PATH: " + benchmark_path)
        os.chdir(benchmark_path)
        # General profiler
        self._logger.debug(f"Profiling metrics app {self._app}")
        cmd, metrics_log_path = self._metrics_profiler_cmd()
        common.execute_cmd(cmd=cmd, logger=self._logger)
        self._logger.debug(f"Saved at {metrics_log_path}")
        # # Get the execution time
        # self._logger.debug(f"Profiling execution time (sections) app {self._app}")
        # cmd, time_log_path = self._time_profiler_cmd()
        # common.execute_cmd(cmd=cmd, logger=self._logger)
        # self._logger.debug(f"Saved at {time_log_path}")
        # Get the SHARED and RF
        self._logger.debug(f"Profiling memory app {self._app}")
        cmd, memory_log_path = self._memory_profiler_cmd()
        common.execute_cmd(cmd=cmd, logger=self._logger)
        self._logger.debug(f"Saved at {memory_log_path}")
        # Get the execution cycles
        self._logger.debug(f"Profiling events app {self._app}")
        cmd, events_log_path = self._events_profiler_cmd()
        common.execute_cmd(cmd=cmd, logger=self._logger)
        self._logger.debug(f"Saved at {events_log_path}")


class ProfilerNvprof(Profiler):
    def __init__(self, *args, **kwargs):
        super(ProfilerNvprof, self).__init__(*args, **kwargs)
        self._profiler_tool = "nvprof"
        self._base_execute_cmd += f"{self._cuda_bin}/{self._profiler_tool}  --cpu-thread-tracing off --csv "
        self._base_execute_cmd += "--profile-child-processes "

    def _time_profiler_cmd(self):
        log_path = self._log_name(target="time", profiler_tool=self._profiler_tool)
        profiler_cmd = f"{self._base_execute_cmd} --trace gpu --profile-api-trace none {self._app_make_execute} "
        profiler_cmd += f"> {log_path} 2>&1"
        profiler_cmd = re.sub(r"--iterations (\d+)",
                              f"--iterations {common.PROFILE_TIME_ITERATIONS}", profiler_cmd)
        return profiler_cmd, log_path

    def _memory_profiler_cmd(self):
        log_path = self._log_name(target="memory", profiler_tool=self._profiler_tool)
        profiler_cmd = f"{self._base_execute_cmd} --print-gpu-trace {self._app_make_execute} > {log_path} 2>&1"
        return profiler_cmd, log_path

    def _metrics_profiler_cmd(self):
        metrics = ",".join(self._metrics.keys())
        log_path = self._log_name(target="metrics", profiler_tool=self._profiler_tool)
        profiler_cmd = f"{self._base_execute_cmd} --metrics {metrics} {self._app_make_execute} > {log_path} 2>&1"
        return profiler_cmd, log_path

    def _events_profiler_cmd(self):
        events = ",".join(self._events.keys())
        log_path = self._log_name(target="events", profiler_tool=self._profiler_tool)
        profiler_cmd = f"{self._base_execute_cmd} --events {events} {self._app_make_execute} > {log_path} 2>&1"
        return profiler_cmd, log_path

    def get_log_name(self, target):
        return self._log_name(target=target, profiler_tool=self._profiler_tool)


class ProfilerNsight(Profiler):
    def __init__(self, *args, **kwargs):
        super(ProfilerNsight, self).__init__(*args, **kwargs)
        self._profiler_tool = "nv-nsight-cu-cli"
        self._base_execute_cmd += f"{self._cuda_bin}/{self._profiler_tool} --csv --target-processes all "
        self._base_execute_cmd += "--print-fp" if float(self._cuda_version) > 11.0 else "--fp"

    def _metrics_profiler_cmd(self):
        metrics = ",".join([",".join(value) for key, value in self._metrics.items() if value is not None])
        log_path = self._log_name(target="metrics", profiler_tool=self._profiler_tool)
        profiler_cmd = f"{self._base_execute_cmd} --metrics {metrics} {self._app_make_execute} > {log_path}"
        return profiler_cmd, log_path

    def _events_profiler_cmd(self):
        events = ",".join([",".join(value) for key, value in self._events.items() if value is not None])
        log_path = self._log_name(target="events", profiler_tool=self._profiler_tool)
        profiler_cmd = f"{self._base_execute_cmd} --metrics {events} {self._app_make_execute} > {log_path}"
        return profiler_cmd, log_path

    def _time_profiler_cmd(self):
        log_path = self._log_name(target="time", profiler_tool=self._profiler_tool)
        time_metrics = ",".join([f"gpu__time_active.{i},gpu__time_duration.{i}" for i in ["min", "max", "avg", "sum"]])
        profiler_cmd = f"{self._base_execute_cmd} --metrics {time_metrics}"
        profiler_cmd += f" {self._app_make_execute} > {log_path}"
        profiler_cmd = re.sub(r"--iterations (\d+)",
                              f"--iterations {common.PROFILE_TIME_ITERATIONS}", profiler_cmd)
        return profiler_cmd, log_path

    def _memory_profiler_cmd(self):
        log_path = self._log_name(target="memory", profiler_tool=self._profiler_tool)
        # blank section is to ease the join
        section_list = " --section ".join([""] + NSIGHT_PROFILE_SECTIONS)
        profiler_cmd = f"{self._base_execute_cmd} {section_list} {self._app_make_execute} > {log_path}"
        return profiler_cmd, log_path

    def get_log_name(self, target):
        return self._log_name(target=target, profiler_tool=self._profiler_tool)
