# Copyright 2022-2025 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from ramble.modkit import *

class JaxTpuInfo(BasicModifier):
    """Define a modifier for gathering TPU info using JAX.

    This modifier uses the JAX library to connect to and query available
    TPU devices, reporting their specifications and memory statistics.
    """

    name = "jax-tpu-info"

    tags("tpu-utility", "diagnostics", "jax-tool", "tpu")

    maintainers("carlosboneti")

    mode("standard", description="Standard execution mode for jax-tpu-info")
    default_mode("standard")

    variable_modification(
        "jax_tpu_log",
        "{experiment_run_dir}/jax_tpu_output.log",
        method="set",
        modes=["standard"],
    )

    archive_pattern("jax_tpu_output.log", modes=["standard"])

    # --- Figures of Merit ---
    figure_of_merit(
        "TPU Device Count",
        fom_regex=r"^tpu\.device_count: (?P<device_count>\d+)",
        group_name="device_count",
        units="",
        fom_type=FomType.INFO,
        log_file="{jax_tpu_log}",
    )

    figure_of_merit_context(
        "tpu",
        regex=r"^tpu\.(?P<tpu_index>\d+)\.",
        output_format="TPU {tpu_index}",
        log_file="{jax_tpu_log}",
    )

    figure_of_merit(
        "Platform",
        fom_regex=r"^tpu\.\d+\.platform: (?P<platform>.+)",
        group_name="platform",
        units="",
        fom_type=FomType.INFO,
        contexts=["tpu"],
        log_file="{jax_tpu_log}",
    )

    figure_of_merit(
        "Device Kind",
        fom_regex=r"^tpu\.\d+\.device_kind: (?P<device_kind>.+)",
        group_name="device_kind",
        units="",
        fom_type=FomType.INFO,
        contexts=["tpu"],
        log_file="{jax_tpu_log}",
    )

    figure_of_merit(
        "Host ID",
        fom_regex=r"^tpu\.\d+\.host_id: (?P<host_id>.+)",
        group_name="host_id",
        units="",
        fom_type=FomType.INFO,
        contexts=["tpu"],
        log_file="{jax_tpu_log}",
    )

    figure_of_merit(
        "Memory Bytes Used",
        fom_regex=r"^tpu\.\d+\.memory_bytes_used: (?P<mem_used>\d+)",
        group_name="mem_used",
        units="B",
        fom_type=FomType.MEASURE,
        contexts=["tpu"],
        log_file="{jax_tpu_log}",
    )

    figure_of_merit(
        "Memory Bytes Limit",
        fom_regex=r"^tpu\.\d+\.memory_bytes_limit: (?P<mem_limit>\d+)",
        group_name="mem_limit",
        units="B",
        fom_type=FomType.INFO,
        contexts=["tpu"],
        log_file="{jax_tpu_log}",
    )

    register_builtin("jax_tpu_exec")

    def jax_tpu_exec(self):
        # This Python script will be executed by the modifier
        script = '''
import jax
import sys

try:
    devices = jax.devices()
    print(f"tpu.device_count: {len(devices)}")

    if not devices:
        print("No JAX devices found.")
        sys.exit(0)

    for i, device in enumerate(devices):
        print(f"tpu.{i}.platform: {device.platform}")
        print(f"tpu.{i}.device_kind: {device.device_kind}")
        print(f"tpu.{i}.id: {device.id}")
        print(f"tpu.{i}.host_id: {device.host_id}")
        print(f"tpu.{i}.slice_index: {device.slice_index}")
        try:
            stats = device.memory_stats()
            print(f"tpu.{i}.memory_bytes_used: {stats["bytes_used"]}")
            print(f"tpu.{i}.memory_bytes_limit: {stats["bytes_limit"]}")
        except Exception as e:
            print(f"tpu.{i}.memory_stats: Not available ({e})")

except Exception as e:
    print(f"Failed to get JAX device info: {e}")
    sys.exit(1)
'''
        # Using sys.executable to ensure we use the python from the correct venv
        return [f"python -c '{script}' >> {{jax_tpu_log}}"]
