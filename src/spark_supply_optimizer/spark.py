"""SparkSession construction helpers."""

from __future__ import annotations

import os
import re
import shutil
import socket
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from spark_supply_optimizer.config import SparkRuntimeConfig

LOCAL_MASTER_PATTERN = re.compile(r"local(?:\[(?P<threads>[^\]]+)])?")


def bytes_to_gib(value: int | None) -> float | None:
    """Convert bytes to GiB for compact resource reports."""

    if value is None:
        return None
    return round(value / (1024**3), 3)


def project_hadoop_home() -> Path:
    """Return the project-local Hadoop helper directory."""

    return Path(__file__).resolve().parents[2] / ".hadoop"


def java_executable_from_home(java_home: str) -> Path:
    """Return the Java executable expected under JAVA_HOME."""

    executable = "java.exe" if os.name == "nt" else "java"
    return Path(java_home) / "bin" / executable


def configure_python_executable() -> None:
    """Point PySpark workers to the same Python executable as the driver."""

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def validate_java_runtime() -> None:
    """Fail fast with an actionable message when Java is not configured."""

    java_home = os.environ.get("JAVA_HOME")
    java_from_path = shutil.which("java")
    if java_home:
        java_executable = java_executable_from_home(java_home)
        if java_executable.exists():
            os.environ["PATH"] = f"{java_executable.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            return

        raise RuntimeError(
            "`JAVA_HOME` is configured, but Java executable was not found at "
            f"`{java_executable}`. Set JAVA_HOME to the JDK directory, not to "
            "the `bin` directory."
        )

    if java_from_path:
        return

    raise RuntimeError(
        "Java runtime is required by PySpark, but neither `java` in PATH nor "
        "`JAVA_HOME` is configured. Use `docker compose run --rm test` for the "
        "supported setup without installing Java on the host, or install OpenJDK "
        "17 and expose it to the current process."
    )


def local_threads_from_master(master: str | None) -> str | None:
    """Extract local Spark worker thread count from a local[N] master URL."""

    if master is None:
        return None
    match = LOCAL_MASTER_PATTERN.fullmatch(master)
    if match is None:
        return None
    return match.group("threads") or "1"


def executor_memory_status(spark: SparkSession) -> list[dict[str, object]]:
    """Return executor memory status from Spark's JVM API when available."""

    try:
        status_map = spark.sparkContext._jsc.sc().getExecutorMemoryStatus()  # noqa: SLF001
        iterator = status_map.iterator()
    except Exception:
        return []

    executors = []
    while iterator.hasNext():
        entry = iterator.next()
        location = str(entry._1())
        host = location.rsplit(":", maxsplit=1)[0]
        memory = entry._2()
        max_memory, remaining_memory = int(memory._1()), int(memory._2())
        executors.append(
            {
                "location": location,
                "host": host,
                "max_memory_bytes": max_memory,
                "max_memory_gib": bytes_to_gib(max_memory),
                "remaining_memory_bytes": remaining_memory,
                "remaining_memory_gib": bytes_to_gib(remaining_memory),
            }
        )
    return executors


def cgroup_memory_limit_bytes() -> int | None:
    """Return container memory limit when running under Linux cgroups."""

    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        try:
            raw_value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw_value and raw_value != "max":
            return int(raw_value)
    return None


def system_memory_bytes() -> int | None:
    """Return physical memory visible from Linux /proc when cgroups do not expose a limit."""

    meminfo = Path("/proc/meminfo")
    try:
        lines = meminfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        if not line.startswith("MemTotal:"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            return int(parts[1]) * 1024
    return None


def effective_memory_limit_bytes() -> int | None:
    """Return the best memory budget signal available to the Spark driver."""

    return cgroup_memory_limit_bytes() or system_memory_bytes()


def auto_driver_memory(memory_limit_bytes: int | None) -> str | None:
    """Choose driver memory from visible container memory without capping CPU."""

    if memory_limit_bytes is None:
        return None
    memory_gib = memory_limit_bytes / (1024**3)
    if memory_gib < 2:
        return "1g"
    usable_gib = max(1, int(memory_gib * 0.75))
    return f"{usable_gib}g"


def driver_process_resources() -> dict[str, object]:
    """Return resources visible to the Python driver process."""

    cgroup_memory_limit = cgroup_memory_limit_bytes()
    physical_memory = system_memory_bytes()
    effective_memory_limit = cgroup_memory_limit or physical_memory
    return {
        "hostname": socket.gethostname(),
        "cpu_count_visible": os.cpu_count(),
        "cgroup_memory_limit_bytes": cgroup_memory_limit,
        "cgroup_memory_limit_gib": bytes_to_gib(cgroup_memory_limit),
        "physical_memory_bytes": physical_memory,
        "physical_memory_gib": bytes_to_gib(physical_memory),
        "effective_memory_limit_bytes": effective_memory_limit,
        "effective_memory_limit_gib": bytes_to_gib(effective_memory_limit),
    }


def spark_runtime_report(spark: SparkSession) -> dict[str, object]:
    """Collect Spark runtime details relevant for run reports."""

    context = spark.sparkContext
    master = context.master
    local_threads = local_threads_from_master(master)
    executors = executor_memory_status(spark)
    executor_hosts = sorted({str(executor["host"]) for executor in executors})
    deploy_mode = "local" if local_threads is not None else "cluster"
    return {
        "spark": {
            "app_id": context.applicationId,
            "app_name": context.appName,
            "master": master,
            "deploy_mode": deploy_mode,
            "default_parallelism": context.defaultParallelism,
            "local_worker_threads": local_threads,
            "note": (
                "local_worker_threads are threads inside one driver process, not computers"
                if deploy_mode == "local"
                else "executor_hosts are Spark executor hosts/containers observed by the driver"
            ),
            "conf": {
                key: value
                for key, value in context.getConf().getAll()
                if key
                in {
                    "spark.driver.memory",
                    "spark.executor.memory",
                    "spark.executor.cores",
                    "spark.local.dir",
                    "spark.sql.shuffle.partitions",
                    "spark.master",
                }
            },
        },
        "cluster_observed": {
            "executor_count": len(executors),
            "executor_host_count": len(executor_hosts),
            "executor_hosts": executor_hosts,
            "executors": executors,
            "physical_computer_count_note": (
                "Spark exposes executor hosts/containers. Physical computer count is reliable "
                "only when workers run on distinct hostnames/IPs."
            ),
        },
        "driver_process": driver_process_resources(),
    }


def build_spark(config: SparkRuntimeConfig) -> SparkSession:
    """Create a SparkSession for local or standalone-cluster execution."""

    configure_python_executable()
    validate_java_runtime()

    effective_driver_memory = config.driver_memory
    if config.auto_tune and effective_driver_memory is None:
        effective_driver_memory = auto_driver_memory(effective_memory_limit_bytes())

    builder = SparkSession.builder.appName(config.app_name)
    if config.master:
        builder = builder.master(config.master)

    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "false")
    builder = builder.config("spark.ui.enabled", "true")
    builder = builder.config("spark.ui.port", "4040")
    builder = builder.config("spark.ui.bindAddress", "0.0.0.0")
    builder = builder.config("spark.executor.heartbeatInterval", "60s")
    builder = builder.config("spark.network.timeout", "600s")
    builder = builder.config("spark.hadoop.parquet.block.size", "16777216")
    builder = builder.config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")

    if config.s3_endpoint:
        builder = configure_s3a(builder, config)

    if config.shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))

    if config.local_dir:
        Path(config.local_dir).mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.local.dir", config.local_dir)

    if config.executor_memory:
        builder = builder.config("spark.executor.memory", config.executor_memory)
    if effective_driver_memory:
        builder = builder.config("spark.driver.memory", effective_driver_memory)

    return builder.getOrCreate()


def configure_s3a(builder, config: SparkRuntimeConfig):
    """Configure Spark to read and write S3-compatible object storage."""

    packages = os.environ.get("SPARK_S3A_PACKAGES", "org.apache.hadoop:hadoop-aws:3.4.2")
    ivy_cache = os.environ.get("SPARK_JARS_IVY", "/opt/spark-ivy")

    builder = builder.config("spark.jars.packages", packages)
    if Path(ivy_cache).exists():
        builder = builder.config("spark.jars.ivy", ivy_cache)

    builder = builder.config("spark.hadoop.fs.s3a.endpoint", config.s3_endpoint)
    builder = builder.config("spark.hadoop.fs.s3a.path.style.access", "true")
    builder = builder.config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    builder = builder.config(
        "spark.hadoop.fs.s3a.impl",
        "org.apache.hadoop.fs.s3a.S3AFileSystem",
    )
    builder = builder.config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    )

    if config.s3_access_key:
        builder = builder.config("spark.hadoop.fs.s3a.access.key", config.s3_access_key)
    if config.s3_secret_key:
        builder = builder.config("spark.hadoop.fs.s3a.secret.key", config.s3_secret_key)

    return builder
