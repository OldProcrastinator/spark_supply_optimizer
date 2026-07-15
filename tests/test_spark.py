import os
import sys
from pathlib import Path

import pytest

from spark_supply_optimizer.config import SparkRuntimeConfig
from spark_supply_optimizer.spark import (
    auto_driver_memory,
    configure_python_executable,
    configure_s3a,
    effective_memory_limit_bytes,
    java_executable_from_home,
    local_threads_from_master,
    validate_java_runtime,
)


def test_configure_python_executable_sets_pyspark_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYSPARK_PYTHON", raising=False)
    monkeypatch.delenv("PYSPARK_DRIVER_PYTHON", raising=False)

    configure_python_executable()

    assert os.environ["PYSPARK_PYTHON"] == sys.executable
    assert os.environ["PYSPARK_DRIVER_PYTHON"] == sys.executable


def test_validate_java_runtime_fails_without_path_or_java_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr("spark_supply_optimizer.spark.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="docker compose run --rm test"):
        validate_java_runtime()


def test_validate_java_runtime_accepts_java_home(monkeypatch: pytest.MonkeyPatch) -> None:
    java_home = Path("C:/Java/jdk-17")
    monkeypatch.setenv("JAVA_HOME", str(java_home))
    monkeypatch.setattr("spark_supply_optimizer.spark.shutil.which", lambda _: None)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: path == java_executable_from_home(str(java_home)),
    )

    validate_java_runtime()


def test_validate_java_runtime_rejects_invalid_java_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAVA_HOME", "C:\\not-a-jdk")
    monkeypatch.setattr("spark_supply_optimizer.spark.shutil.which", lambda _: None)
    monkeypatch.setattr(Path, "exists", lambda _: False)

    with pytest.raises(RuntimeError, match="JAVA_HOME"):
        validate_java_runtime()


def test_local_threads_from_master_distinguishes_local_mode() -> None:
    assert local_threads_from_master("local") == "1"
    assert local_threads_from_master("local[2]") == "2"
    assert local_threads_from_master("local[*]") == "*"
    assert local_threads_from_master("spark://spark-master:7077") is None


class FakeSparkBuilder:
    def __init__(self) -> None:
        self.options: dict[str, str] = {}

    def config(self, key: str, value: str):
        self.options[key] = value
        return self


def test_configure_s3a_adds_minio_spark_options() -> None:
    builder = FakeSparkBuilder()
    config = SparkRuntimeConfig(
        s3_endpoint="http://minio:9000",
        s3_access_key="minioadmin",
        s3_secret_key="minioadmin123",
    )

    configured = configure_s3a(builder, config)

    assert configured.options["spark.hadoop.fs.s3a.endpoint"] == "http://minio:9000"
    assert configured.options["spark.hadoop.fs.s3a.path.style.access"] == "true"
    assert configured.options["spark.hadoop.fs.s3a.access.key"] == "minioadmin"
    assert configured.options["spark.hadoop.fs.s3a.secret.key"] == "minioadmin123"
    assert "hadoop-aws:3.4.2" in configured.options["spark.jars.packages"]


def test_build_spark_applies_local_dir_and_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_builder = FakeSparkBuilder()

    class FakeSparkSession:
        builder = fake_builder

    monkeypatch.setattr("spark_supply_optimizer.spark.SparkSession", FakeSparkSession)
    monkeypatch.setattr("spark_supply_optimizer.spark.configure_python_executable", lambda: None)
    monkeypatch.setattr("spark_supply_optimizer.spark.validate_java_runtime", lambda: None)

    fake_builder.appName = lambda _: fake_builder
    fake_builder.master = lambda value: fake_builder.config("spark.master", value)
    fake_builder.getOrCreate = lambda: object()

    result = __import__("spark_supply_optimizer.spark", fromlist=["build_spark"]).build_spark(
        SparkRuntimeConfig(
            master="local[2]",
            driver_memory="6g",
            local_dir=str(tmp_path / "spark-tmp"),
        )
    )

    assert result is not None
    assert fake_builder.options["spark.master"] == "local[2]"
    assert fake_builder.options["spark.driver.memory"] == "6g"
    assert fake_builder.options["spark.local.dir"] == str(tmp_path / "spark-tmp")
    assert fake_builder.options["spark.ui.enabled"] == "true"
    assert fake_builder.options["spark.ui.port"] == "4040"
    assert fake_builder.options["spark.ui.bindAddress"] == "0.0.0.0"
    assert fake_builder.options["spark.executor.heartbeatInterval"] == "60s"
    assert fake_builder.options["spark.network.timeout"] == "600s"
    assert fake_builder.options["spark.hadoop.parquet.block.size"] == "16777216"


def test_auto_driver_memory_uses_container_memory_without_capping_cpu() -> None:
    assert auto_driver_memory(8 * 1024**3) == "6g"
    assert auto_driver_memory(None) is None


def test_effective_memory_limit_falls_back_to_visible_system_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("spark_supply_optimizer.spark.cgroup_memory_limit_bytes", lambda: None)
    monkeypatch.setattr("spark_supply_optimizer.spark.system_memory_bytes", lambda: 16 * 1024**3)

    assert effective_memory_limit_bytes() == 16 * 1024**3
