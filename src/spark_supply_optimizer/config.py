"""Runtime configuration objects for the M5 Spark pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SparkRuntimeConfig:
    """Spark runtime options shared by all commands."""

    app_name: str = "m5-spark-baseline"
    master: str | None = "local[*]"
    shuffle_partitions: int | None = None
    executor_memory: str | None = None
    driver_memory: str | None = None
    local_dir: str | None = None
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    auto_tune: bool = True


@dataclass(frozen=True)
class M5FeatureConfig:
    """Feature engineering options for the M5 dataset."""

    forecast_horizon: int = 28
    validation_days: int = 28
    limit_days: int | None = None
    output_partitions: int | None = None
    unpivot_chunk_size: int = 16
    lags: tuple[int, ...] = (7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 28)
    auto_partitions: bool = False
