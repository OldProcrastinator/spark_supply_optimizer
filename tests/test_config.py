from spark_supply_optimizer.config import M5FeatureConfig, SparkRuntimeConfig


def test_default_feature_config_is_small_and_explicit() -> None:
    config = M5FeatureConfig()

    assert config.forecast_horizon == 28
    assert config.validation_days == 28
    assert config.output_partitions is None
    assert config.unpivot_chunk_size == 16
    assert config.lags == (7, 14, 28)
    assert config.auto_partitions is False


def test_runtime_config_accepts_cluster_master() -> None:
    config = SparkRuntimeConfig(master="spark://100.64.0.10:7077", shuffle_partitions=12)

    assert config.master == "spark://100.64.0.10:7077"
    assert config.shuffle_partitions == 12


def test_runtime_config_accepts_minio_endpoint() -> None:
    config = SparkRuntimeConfig(
        s3_endpoint="http://192.168.1.10:9000",
        s3_access_key="minioadmin",
        s3_secret_key="minioadmin123",
    )

    assert config.s3_endpoint == "http://192.168.1.10:9000"


def test_runtime_config_accepts_local_spill_directory() -> None:
    config = SparkRuntimeConfig(local_dir="/app/artifacts/spark-tmp")

    assert config.local_dir == "/app/artifacts/spark-tmp"


def test_runtime_config_auto_tune_is_enabled_by_default() -> None:
    assert SparkRuntimeConfig().auto_tune is True
