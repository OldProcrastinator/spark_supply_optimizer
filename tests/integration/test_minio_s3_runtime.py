import os
import uuid
from pathlib import Path

import pytest

from spark_supply_optimizer.config import M5FeatureConfig, SparkRuntimeConfig
from spark_supply_optimizer.m5 import prepare_features, train_model
from spark_supply_optimizer.spark import build_spark


def write_synthetic_m5_inputs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    days = [f"d_{day}" for day in range(1, 36)]
    sales_header = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *days]
    sales_rows = []
    for index in range(12):
        values = [str((index + day) % 7) for day in range(1, 36)]
        sales_rows.append(
            [
                f"item_{index}_store_1_validation",
                f"item_{index}",
                "dept_1",
                "cat_1",
                "store_1",
                "state_1",
                *values,
            ]
        )
    (root / "sales_train_validation.csv").write_text(
        "\n".join([",".join(sales_header), *(",".join(row) for row in sales_rows)]) + "\n",
        encoding="utf-8",
    )

    calendar_rows = [
        "d,wm_yr_wk,wday,month,year,snap_CA,snap_TX,snap_WI",
        *(
            f"d_{day},1001,{((day - 1) % 7) + 1},1,2011,0,0,0"
            for day in range(1, 36)
        ),
    ]
    (root / "calendar.csv").write_text("\n".join(calendar_rows) + "\n", encoding="utf-8")

    price_rows = ["store_id,item_id,wm_yr_wk,sell_price"]
    price_rows.extend(f"store_1,item_{index},1001,1.0" for index in range(12))
    (root / "sell_prices.csv").write_text("\n".join(price_rows) + "\n", encoding="utf-8")


def test_spark_can_write_and_read_parquet_through_minio() -> None:
    bucket = os.environ.get("MINIO_BUCKET", "m5-data")
    endpoint = os.environ.get("MINIO_INTEGRATION_ENDPOINT", "http://minio:9000")
    output_dir = f"s3a://{bucket}/integration-tests/parquet-{uuid.uuid4()}"

    spark = build_spark(
        SparkRuntimeConfig(
            app_name="spark-minio-smoke-test",
            master="local[2]",
            shuffle_partitions=2,
            s3_endpoint=endpoint,
            s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        )
    )
    try:
        spark.createDataFrame([(1, "from-minio"), (2, "from-s3a")], ["id", "label"]).write.mode(
            "overwrite"
        ).parquet(output_dir)

        rows = spark.read.parquet(output_dir).orderBy("id").collect()

        assert [(row.id, row.label) for row in rows] == [
            (1, "from-minio"),
            (2, "from-s3a"),
        ]
    finally:
        spark.stop()


def test_train_fails_cleanly_when_minio_features_are_missing() -> None:
    bucket = os.environ.get("MINIO_BUCKET", "m5-data")
    endpoint = os.environ.get("MINIO_INTEGRATION_ENDPOINT", "http://minio:9000")
    missing_features_dir = f"s3a://{bucket}/integration-tests/missing-features-{uuid.uuid4()}"
    model_dir = f"s3a://{bucket}/integration-tests/model-{uuid.uuid4()}"

    spark = build_spark(
        SparkRuntimeConfig(
            app_name="spark-minio-missing-features-test",
            master="local[2]",
            shuffle_partitions=2,
            s3_endpoint=endpoint,
            s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        )
    )
    try:
        with pytest.raises(RuntimeError, match="m5-spark prepare") as error:
            train_model(
                spark,
                missing_features_dir,
                model_dir,
                M5FeatureConfig(),
                max_depth=1,
                num_trees=1,
            )

        assert missing_features_dir in str(error.value)
    finally:
        spark.stop()


def test_prepare_writes_features_to_minio_without_window_shuffle_oom(tmp_path: Path) -> None:
    bucket = os.environ.get("MINIO_BUCKET", "m5-data")
    endpoint = os.environ.get("MINIO_INTEGRATION_ENDPOINT", "http://minio:9000")
    local_data_dir = tmp_path / "synthetic_m5"
    s3_prefix = f"integration-tests/prepare-{uuid.uuid4()}"
    s3_data_dir = f"s3a://{bucket}/{s3_prefix}/data"
    s3_features_dir = f"s3a://{bucket}/{s3_prefix}/features"

    write_synthetic_m5_inputs(local_data_dir)

    spark = build_spark(
        SparkRuntimeConfig(
            app_name="spark-minio-prepare-test",
            master="local[*]",
            shuffle_partitions=4,
            s3_endpoint=endpoint,
            s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        )
    )
    try:
        hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()  # noqa: SLF001
        for path in local_data_dir.iterdir():
            target_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(  # noqa: SLF001
                f"{s3_data_dir}/{path.name}"
            )
            spark.sparkContext._jvm.org.apache.hadoop.fs.FileUtil.copy(  # noqa: SLF001
                spark.sparkContext._jvm.java.io.File(str(path)),  # noqa: SLF001
                target_path.getFileSystem(hadoop_config),
                target_path,
                False,
                hadoop_config,
            )

        prepare_features(
            spark,
            s3_data_dir,
            s3_features_dir,
            M5FeatureConfig(lags=(1, 7), rolling_windows=(2,), unpivot_chunk_size=8),
        )

        rows = spark.read.parquet(s3_features_dir).count()
        assert rows > 0
    finally:
        spark.stop()


def test_prepare_real_uploaded_m5_sample_through_minio_without_write_shuffle_oom() -> None:
    bucket = os.environ.get("MINIO_BUCKET", "m5-data")
    endpoint = os.environ.get("MINIO_INTEGRATION_ENDPOINT", "http://minio:9000")
    s3_data_dir = f"s3a://{bucket}/data"
    s3_features_dir = f"s3a://{bucket}/integration-tests/real-prepare-{uuid.uuid4()}/features"

    spark = build_spark(
        SparkRuntimeConfig(
            app_name="spark-minio-real-prepare-test",
            master="local[*]",
            s3_endpoint=endpoint,
            s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
            local_dir=os.environ.get("SPARK_LOCAL_DIR", "artifacts/spark-tmp"),
        )
    )
    try:
        prepare_features(
            spark,
            s3_data_dir,
            s3_features_dir,
            M5FeatureConfig(limit_days=128),
        )

        features = spark.read.parquet(s3_features_dir)
        assert features.count() > 0
        assert features.rdd.getNumPartitions() > 0
    finally:
        spark.stop()
