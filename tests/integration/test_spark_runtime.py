from pathlib import Path

from spark_supply_optimizer.config import SparkRuntimeConfig
from spark_supply_optimizer.spark import build_spark


def test_local_spark_can_write_and_read_parquet(tmp_path: Path) -> None:
    """Smoke-test the actual local Spark runtime, including Java and Hadoop helpers."""

    spark = build_spark(
        SparkRuntimeConfig(
            app_name="spark-runtime-smoke-test",
            master="local[2]",
            shuffle_partitions=2,
        )
    )
    try:
        output_dir = tmp_path / "parquet_smoke"
        spark.createDataFrame([(1, "a"), (2, "b")], ["id", "label"]).write.mode(
            "overwrite"
        ).parquet(str(output_dir))

        rows = spark.read.parquet(str(output_dir)).orderBy("id").collect()

        assert [(row.id, row.label) for row in rows] == [(1, "a"), (2, "b")]
    finally:
        spark.stop()
