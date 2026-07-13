"""M5 dataset loading, feature engineering, training, and inference."""

from __future__ import annotations

import time
from functools import reduce
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_supply_optimizer.config import M5FeatureConfig
from spark_supply_optimizer.metrics import regression_metrics

ID_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
CATEGORICAL_COLUMNS = ("dept_id", "cat_id", "store_id", "state_id")
BASE_NUMERIC_FEATURES = (
    "item_hash",
    "day_number",
    "wday",
    "month",
    "year",
    "snap",
    "sell_price",
)
M5_INPUT_FILES = (
    "sales_train_validation.csv",
    "calendar.csv",
    "sell_prices.csv",
)


def join_uri(root: str | Path, name: str) -> str:
    """Join local paths and Spark URI paths without breaking schemes like s3a://."""

    root_text = str(root).rstrip("/")
    if "://" in root_text:
        return f"{root_text}/{name}"
    return str(Path(root_text) / name)


def day_columns(columns: list[str]) -> list[str]:
    """Return M5 day columns sorted by their numeric day index."""

    return sorted(
        (column for column in columns if column.startswith("d_")),
        key=lambda value: int(value[2:]),
    )


def stack_expression(day_cols: list[str]) -> str:
    """Build a Spark SQL stack expression for wide-to-long conversion."""

    pairs = ", ".join(f"'{column}', `{column}`" for column in day_cols)
    return f"stack({len(day_cols)}, {pairs}) as (d, sales)"


def history_stack_expression(day_cols: list[str], config: M5FeatureConfig) -> str:
    """Build stack expression that derives history features from wide M5 day columns."""

    output_columns = [
        "d",
        "sales",
        *(f"lag_{lag}" for lag in config.lags),
        *(f"rolling_mean_{window}" for window in config.rolling_windows),
    ]
    rows = []
    for column in day_cols:
        day_number = int(column[2:])
        values = [f"'{column}'", f"CAST(`{column}` AS DOUBLE)"]
        for lag in config.lags:
            lag_column = f"d_{day_number - lag}"
            values.append(
                f"CAST(`{lag_column}` AS DOUBLE)" if day_number > lag else "CAST(NULL AS DOUBLE)"
            )
        for window_size in config.rolling_windows:
            window_columns = [
                f"`d_{day}`" for day in range(day_number - window_size, day_number) if day > 0
            ]
            if len(window_columns) < window_size:
                values.append("CAST(NULL AS DOUBLE)")
                continue
            total = " + ".join(f"CAST({value} AS DOUBLE)" for value in window_columns)
            values.append(f"(({total}) / {float(window_size)})")
        rows.append(", ".join(values))

    return f"stack({len(day_cols)}, {', '.join(rows)}) as ({', '.join(output_columns)})"


def chunks(values: list[str], size: int) -> list[list[str]]:
    """Split values into non-empty chunks."""

    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [values[index : index + size] for index in range(0, len(values), size)]


def unpivot_sales(sales: DataFrame, selected_days: list[str], chunk_size: int) -> DataFrame:
    """Convert M5 wide sales columns into long format without huge Spark codegen plans."""

    frames = [
        sales.select(*ID_COLUMNS, F.expr(stack_expression(day_chunk)))
        for day_chunk in chunks(selected_days, chunk_size)
    ]
    return reduce(lambda left, right: left.unionByName(right), frames)


def load_raw_inputs(
    spark: SparkSession,
    data_dir: str | Path,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Load raw M5 CSV files with Spark schema inference."""

    sales = spark.read.csv(
        join_uri(data_dir, "sales_train_validation.csv"),
        header=True,
        inferSchema=True,
    )
    calendar = spark.read.csv(join_uri(data_dir, "calendar.csv"), header=True, inferSchema=True)
    prices = spark.read.csv(join_uri(data_dir, "sell_prices.csv"), header=True, inferSchema=True)
    return sales, calendar, prices


def spark_path_exists(spark: SparkSession, path: str | Path) -> bool:
    """Return whether a local/Hadoop/S3A path exists from Spark's point of view."""

    hadoop_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(str(path))  # noqa: SLF001
    filesystem = hadoop_path.getFileSystem(  # noqa: SLF001
        spark.sparkContext._jsc.hadoopConfiguration()  # noqa: SLF001
    )
    return bool(filesystem.exists(hadoop_path))


def require_spark_path(spark: SparkSession, path: str | Path, producer_command: str) -> None:
    """Fail with a short actionable message when a required Spark path is missing."""

    if spark_path_exists(spark, path):
        return

    raise RuntimeError(
        f"Required path does not exist: {path}. Run `{producer_command}` first and wait "
        "until it finishes successfully."
    )


def build_features(
    sales: DataFrame,
    calendar: DataFrame,
    prices: DataFrame,
    config: M5FeatureConfig,
) -> DataFrame:
    """Create the distributed training table used by Spark ML."""

    selected_days = day_columns(sales.columns)
    if config.limit_days is not None:
        selected_days = selected_days[-config.limit_days :]

    return build_features_for_days(sales, calendar, prices, config, selected_days)


def build_features_for_days(
    sales: DataFrame,
    calendar: DataFrame,
    prices: DataFrame,
    config: M5FeatureConfig,
    selected_days: list[str],
) -> DataFrame:
    """Create features for a bounded set of M5 days."""

    long_sales = build_sales_history(sales, selected_days, config)

    calendar_features = (
        calendar.select(
            "d",
            "wm_yr_wk",
            "wday",
            "month",
            "year",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        )
        .withColumn(
            "snap",
            F.when(F.col("snap_CA") == 1, F.lit(1))
            .when(F.col("snap_TX") == 1, F.lit(1))
            .when(F.col("snap_WI") == 1, F.lit(1))
            .otherwise(F.lit(0)),
        )
        .drop("snap_CA", "snap_TX", "snap_WI")
    )
    selected_calendar = calendar_features.filter(F.col("d").isin(selected_days))
    selected_weeks = selected_calendar.select("wm_yr_wk").distinct()
    selected_prices = prices.join(F.broadcast(selected_weeks), on="wm_yr_wk", how="inner")

    features = (
        long_sales.join(F.broadcast(selected_calendar), on="d", how="left")
        .join(F.broadcast(selected_prices), on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .fillna({"sell_price": 0.0, "snap": 0})
        .withColumn("item_hash", F.pmod(F.xxhash64("item_id"), F.lit(1024)).cast("double"))
    )

    required_feature_columns = feature_columns(config)
    return features.dropna(subset=required_feature_columns + ["sales"])


def build_sales_history(
    sales: DataFrame,
    selected_days: list[str],
    config: M5FeatureConfig,
) -> DataFrame:
    """Build lag and rolling features without a distributed window sort."""

    frames = [
        sales.select(*ID_COLUMNS, F.expr(history_stack_expression(day_chunk, config)))
        for day_chunk in chunks(selected_days, config.unpivot_chunk_size)
    ]
    return reduce(lambda left, right: left.unionByName(right), frames).withColumn(
        "day_number",
        F.regexp_extract("d", r"d_(\d+)", 1).cast("int"),
    )


def feature_columns(config: M5FeatureConfig) -> list[str]:
    """Return the numeric feature columns expected by the model pipeline."""

    return [
        *BASE_NUMERIC_FEATURES,
        *(f"lag_{lag}" for lag in config.lags),
        *(f"rolling_mean_{window}" for window in config.rolling_windows),
    ]


def split_train_validation(
    features: DataFrame,
    validation_days: int,
) -> tuple[DataFrame, DataFrame]:
    """Split by the last N available days to avoid time leakage."""

    max_day = features.agg(F.max("day_number").alias("max_day")).first()["max_day"]
    cutoff = int(max_day) - validation_days
    train = features.filter(F.col("day_number") <= cutoff)
    validation = features.filter(F.col("day_number") > cutoff)
    return train, validation


def build_model_pipeline(config: M5FeatureConfig, max_depth: int, num_trees: int) -> Pipeline:
    """Build a transparent Spark ML pipeline for the baseline regressor."""

    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in CATEGORICAL_COLUMNS
    ]
    assembler = VectorAssembler(
        inputCols=[*feature_columns(config), *(f"{column}_idx" for column in CATEGORICAL_COLUMNS)],
        outputCol="features",
        handleInvalid="keep",
    )
    regressor = RandomForestRegressor(
        labelCol="sales",
        featuresCol="features",
        predictionCol="prediction",
        maxDepth=max_depth,
        numTrees=num_trees,
        seed=42,
    )
    return Pipeline(stages=[*indexers, assembler, regressor])


def prepare_features(
    spark: SparkSession,
    data_dir: str | Path,
    output_dir: str | Path,
    config: M5FeatureConfig,
) -> None:
    """Materialize engineered features as Parquet."""

    sales, calendar, prices = load_raw_inputs(spark, data_dir)
    selected_days = day_columns(sales.columns)
    if config.limit_days is not None:
        selected_days = selected_days[-config.limit_days :]

    output_partitions = config.output_partitions
    if output_partitions is None and config.auto_partitions:
        output_partitions = auto_output_partitions(spark)

    write_mode = "overwrite"
    for day_chunk in chunks(selected_days, config.unpivot_chunk_size):
        features = build_features_for_days(sales, calendar, prices, config, day_chunk)
        if output_partitions is not None:
            features = features.repartition(output_partitions)
        features.write.mode(write_mode).parquet(str(output_dir))
        write_mode = "append"


def auto_output_partitions(spark: SparkSession) -> int:
    """Choose enough output partitions for full M5 without reducing CPU parallelism."""

    parallelism = max(1, int(spark.sparkContext.defaultParallelism))
    return max(32, parallelism * 4)


def train_model(
    spark: SparkSession,
    features_dir: str | Path,
    model_dir: str | Path,
    config: M5FeatureConfig,
    max_depth: int,
    num_trees: int,
) -> dict[str, float]:
    """Train the baseline model and save it as a Spark ML PipelineModel."""

    require_spark_path(spark, features_dir, "m5-spark prepare")
    features = spark.read.parquet(str(features_dir))
    train, validation = split_train_validation(features, config.validation_days)
    model = build_model_pipeline(config, max_depth=max_depth, num_trees=num_trees).fit(train)
    predictions = model.transform(validation)
    metrics = regression_metrics(predictions)
    model.write().overwrite().save(str(model_dir))
    return metrics


def predict_latest(
    spark: SparkSession,
    features_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    horizon: int,
) -> None:
    """Score the last available days as a simple inference smoke path."""

    require_spark_path(spark, features_dir, "m5-spark prepare")
    require_spark_path(spark, model_dir, "m5-spark train")
    features = spark.read.parquet(str(features_dir))
    max_day = features.agg(F.max("day_number").alias("max_day")).first()["max_day"]
    scoring = features.filter(F.col("day_number") > int(max_day) - horizon)
    model = PipelineModel.load(str(model_dir))
    (
        model.transform(scoring)
        .select("id", "d", F.greatest(F.col("prediction"), F.lit(0.0)).alias("prediction"))
        .write.mode("overwrite")
        .csv(str(output_dir), header=True)
    )


def benchmark_feature_scan(spark: SparkSession, features_dir: str | Path) -> dict[str, object]:
    """Run a repeatable Spark job that is useful for one-node vs cluster timing."""

    require_spark_path(spark, features_dir, "m5-spark prepare")
    features = spark.read.parquet(str(features_dir))
    input_partitions = features.rdd.getNumPartitions()
    started = time.perf_counter()
    summary = features.agg(
        F.count("*").alias("input_rows"),
    ).first()
    aggregation_groups = (
        features.groupBy("store_id", "cat_id")
        .agg(F.count("*").alias("rows"), F.avg("sales").alias("avg_sales"))
        .count()
    )
    elapsed = time.perf_counter() - started
    return {
        "workload": {
            "input_rows": int(summary["input_rows"]),
            "input_partitions": int(input_partitions),
            "aggregation_groups": int(aggregation_groups),
            "aggregation_groups_note": "store_id x cat_id groups, not machines",
        },
        "benchmark_duration_seconds": elapsed,
    }
