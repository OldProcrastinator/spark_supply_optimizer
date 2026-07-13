from spark_supply_optimizer.config import M5FeatureConfig
from spark_supply_optimizer.m5 import (
    CATEGORICAL_COLUMNS,
    auto_output_partitions,
    build_sales_history,
    chunks,
    day_columns,
    feature_columns,
    history_stack_expression,
    join_uri,
    require_spark_path,
    stack_expression,
)


def test_day_columns_are_sorted_numerically() -> None:
    columns = ["id", "d_10", "d_2", "store_id", "d_1"]

    assert day_columns(columns) == ["d_1", "d_2", "d_10"]


def test_stack_expression_contains_all_day_pairs() -> None:
    expression = stack_expression(["d_1", "d_2"])

    assert expression == "stack(2, 'd_1', `d_1`, 'd_2', `d_2`) as (d, sales)"


def test_history_stack_expression_derives_features_from_wide_columns() -> None:
    config = M5FeatureConfig(lags=(1,), rolling_windows=(2,))

    expression = history_stack_expression(["d_3"], config)

    assert "lag_1" in expression
    assert "rolling_mean_2" in expression
    assert "CAST(`d_2` AS DOUBLE)" in expression
    assert "CAST(`d_1` AS DOUBLE) + CAST(`d_2` AS DOUBLE)" in expression


def test_chunks_split_large_unpivot_into_bounded_plans() -> None:
    values = ["d_1", "d_2", "d_3", "d_4", "d_5"]

    assert chunks(values, 2) == [["d_1", "d_2"], ["d_3", "d_4"], ["d_5"]]


def test_feature_columns_follow_config() -> None:
    config = M5FeatureConfig(lags=(1, 7), rolling_windows=(3,))

    assert "item_hash" in feature_columns(config)
    assert "lag_1" in feature_columns(config)
    assert "lag_7" in feature_columns(config)
    assert "rolling_mean_3" in feature_columns(config)


def test_high_cardinality_item_id_is_not_tree_categorical_feature() -> None:
    assert "item_id" not in CATEGORICAL_COLUMNS


def test_join_uri_preserves_s3a_scheme() -> None:
    assert join_uri("s3a://m5/data/", "calendar.csv") == "s3a://m5/data/calendar.csv"


def test_require_spark_path_explains_missing_prepare_output(monkeypatch) -> None:
    monkeypatch.setattr("spark_supply_optimizer.m5.spark_path_exists", lambda *_: False)

    try:
        require_spark_path(object(), "s3a://m5-data/artifacts/features", "m5-spark prepare")
    except RuntimeError as error:
        assert "s3a://m5-data/artifacts/features" in str(error)
        assert "m5-spark prepare" in str(error)
    else:
        raise AssertionError("Expected missing Spark path to raise RuntimeError.")


def test_auto_output_partitions_scales_with_spark_parallelism() -> None:
    class FakeSparkContext:
        defaultParallelism = 20

    class FakeSpark:
        sparkContext = FakeSparkContext()

    assert auto_output_partitions(FakeSpark()) == 80


def test_build_sales_history_plan_does_not_use_window_sort(spark) -> None:
    sales = spark.createDataFrame(
        [
            ("item_1", "item", "dept", "cat", "store", "state", 1, 2, 3),
            ("item_2", "item", "dept", "cat", "store", "state", 2, 4, 6),
        ],
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d_1", "d_2", "d_3"],
    )

    history = build_sales_history(
        sales,
        ["d_1", "d_2", "d_3"],
        M5FeatureConfig(lags=(1,), rolling_windows=(2,), unpivot_chunk_size=2),
    )

    assert "Window" not in history._jdf.queryExecution().analyzed().toString()  # noqa: SLF001
    assert "Sort" not in history._jdf.queryExecution().executedPlan().toString()  # noqa: SLF001
